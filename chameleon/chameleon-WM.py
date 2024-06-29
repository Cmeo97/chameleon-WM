import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast

from chameleon.inference.chameleon import ChameleonInferenceModel

class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, transformer_hidden_dim, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim*stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim*stoch_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits


class RewardDecoder(nn.Module):
    def __init__(self, num_classes, embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(transformer_hidden_dim, num_classes)

    def forward(self, feat):
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward


class TerminationDecoder(nn.Module):
    def __init__(self,  embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits, z_mask=None):
        # z_mask corresponds to the masking applied in maskgit
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist)
        if z_mask is not None:
            kl_div = kl_div * z_mask
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div


class WorldModel(nn.Module):
    def __init__(self, in_channels, action_dim,
                 transformer_max_length, transformer_hidden_dim, transformer_num_layers, transformer_num_heads, conf):
        super().__init__()
        self.model = ChameleonInferenceModel(
        "./data/models/7b/",
        "./data/tokenizer/text_tokenizer.json",
        "./data/tokenizer/vqgan.yaml",
        "./data/tokenizer/vqgan.ckpt",
        )

        #self.reward_decoder = RewardDecoder(
        #    num_classes=255,
        #    embedding_size=self.stoch_flattened_dim,
        #    transformer_hidden_dim=transformer_hidden_dim
        #)
        #self.termination_decoder = TerminationDecoder(
        #    embedding_size=self.stoch_flattened_dim,
        #    transformer_hidden_dim=transformer_hidden_dim
        #)
        #self.maskgit = MaskGit(
        #    shape=None,
        #    vocab_size=conf.Models.MaskGit.VocabSize,
        #    vocab_dim=conf.Models.MaskGit.VocabDim,
        #    mask_schedule=conf.Models.MaskGit.MaskSchedule,
        #    tfm_kwargs={
        #        "embed_dim": conf.Models.MaskGit.TmfArgs.EmbedDim,
        #        "mlp_dim": conf.Models.MaskGit.TmfArgs.MlpDim,
        #        "num_heads": conf.Models.MaskGit.TmfArgs.NumHeads,
        #        "num_layers": conf.Models.MaskGit.TmfArgs.NumLayers,
        #        "dropout": conf.Models.MaskGit.TmfArgs.Dropout,
        #        "attention_dropout": conf.Models.MaskGit.TmfArgs.AttentionDropout,
        #        "vocab_dim": conf.Models.MaskGit.VocabDim,
        #        "input_dim": self.transformer_hidden_dim // self.stoch_dim,
        #    }
        #)

        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.T_draft = conf.Models.MaskGit.T_draft
        self.T_revise = conf.Models.MaskGit.T_revise
        self.M = conf.Models.MaskGit.M

    def encode_obs(self, obs):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
        return flattened_sample

    def calc_last_dist_feat(self, latent, action):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            rdist = rearrange(last_dist_feat, "B 1 (K C) -> (B 1) K C", K=self.stoch_dim)
            sample_shape=(self.stoch_dim)
            prior_sample =  self.maskgit.sample(rdist.shape[0], self.T_draft, self.T_revise, self.M, cond=rdist, sample_shape=sample_shape)
            prior_sample = rearrange(prior_sample, "(B 1) K -> B 1 K", B=rdist.shape[0])
            prior_sample = F.one_hot(prior_sample, num_classes=self.stoch_dim).float()
            prior_flattened_sample = self.flatten_sample(prior_sample)
        return prior_flattened_sample, last_dist_feat

    def predict_next(self, last_flattened_sample, action, log_video=True):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist_feat = self.storm_transformer.forward_with_kv_cache(last_flattened_sample, action)
            sample_shape=(self.stoch_dim)
            rdist = rearrange(dist_feat, "B 1 (K C) -> (B 1) K C", K=self.stoch_dim)
            prior_sample =  self.maskgit.sample(rdist.shape[0], self.T_draft, self.T_revise, self.M, cond=rdist, sample_shape=sample_shape)
            prior_sample = rearrange(prior_sample, "(B 1) K -> B 1 K", B=rdist.shape[0])
            prior_sample = F.one_hot(prior_sample, num_classes=self.stoch_dim).float()

            # decoding
            prior_flattened_sample = self.flatten_sample(prior_sample)
            if log_video:
                obs_hat = self.image_decoder(prior_flattened_sample)
            else:
                obs_hat = None
            reward_hat = self.reward_decoder(dist_feat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, dist_feat

    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample

    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_flattened_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, self.transformer_hidden_dim)
            scalar_size = (imagine_batch_size, imagine_batch_length)
            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device="cuda")
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device="cuda")
            self.action_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")

    def imagine_data(self, agent: agents.ActorCriticAgent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video, logger):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype)
        obs_hat_list = []

        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)
        context_latent = self.encode_obs(sample_obs)
        for i in range(sample_obs.shape[1]):
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1],
                log_video=log_video
            )
        self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_dist_feat

        for i in range(imagine_batch_length):
            action = agent.sample(torch.cat([self.latent_buffer[:, i:i+1], self.hidden_buffer[:, i:i+1]], dim=-1))
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.latent_buffer[:, i:i+1], self.action_buffer[:, i:i+1], log_video=log_video)

            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_dist_feat
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            if log_video:
                obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env

        if log_video:
            logger.log("Imagine/predict_video", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).cpu().float().detach().numpy())

        return torch.cat([self.latent_buffer, self.hidden_buffer], dim=-1), self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer

    #def update(self, obs, action, reward, termination, logger=None):
    #    self.train()
    #    batch_size, batch_length = obs.shape[:2]
#
    #    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
    #        embedding = self.encoder(obs)
    #        post_logits = self.dist_head.forward_post(embedding)
    #        sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
    #        sample_encodings = torch.argmax(sample, dim=-1)
    #        flattened_sample = self.flatten_sample(sample)
#
    #        # decoding image
    #        obs_hat = self.image_decoder(flattened_sample)
#
    #        # transformer
    #        temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device)
    #        dist_feat = self.storm_transformer(flattened_sample, action, temporal_mask)
    #        # chage the shape of dist_feat such that it matches the shape required by maskgit
    #        dist_feat = rearrange(dist_feat, "B L (K C) -> (B L) K C", K=self.stoch_dim)
    #        sample_encodings = rearrange(sample_encodings, "B L K -> (B L) K")
    #        z_logits, z_labels, z_mask = self.maskgit(sample_encodings, dist_feat)
    #        z_logits = rearrange(z_logits, "(B L) K C -> B L K C", B=batch_size) # bring back to original shape
    #        z_logits = self.dist_head.unimix(z_logits) # NOTE: add unimix to the logits
    #        z_labels = rearrange(z_labels, "(B L) K C -> B L K C", B=batch_size) # bring back to original shape
    #        z_mask = rearrange(z_mask, "(B L) K -> B L K", B=batch_size) # bring back to original shape
    #        dist_feat = rearrange(dist_feat, "(B L) K C -> B L (K C)", B=batch_size) # bring back to original shape
    #        # decoding reward and termination with dist_feat
    #        reward_hat = self.reward_decoder(dist_feat)
    #        termination_hat = self.termination_decoder(dist_feat)
#
    #        # env loss
    #        reconstruction_loss = self.mse_loss_func(obs_hat, obs)
    #        reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
    #        termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)
    #        # dyn-rep loss
    #        dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), z_logits[:, :-1], z_mask[:, :-1]) 
    #        representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], z_logits[:, :-1].detach(), z_mask[:, :-1].detach())
    #        total_loss = reconstruction_loss + reward_loss + termination_loss + 0.5*dynamics_loss + 0.1*representation_loss
#
    #    # gradient descent
    #    self.scaler.scale(total_loss).backward()
    #    self.scaler.unscale_(self.optimizer)  # for clip grad
    #    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
    #    self.scaler.step(self.optimizer)
    #    self.scaler.update()
    #    self.optimizer.zero_grad(set_to_none=True)
#
    #    if logger is not None:
    #        logger.log("WorldModel/reconstruction_loss", reconstruction_loss.item())
    #        logger.log("WorldModel/reward_loss", reward_loss.item())
    #        logger.log("WorldModel/termination_loss", termination_loss.item())
    #        logger.log("WorldModel/dynamics_loss", dynamics_loss.item())
    #        logger.log("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item())
    #        logger.log("WorldModel/representation_loss", representation_loss.item())
    #        logger.log("WorldModel/representation_real_kl_div", representation_real_kl_div.item())
    #        logger.log("WorldModel/total_loss", total_loss.item())
#
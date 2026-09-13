import math
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput

from models.archs import define_network
from models.base_model import BaseModel

loss_module = __import__('models.losses', fromlist=['losses'])


class FixedStepDDIMScheduler(DDIMScheduler):
    """DDIM scheduler whose step can use an explicitly scheduled predecessor."""

    def step(self, model_output, timestep, sample, previous_timestep=None):
        if previous_timestep is None:
            return super().step(model_output, timestep, sample, eta=0.0)
        alpha_prod_t = self.alphas_cumprod[timestep].to(
            device=sample.device, dtype=sample.dtype)
        alpha_prod_t_previous = self.alphas_cumprod[previous_timestep].to(
            device=sample.device, dtype=sample.dtype)
        beta_prod_t = 1 - alpha_prod_t
        prediction_type = self.config.prediction_type
        if prediction_type == 'epsilon':
            pred_original_sample = (
                sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
            pred_epsilon = model_output
        elif prediction_type == 'sample':
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t.sqrt() * pred_original_sample) / beta_prod_t.sqrt()
        elif prediction_type == 'v_prediction':
            pred_original_sample = (
                alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output)
            pred_epsilon = (
                alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample)
        else:
            raise ValueError(f'Unsupported DDIM prediction type: {prediction_type}')
        previous_sample = (
            alpha_prod_t_previous.sqrt() * pred_original_sample +
            (1 - alpha_prod_t_previous).sqrt() * pred_epsilon)
        return DDIMSchedulerOutput(
            prev_sample=previous_sample, pred_original_sample=pred_original_sample)


class LearnedTimestep(nn.Module):
    """A globally learned timestep constrained to the scheduler range."""

    def __init__(self, initial_timestep):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(
            float(initial_timestep), dtype=torch.float32))

    def forward(self, batch_size):
        return 499.5 * (torch.tanh(self.value) + 1.0).expand(batch_size)


class XStartPromptTokens(nn.Module):
    """Convert a 4x64x64 x-start latent into 22 additional prompt tokens."""

    def __init__(self, prompt_dim=768, grid_rows=2, grid_columns=11):
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_columns = grid_columns
        self.region_size = 4 * math.ceil(64 / grid_rows) * math.ceil(
            64 / grid_columns)
        self.projection = nn.Linear(self.region_size, prompt_dim)

    def forward(self, xstart):
        batch_size, channels, height, width = xstart.shape
        if (channels, height, width) != (4, 64, 64):
            raise ValueError(
                'Predicted x_start prompt tokens require [B, 4, 64, 64] latents.')
        rows = torch.tensor_split(xstart, self.grid_rows, dim=2)
        tokens = []
        for row in rows:
            for region in torch.tensor_split(row, self.grid_columns, dim=3):
                flattened = region.flatten(1)
                tokens.append(F.pad(
                    flattened, (0, self.region_size - flattened.shape[1])))
        return self.projection(torch.stack(tokens, dim=1))


class ReverseDistillLearnedTModel(BaseModel):
    """Reverse distillation with a trainable global timestep and x-t regression."""

    def __init__(self, opt):
        super().__init__(opt)
        self.cfg_sacle = opt.get('cfg_sacle')
        self.ema_decay = opt.get('train', {}).get('ema_decay', 0.95)
        self.use_pred_xstart_prompt = opt.get('use_pred_xstart_prompt', False)
        self.use_sd_inference = opt.get('use_sd_inference', False)
        self.sd_inference_timesteps = tuple(
            opt.get('sd_inference_timesteps', [100, 150, 200]))
        self.prompt_dim = opt.get('prompt_dim', 768)
        self.requires_unconditional_prompt_embeddings = True
        self.sd_model_id = opt.get('sd_model_id', 'sd-legacy/stable-diffusion-v1-5')
        self.online_source_timestep = opt.get('online_source_timestep', 999)
        self.online_target_timestep = opt.get('online_target_timestep', 850)
        self.sd_unet = None
        self.sd_scheduler = None

        scheduler = DDPMScheduler.from_pretrained(
            'sd-legacy/stable-diffusion-v1-5', subfolder='scheduler')
        self.alpha_cumprods = scheduler.alphas_cumprod.to(
            device=self.device, dtype=torch.float32)
        if self.use_sd_inference:
            if (len(self.sd_inference_timesteps) != 3
                    or tuple(sorted(self.sd_inference_timesteps)) != (
                        self.sd_inference_timesteps)):
                raise ValueError(
                    'sd_inference_timesteps must contain three ascending timesteps.')
            if (self.sd_inference_timesteps[0] < 0
                    or self.sd_inference_timesteps[-1]
                    >= self.alpha_cumprods.numel()):
                raise ValueError('sd_inference_timesteps are outside the scheduler range.')

        xstart_opt = deepcopy(opt['network_xstart'])
        xt_opt = deepcopy(opt['network_xt'])
        self.net_xstart = self.model_to_device(define_network(xstart_opt))
        self.net_xt = self.model_to_device(define_network(xt_opt))
        self.net_xstart_ema = self.model_to_device(
            define_network(deepcopy(opt['network_xstart'])))
        self.net_xt_ema = self.model_to_device(
            define_network(deepcopy(opt['network_xt'])))
        self.xstart_prompt_tokens = None
        self.xstart_prompt_tokens_ema = None
        if self.use_pred_xstart_prompt:
            self.xstart_prompt_tokens = self.model_to_device(
                XStartPromptTokens(self.prompt_dim))
            self.xstart_prompt_tokens_ema = self.model_to_device(
                XStartPromptTokens(self.prompt_dim))
        self.sd_prompt_tokens = []
        self.sd_prompt_tokens_ema = []
        if self.use_sd_inference:
            self.sd_prompt_tokens = [
                self.model_to_device(XStartPromptTokens(self.prompt_dim))
                for _ in range(7)
            ]
            self.sd_prompt_tokens_ema = [
                self.model_to_device(XStartPromptTokens(self.prompt_dim))
                for _ in range(7)
            ]

        self.learned_timestep_module = self.model_to_device(
            LearnedTimestep(opt.get('learned_timestep_init', 250.0)))
        self._load_networks()
        self._sync_ema_if_not_loaded()
        self.get_bare_model(self.net_xstart_ema).train(False)
        self.get_bare_model(self.net_xt_ema).train(False)
        if self.is_train:
            self.init_training_settings()

    def _load_networks(self):
        path_opt = self.opt['path']
        network_specs = [
            (self.net_xstart, 'pretrain_network_xstart'),
            (self.net_xt, 'pretrain_network_xt'),
            (self.net_xstart_ema, 'pretrain_network_xstart_ema'),
            (self.net_xt_ema, 'pretrain_network_xt_ema'),
        ]
        for index, network in enumerate(self.sd_prompt_tokens):
            network_specs.append(
                (network, f'pretrain_sd_prompt_token_{index}'))
        for index, network in enumerate(self.sd_prompt_tokens_ema):
            network_specs.append(
                (network, f'pretrain_sd_prompt_token_{index}_ema'))
        self._loaded_ema = {}
        for network, path_key in network_specs:
            if path_opt.get(path_key) is not None:
                self.load_network(network, path_opt[path_key],
                                  path_opt.get('strict_load_g', True),
                                  param_key=path_opt.get('param_key', 'params'))
                self._loaded_ema[path_key] = True

    def _sync_ema_if_not_loaded(self):
        if not self._loaded_ema.get('pretrain_network_xstart_ema'):
            self.get_bare_model(self.net_xstart_ema).load_state_dict(
                self.get_bare_model(self.net_xstart).state_dict())
        if not self._loaded_ema.get('pretrain_network_xt_ema'):
            self.get_bare_model(self.net_xt_ema).load_state_dict(
                self.get_bare_model(self.net_xt).state_dict())
        if self.xstart_prompt_tokens is not None:
            self.get_bare_model(self.xstart_prompt_tokens_ema).load_state_dict(
                self.get_bare_model(self.xstart_prompt_tokens).state_dict())
        for index, (ema_tokens, tokens) in enumerate(zip(
                self.sd_prompt_tokens_ema, self.sd_prompt_tokens)):
            if not self._loaded_ema.get(f'pretrain_sd_prompt_token_{index}_ema'):
                self.get_bare_model(ema_tokens).load_state_dict(
                    self.get_bare_model(tokens).state_dict())

    def init_training_settings(self):
        train_opt = self.opt['train']
        self.cri_xstart = self._build_loss(train_opt.get('xstart_opt'))
        self.cri_xt = self._build_loss(train_opt.get('xt_opt'))
        self.cri_xT = self._build_loss(
            train_opt.get('xT_opt', train_opt.get('xt_opt')))
        self.xstart_loss_weight = train_opt.get('xstart_loss_weight', 1.0)
        self.xt_loss_weight = train_opt.get('xt_loss_weight', 1.0)
        self.xT_loss_weight = train_opt.get('xT_loss_weight', 1.0)
        if self.cri_xstart is None or self.cri_xt is None or self.cri_xT is None:
            raise ValueError('xstart_opt and xt_opt must be configured.')
        self.setup_optimizers()
        self.setup_schedulers()

    def _build_loss(self, loss_opt):
        loss_opt = deepcopy(loss_opt)
        loss_type = loss_opt.pop('type')
        return getattr(loss_module, loss_type)(**loss_opt).to(self.device)

    def _build_optimizer(self, optimizer_opt, parameters):
        optimizer_opt = deepcopy(optimizer_opt)
        optimizer_type = optimizer_opt.pop('type')
        return getattr(torch.optim, optimizer_type)(parameters, **optimizer_opt)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        xstart_parameters = list(self.net_xstart.parameters())
        if self.xstart_prompt_tokens is not None:
            xstart_parameters += list(self.xstart_prompt_tokens.parameters())
        self.optimizer_xstart = self._build_optimizer(
            train_opt['optim_xstart'], xstart_parameters)
        xt_parameters = (list(self.net_xt.parameters()) +
                         list(self.learned_timestep_module.parameters()))
        for prompt_tokens in self.sd_prompt_tokens:
            xt_parameters += list(prompt_tokens.parameters())
        self.optimizer_xt = self._build_optimizer(
            train_opt['optim_xt'], xt_parameters)
        self.optimizers.extend([self.optimizer_xstart, self.optimizer_xt])

    def feed_data(self, data, is_val=False):
        self.lq = data['final_latents'].to(self.device)
        self.gt = data['gt'].to(self.device)
        self.cond_xstart = data['cond_xstart'].to(self.device)
        self.uncond_xstart = data['uncond_xstart'].to(self.device)
        self.prompt_emb = data['prompt'].to(self.device)
        if self.cfg_sacle is None:
            self.xstart_target = self.cond_xstart
        else:
            self.xstart_target = self.uncond_xstart + self.cfg_sacle * (
                self.cond_xstart - self.uncond_xstart)

    def _load_stable_diffusion(self):
        if self.sd_unet is None:
            self.sd_unet = UNet2DConditionModel.from_pretrained(
                self.sd_model_id, subfolder='unet').to(self.device).eval()
            self.sd_unet.requires_grad_(False)
            self.sd_scheduler = FixedStepDDIMScheduler.from_pretrained(
                self.sd_model_id, subfolder='scheduler')
            self.sd_scheduler.set_timesteps(
                self.sd_scheduler.config.num_train_timesteps, device=self.device)

    @torch.no_grad()
    def prepare_online_batch(self, data):
        """Produce x_850 and fresh conditional/unconditional x-start targets."""
        if 'uncond_prompt' not in data:
            raise ValueError('Online reverse distillation requires uncond_prompt embeddings.')
        self._load_stable_diffusion()
        x_terminal = data['gt'].to(self.device)
        cond_xstart = data['cond_xstart'].to(self.device)
        uncond_xstart = data['uncond_xstart'].to(self.device)
        alpha_source = self.sd_scheduler.alphas_cumprod[
            self.online_source_timestep].to(device=self.device, dtype=x_terminal.dtype)
        source_noise_scale = (1 - alpha_source).sqrt()
        cond_epsilon = (x_terminal - alpha_source.sqrt() * cond_xstart) / source_noise_scale
        uncond_epsilon = (x_terminal - alpha_source.sqrt() * uncond_xstart) / source_noise_scale
        if self.cfg_sacle is None:
            epsilon = cond_epsilon
        else:
            epsilon = uncond_epsilon + self.cfg_sacle * (cond_epsilon - uncond_epsilon)
        x_target = self.sd_scheduler.step(
            epsilon, self.online_source_timestep, x_terminal,
            previous_timestep=self.online_target_timestep).prev_sample
        timestep = torch.full(
            (x_target.shape[0],), self.online_target_timestep,
            device=self.device, dtype=torch.long)
        alpha_target = self.sd_scheduler.alphas_cumprod[
            self.online_target_timestep].to(device=self.device, dtype=x_target.dtype)
        target_noise_scale = (1 - alpha_target).sqrt()
        cond_epsilon = self.sd_unet(
            x_target, timestep, encoder_hidden_states=data['prompt'].to(self.device)).sample
        uncond_epsilon = self.sd_unet(
            x_target, timestep,
            encoder_hidden_states=data['uncond_prompt'].to(self.device)).sample
        online_data = dict(data)
        # online_data['final_latents'] = x_target
        online_data['cond_xstart'] = (
            x_target - target_noise_scale * cond_epsilon) / alpha_target.sqrt()
        online_data['uncond_xstart'] = (
            x_target - target_noise_scale * uncond_epsilon) / alpha_target.sqrt()
        return online_data

    def learned_timestep(self, batch_size):
        return self.learned_timestep_module(batch_size)

    def _alpha_from_timestep(self, timestep):
        low_idx = timestep.floor().long().clamp(max=self.alpha_cumprods.numel() - 1)
        high_idx = (low_idx + 1).clamp(max=self.alpha_cumprods.numel() - 1)
        interpolation = timestep - low_idx.to(timestep.dtype)
        log_alpha = self.alpha_cumprods[low_idx].log() + interpolation * (
            self.alpha_cumprods[high_idx].log() - self.alpha_cumprods[low_idx].log())
        return log_alpha.exp()

    def _xt_prompt(self, pred_xstart, use_ema_model=False):
        if not self.use_pred_xstart_prompt:
            return self.prompt_emb
        prompt_encoder = (self.xstart_prompt_tokens_ema if use_ema_model
                          else self.xstart_prompt_tokens)
        latent_tokens = prompt_encoder(pred_xstart.detach())
        return torch.cat([self.prompt_emb, latent_tokens], dim=1)

    @torch.no_grad()
    def _sd_probe_features(self):
        """Return conditional SD epsilons and x-starts at three noisy levels."""
        self._load_stable_diffusion()
        epsilons = []
        xstarts = []
        for probe_timestep in self.sd_inference_timesteps:
            alpha_cumprod = self.sd_scheduler.alphas_cumprod[
                probe_timestep].to(device=self.device, dtype=self.lq.dtype)
            noise_scale = (1.0 - alpha_cumprod).sqrt()
            timestep = torch.full(
                (self.lq.shape[0],), probe_timestep,
                device=self.device, dtype=torch.long)
            noisy_lq = self.sd_scheduler.add_noise(
                self.lq, torch.randn_like(self.lq), timestep)
            epsilon = self.sd_unet(
                noisy_lq, timestep, encoder_hidden_states=self.prompt_emb).sample
            epsilons.append(epsilon)
            xstarts.append(
                (noisy_lq - noise_scale * epsilon) / alpha_cumprod.sqrt())
        return epsilons, xstarts

    def _sd_xt_prompt(self, epsilons, xstarts, use_ema_model=False):
        prompt_encoders = (self.sd_prompt_tokens_ema if use_ema_model
                           else self.sd_prompt_tokens)
        prompt_inputs = [*epsilons, *xstarts, self.lq]
        if len(prompt_encoders) != len(prompt_inputs):
            raise RuntimeError('SD prompt token encoders are not initialized.')
        latent_tokens = [
            self.get_bare_model(prompt_encoder)(prompt_input.detach())
            for prompt_encoder, prompt_input in zip(prompt_encoders, prompt_inputs)
        ]
        return torch.cat([self.prompt_emb, *latent_tokens], dim=1)

    def _predict(self, use_ema_model=False):
        xstart_net = self.net_xstart_ema if use_ema_model else self.net_xstart
        xt_net = self.net_xt_ema if use_ema_model else self.net_xt
        pred_xstart = self.get_bare_model(xstart_net)(self.lq, self.prompt_emb)
        timestep = self.learned_timestep(self.lq.shape[0])
        alpha_cumprod = self._alpha_from_timestep(timestep).view(-1, 1, 1, 1)
        if self.use_sd_inference:
            epsilons, xstarts = self._sd_probe_features()
            epsilon_100, epsilon_150, epsilon_200 = epsilons
            delta_epsilon_150 = epsilon_150 - epsilon_100
            delta_epsilon_200 = epsilon_200 - epsilon_150
            updated_epsilon = (epsilon_200 + delta_epsilon_200 +
                               (delta_epsilon_200 - delta_epsilon_150))
            sd_xt = (torch.sqrt(alpha_cumprod) * pred_xstart.detach() +
                     torch.sqrt(1.0 - alpha_cumprod) * updated_epsilon)
            xt_residual = self.get_bare_model(xt_net)(
                self.lq,
                self._sd_xt_prompt(epsilons, xstarts, use_ema_model))
            pred_xt = sd_xt + xt_residual
        else:
            pred_xt = self.get_bare_model(xt_net)(
                self.lq, self._xt_prompt(pred_xstart, use_ema_model))
        target_xt = (torch.sqrt(alpha_cumprod) * self.xstart_target +
                     torch.sqrt(1.0 - alpha_cumprod) * self.gt)
        pred_xT = (
            pred_xt - torch.sqrt(alpha_cumprod) * pred_xstart.detach()
        ) / torch.sqrt((1.0 - alpha_cumprod).clamp_min(1e-6))
        return pred_xstart, pred_xt, target_xt, pred_xT, timestep

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_xstart.zero_grad()
        self.optimizer_xt.zero_grad()
        pred_xstart, pred_xt, target_xt, pred_xT, timestep = self._predict()
        l_xstart = self.xstart_loss_weight * self.cri_xstart(
            pred_xstart, self.xstart_target)
        l_xt = self.xt_loss_weight * self.cri_xt(pred_xt, target_xt)
        l_xT = self.xT_loss_weight * self.cri_xT(pred_xT, self.gt)
        l_total = l_xstart + l_xt + l_xT
        l_total.backward()
        self.optimizer_xstart.step()
        self.optimizer_xt.step()
        self.update_ema()
        self.output = pred_xT.detach()
        self.log_dict = self.reduce_loss_dict(OrderedDict([
            ('l_xstart', l_xstart), ('l_xt', l_xt), ('l_xT', l_xT),
            ('timestep_learned', timestep.mean()), ('l_total', l_total),
        ]))

    @torch.no_grad()
    def update_ema(self):
        pairs = [(self.net_xstart_ema, self.net_xstart), (self.net_xt_ema, self.net_xt)]
        if self.xstart_prompt_tokens is not None:
            pairs.append((self.xstart_prompt_tokens_ema, self.xstart_prompt_tokens))
        pairs.extend(zip(self.sd_prompt_tokens_ema, self.sd_prompt_tokens))
        for ema_net, net in pairs:
            for ema_param, param in zip(self.get_bare_model(ema_net).parameters(),
                                        self.get_bare_model(net).parameters()):
                ema_param.mul_(self.ema_decay).add_(param.detach(), alpha=1 - self.ema_decay)

    def test(self, use_ema_model=True):
        with torch.no_grad():
            (self.pred_xstart, self.pred_xt, self.target_xt, self.output,
             self.pred_timestep) = (
                self._predict(use_ema_model))

    def save(self, epoch, current_iter):
        self.save_network(self.net_xstart, 'net_xstart', current_iter)
        self.save_network(self.net_xt, 'net_xt', current_iter)
        self.save_network(self.net_xstart_ema, 'net_xstart_ema', current_iter)
        self.save_network(self.net_xt_ema, 'net_xt_ema', current_iter)
        self.save_network(
            self.learned_timestep_module, 'learned_timestep', current_iter)
        if self.xstart_prompt_tokens is not None:
            self.save_network(self.xstart_prompt_tokens, 'xstart_prompt_tokens', current_iter)
        for index, prompt_tokens in enumerate(self.sd_prompt_tokens):
            self.save_network(
                prompt_tokens, f'sd_prompt_token_{index}', current_iter)
        for index, prompt_tokens in enumerate(self.sd_prompt_tokens_ema):
            self.save_network(
                prompt_tokens, f'sd_prompt_token_{index}_ema', current_iter)
        self.save_training_state(epoch, current_iter)
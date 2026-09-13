import importlib
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
from diffusers import DDPMScheduler

from models.archs import define_network
from models.archs.NAFNet_arch import SimpleGate
from models.base_model import BaseModel
from models.reverse_distill_learned_t_model import XStartPromptTokens
from utils.adv_distortion import image_distortion, pil_to_tensor, tensor_to_pil

loss_module = importlib.import_module('models.losses')


class XTWithTimestepHead(nn.Module):
    """Predict x_t with selectable noisy-level estimation heads."""

    def __init__(self, backbone, img_channel, timestep_head_type='classifier'):
        super().__init__()
        self.backbone = backbone
        self.timestep_head_type = (
            'classifier' if timestep_head_type == 'current' else timestep_head_type)
        if self.timestep_head_type not in ('classifier', 'sigmoid', 'fft'):
            raise ValueError(
            'timestep_head_type must be classifier, sigmoid, fft, or current.')
        self.timestep_pool = nn.AdaptiveAvgPool2d(1)
        self.timestep_projection = nn.Conv2d(img_channel, 1000, 1)
        if self.timestep_head_type == 'classifier':
            self.timestep_classifier = nn.Identity()
        elif self.timestep_head_type == 'fft':
            self.fft_height = 64
            self.fft_width = 64
            frequency_u = torch.fft.fftfreq(self.fft_height) * self.fft_height
            frequency_v = torch.fft.rfftfreq(self.fft_width) * self.fft_width
            frequency_distance = torch.sqrt(
                frequency_u[:, None].square() + frequency_v[None, :].square())
            frequency_bin = frequency_distance.floor().long()
            self.register_buffer('fft_frequency_bin', frequency_bin.flatten())
            self.register_buffer(
                'fft_bin_counts', torch.bincount(
                    frequency_bin.flatten(), minlength=frequency_bin.max().item() + 1))
            self.fft_num_bins = self.fft_bin_counts.numel()
            self.timestep_regressor = nn.Sequential(
                nn.Linear(img_channel * self.fft_num_bins, 2),
                SimpleGate())
        else:
            self.timestep_regressor = nn.Conv2d(1000, 1, 1)
        self.register_buffer('timestep_values', torch.arange(1000, dtype=torch.float32))

    def predict_timestep(self, timestep_input):
        """Predict timestep values from an x_t feature without running the backbone."""
        if self.timestep_head_type == 'classifier':
            timestep_features = self.timestep_pool(
                self.timestep_projection(timestep_input))
            timestep_logits = self.timestep_classifier(timestep_features).flatten(1)
            timestep_probs = torch.softmax(timestep_logits, dim=1)
            timestep_base = (timestep_probs * self.timestep_values).sum(dim=1) / 999
        elif self.timestep_head_type == 'fft':
            # This head was designed for four-channel 64x64 diffusion latents.
            if (timestep_input.shape[1] != 4
                    or timestep_input.shape[-2:] != (
                        self.fft_height, self.fft_width)):
                raise ValueError(
                    'The fft timestep head requires noisy inputs shaped [B, 4, 64, 64].')
            # rfft2 transforms H/W to frequency coordinates (u, v); abs() gives
            # their magnitudes, with shape [B, 4, 64, 33] for a 64x64 input.
            fft_magnitude = torch.fft.rfft2(
                timestep_input, dim=(-2, -1)).abs()
            # Flatten the (u, v) axes so every magnitude has one bin-map index:
            # [B, 4, 64, 33] -> [B, 4, 2112].
            fft_magnitude = fft_magnitude.flatten(2)
            # Allocate one accumulated magnitude per batch item, latent channel,
            # and radial bin: [B, 4, fft_num_bins].
            fft_bin_sum = fft_magnitude.new_zeros(
                fft_magnitude.shape[0], fft_magnitude.shape[1], self.fft_num_bins)
            # Add each |FFT(u,v)| to bin floor(sqrt(u^2 + v^2)). Thus, entries
            # in the same integer-width radial interval share a bin; this is not
            # exact equal-distance grouping unless their distances are equal too.
            fft_bin_sum.scatter_add_(
                2,
                self.fft_frequency_bin.view(1, 1, -1).expand_as(fft_magnitude),
                fft_magnitude)
            # Divide every bin sum by its number of assigned (u, v) coordinates
            # to obtain the per-channel average magnitude: [B, 4, fft_num_bins].
            fft_bin_mean = fft_bin_sum / self.fft_bin_counts.view(1, 1, -1)
            # Concatenate all channel/bin averages, then apply Linear(..., 2) and
            # SimpleGate to produce one raw base-timestep scalar per item: [B].
            timestep_base = self.timestep_regressor(
                fft_bin_mean.flatten(1)).squeeze(1)
            timestep_logits = None
        else:
            timestep_logits = None
            timestep_features = self.timestep_pool(
                self.timestep_projection(timestep_input))
            timestep_base = self.timestep_regressor(
                timestep_features).flatten(1).squeeze(1)
            timestep_base = torch.sigmoid(timestep_base)
        timestep = timestep_base * 999
        return timestep_logits, timestep_base, timestep

    def forward(self, inp, prompt_emb=None, timestep_classifier_input=None):
        if prompt_emb is None:
            x_t = self.backbone(inp)
        else:
            x_t = self.backbone(inp, prompt_emb)
        if timestep_classifier_input is None:
            timestep_classifier_input = x_t
        timestep_output = self.predict_timestep(timestep_classifier_input)
        return x_t, *timestep_output


class ReverseDistillModel(BaseModel):
    """Two-branch reverse-distillation model for x_start and x_T prediction."""

    def __init__(self, opt):
        super(ReverseDistillModel, self).__init__(opt)
        self.cfg_sacle = opt.get('cfg_sacle')
        self.ema_decay = opt.get('train', {}).get('ema_decay', 0.95)
        self.alpha_cumprod_method = opt.get(
            'alpha_cumprod_method', 'linear_alpha')
        self.use_pred_xstart_prompt = opt.get('use_pred_xstart_prompt', False)
        self.xstart_is_nafnet = opt['network_xstart']['type'] == 'NAFNet'
        self.xt_is_nafnet = opt['network_xt']['type'] == 'NAFNet'
        self._parallel_inference_streams = None
        self.use_prompt_emb = True
        self.use_afs = opt.get('use_afs', False)
        self.afs_timestep = 990
        self.use_adv_training = opt.get('use_adv_training', False)
        self.adv_training_prob = opt.get('adv_training_prob', 1.0)
        if not 0 <= self.adv_training_prob <= 1:
            raise ValueError('adv_training_prob must be between 0 and 1.')
        self.adv_distortion = opt.get('adv_distortion', {})
        self.adv_vae_batch_size = opt.get('adv_vae_batch_size', 4)
        if (not isinstance(self.adv_vae_batch_size, int)
                or self.adv_vae_batch_size < 1):
            raise ValueError('adv_vae_batch_size must be a positive integer.')
        self.adv_vae = None
        if self.alpha_cumprod_method not in ('linear_alpha', 'log_alpha_cumprod'):
            raise ValueError(
                'alpha_cumprod_method must be linear_alpha or log_alpha_cumprod.')

        scheduler = DDPMScheduler.from_pretrained(
            opt.get('sd_model_id', 'sd-legacy/stable-diffusion-v1-5'),
            subfolder='scheduler')
        self.alpha_cumprods = scheduler.alphas_cumprod.to(
            device=self.device, dtype=torch.float32)
        self.log_alpha_cumprods = self.alpha_cumprods.log()
        self.alphas = torch.cat([
            self.alpha_cumprods[:1],
            self.alpha_cumprods[1:] / self.alpha_cumprods[:-1],
        ])
        self.num_timestep_classes = self.alpha_cumprods.numel()
        self.inference_cfg = 5.5

        xstart_opt = deepcopy(opt['network_xstart'])
        xt_opt = deepcopy(opt['network_xt'])
        self.net_xstart = self.model_to_device(define_network(xstart_opt))
        if self.use_afs:
            self.net_xt = self.model_to_device(define_network(xt_opt))
        else:
            self.net_xt = self.model_to_device(XTWithTimestepHead(
                define_network(xt_opt), xt_opt['img_channel'],
                opt.get('timestep_head_type', 'classifier')))
        self.net_xstart_ema = self.model_to_device(
            define_network(deepcopy(opt['network_xstart'])))
        if self.use_afs:
            self.net_xt_ema = self.model_to_device(
                define_network(deepcopy(opt['network_xt'])))
        else:
            self.net_xt_ema = self.model_to_device(XTWithTimestepHead(
                define_network(deepcopy(opt['network_xt'])), xt_opt['img_channel'],
                opt.get('timestep_head_type', 'classifier')))
        self.xstart_prompt_tokens = None
        self.xstart_prompt_tokens_ema = None
        if self.use_pred_xstart_prompt:
            self.xstart_prompt_tokens = self.model_to_device(XStartPromptTokens())
            self.xstart_prompt_tokens_ema = self.model_to_device(
                XStartPromptTokens())

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
        if self.xstart_prompt_tokens is not None:
            network_specs.extend([
                (self.xstart_prompt_tokens, 'pretrain_xstart_prompt_tokens'),
                (self.xstart_prompt_tokens_ema,
                 'pretrain_xstart_prompt_tokens_ema'),
            ])
        self._loaded_ema = {}
        for network, path_key in network_specs:
            load_path = path_opt.get(path_key)
            if load_path is not None:
                self.load_network(
                    network,
                    load_path,
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
            if not self._loaded_ema.get('pretrain_xstart_prompt_tokens_ema'):
                self.get_bare_model(self.xstart_prompt_tokens_ema).load_state_dict(
                    self.get_bare_model(self.xstart_prompt_tokens).state_dict())

    def init_training_settings(self):
        self.get_bare_model(self.net_xstart).train()
        self.get_bare_model(self.net_xt).train()
        train_opt = self.opt['train']
        self.cri_xstart = self._build_loss(train_opt.get('xstart_opt'))
        self.cri_xt = self._build_loss(train_opt.get('xt_opt'))
        self.cri_timestep = nn.CrossEntropyLoss().to(self.device)
        self.cri_timestep_regression = nn.MSELoss().to(self.device)
        self.xstart_loss_weight = train_opt.get('xstart_loss_weight', 1.0)
        self.xt_loss_weight = train_opt.get('xt_loss_weight', 1.0)
        self.timestep_loss_weight = train_opt.get('timestep_loss_weight', 1.0)
        self.timestep_regression_loss_weight = train_opt.get(
            'timestep_regression_loss_weight', 1.0)
        if self.cri_xstart is None or self.cri_xt is None:
            raise ValueError('Both xstart_opt and xt_opt must be configured.')
        self.setup_optimizers()
        self.setup_schedulers()

    def _build_loss(self, loss_opt):
        if loss_opt is None:
            return None
        loss_opt = deepcopy(loss_opt)
        loss_type = loss_opt.pop('type')
        return getattr(loss_module, loss_type)(**loss_opt).to(self.device)

    def _build_optimizer(self, optimizer_opt, parameters):
        optimizer_opt = deepcopy(optimizer_opt)
        optimizer_type = optimizer_opt.pop('type')
        if optimizer_type == 'Adam':
            return torch.optim.Adam(parameters, **optimizer_opt)
        if optimizer_type == 'AdamW':
            return torch.optim.AdamW(parameters, **optimizer_opt)
        if optimizer_type == 'SGD':
            return torch.optim.SGD(parameters, **optimizer_opt)
        raise NotImplementedError(
            f'optimizer {optimizer_type} is not supported yet.')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        self.optimizer_xstart = self._build_optimizer(
            train_opt['optim_xstart'], self.net_xstart.parameters())
        xt_parameters = list(self.net_xt.parameters())
        if self.xstart_prompt_tokens is not None:
            xt_parameters += list(self.xstart_prompt_tokens.parameters())
        self.optimizer_xt = self._build_optimizer(
            train_opt['optim_xt'], xt_parameters)
        self.optimizers.extend([self.optimizer_xstart, self.optimizer_xt])

    def _load_adv_vae(self):
        if self.adv_vae is None:
            from diffusers import AutoencoderKL
            self.adv_vae = AutoencoderKL.from_pretrained(
                self.opt.get('sd_model_id', 'sd-legacy/stable-diffusion-v1-5'),
                subfolder='vae').to(self.device).eval()
            self.adv_vae.requires_grad_(False)

    @torch.no_grad()
    def _perturb_latents(self, latents):
        self._load_adv_vae()
        scaling_factor = self.adv_vae.config.scaling_factor
        perturbed_latent_batches = []
        for latent_batch in latents.split(self.adv_vae_batch_size):
            decoded_images = self.adv_vae.decode(
                latent_batch / scaling_factor).sample
            perturbed_images = []
            for image in decoded_images:
                perturbed_image, _ = image_distortion(
                    tensor_to_pil(image), self.adv_distortion)
                perturbed_images.append(pil_to_tensor(perturbed_image))
            perturbed_images = torch.stack(perturbed_images).to(
                device=self.device, dtype=decoded_images.dtype)
            perturbed_latent_batches.append(
                self.adv_vae.encode(
                    perturbed_images).latent_dist.mode() * scaling_factor)
        return torch.cat(perturbed_latent_batches).to(dtype=latents.dtype)

    def feed_data(self, data, is_val=False, use_prompt_emb=True):
        self.lq = data['final_latents'].to(self.device)
        if self.use_adv_training:
            perturb_mask = torch.rand(
                self.lq.shape[0], device=self.device) < self.adv_training_prob
            if perturb_mask.any():
                self.lq = self.lq.clone()
                self.lq[perturb_mask] = self._perturb_latents(
                    self.lq[perturb_mask])
        self.gt = data.get('gt')
        if self.gt is not None:
            self.gt = self.gt.to(self.device)
        self.cond_xstart = data.get('cond_xstart')
        if self.cond_xstart is not None:
            self.cond_xstart = self.cond_xstart.to(self.device)
        self.uncond_xstart = data.get('uncond_xstart')
        if self.uncond_xstart is not None:
            self.uncond_xstart = self.uncond_xstart.to(self.device)
        self.use_prompt_emb = use_prompt_emb
        # if use_prompt_emb:
        #     self.prompt_emb = data['prompt'].to(self.device)
        # else:
        #     self.prompt_emb = None
        self.prompt_emb = None

        if (self.cfg_sacle is not None and self.cond_xstart is not None
                and self.uncond_xstart is not None):
            self.xstart_target = self.uncond_xstart + self.cfg_sacle * (
                self.cond_xstart - self.uncond_xstart)
        else:
            self.xstart_target = self.cond_xstart

    def _xt_prompt(self, pred_xstart, use_ema_model=False):
        if self.xt_is_nafnet:
            return None
        if self.xstart_prompt_tokens is None:
            return self.prompt_emb
        prompt_tokens = (self.xstart_prompt_tokens_ema if use_ema_model
                         else self.xstart_prompt_tokens)
        latent_tokens = prompt_tokens(pred_xstart.detach())
        if self.prompt_emb is None:
            return latent_tokens
        return torch.cat([self.prompt_emb, latent_tokens], dim=1)

    def _run_network(self, network, inp, prompt_emb, is_nafnet):
        network = self.get_bare_model(network)
        if is_nafnet:
            return network(inp)
        return network(inp, prompt_emb)

    def _run_parallel_nafnet_branches(
            self, xstart_net, xt_net, xstart_prompt):
        if self._parallel_inference_streams is None:
            self._parallel_inference_streams = (
                torch.cuda.Stream(device=self.lq.device),
                torch.cuda.Stream(device=self.lq.device),
            )
        current_stream = torch.cuda.current_stream(self.lq.device)
        xstart_stream, xt_stream = self._parallel_inference_streams
        xstart_stream.wait_stream(current_stream)
        xt_stream.wait_stream(current_stream)
        with torch.cuda.stream(xstart_stream):
            pred_xstart = self._run_network(
                xstart_net, self.lq, xstart_prompt, self.xstart_is_nafnet)
        with torch.cuda.stream(xt_stream):
            xt_output = self._run_network(xt_net, self.lq, None, True)
        current_stream.wait_stream(xstart_stream)
        current_stream.wait_stream(xt_stream)
        pred_xstart.record_stream(current_stream)
        if torch.is_tensor(xt_output):
            xt_output.record_stream(current_stream)
        else:
            for output in xt_output:
                if torch.is_tensor(output):
                    output.record_stream(current_stream)
        return pred_xstart, xt_output

    def _alpha_from_timestep(self, timestep):
        low_idx = timestep.floor().long().clamp(
            min=0, max=self.alpha_cumprods.numel() - 1)
        high_idx = (low_idx + 1).clamp(max=self.alpha_cumprods.numel() - 1)
        interpolation = timestep - low_idx.to(timestep.dtype)
        if self.alpha_cumprod_method == 'log_alpha_cumprod':
            log_alpha = self.log_alpha_cumprods[low_idx] + interpolation * (
                self.log_alpha_cumprods[high_idx] - self.log_alpha_cumprods[low_idx])
            return log_alpha.exp()

        alpha = self.alphas[low_idx] + interpolation * (
            self.alphas[high_idx] - self.alphas[low_idx])
        return self.alpha_cumprods[low_idx] * alpha.pow(interpolation)

    def _sample_timestep_input(self):
        timestep_class = torch.randint(
            0,
            self.num_timestep_classes,
            (self.gt.shape[0],),
            device=self.device)
        alpha_cumprod = self.alpha_cumprods[timestep_class].view(-1, 1, 1, 1)
        timestep_input = (
            torch.sqrt(alpha_cumprod) * self.xstart_target
            + torch.sqrt(1.0 - alpha_cumprod) * self.gt)
        timestep_target = timestep_class.to(torch.float32) / (
            self.num_timestep_classes - 1)
        return timestep_input, timestep_class, timestep_target

    def _predict(self, use_ema_model=False, clip_timestep=False):
        xstart_net = self.net_xstart_ema if use_ema_model else self.net_xstart
        xt_net = self.net_xt_ema if use_ema_model else self.net_xt
        xstart_prompt = self.prompt_emb if self.use_prompt_emb else None
        if not self.is_train and self.xt_is_nafnet and self.lq.is_cuda:
            pred_xstart, xt_output = self._run_parallel_nafnet_branches(
                xstart_net, xt_net, xstart_prompt)
        else:
            pred_xstart = self._run_network(
                xstart_net, self.lq, xstart_prompt, self.xstart_is_nafnet)
            xt_output = None
        construct_xstart = pred_xstart.detach()
        if self.use_afs:
            residual_xT = (xt_output if xt_output is not None else
                           self._run_network(
                               xt_net,
                               self.lq,
                               self._xt_prompt(construct_xstart, use_ema_model),
                               self.xt_is_nafnet))
            pred_timestep = self.lq.new_full(
                (self.lq.shape[0],), self.afs_timestep)
            alpha_cumprod = self._alpha_from_timestep(pred_timestep).view(
                -1, 1, 1, 1)
            alpha_t = torch.sqrt(alpha_cumprod)
            sigma_t = torch.sqrt(1.0 - alpha_cumprod)
            raw_xT = alpha_t * construct_xstart / ( (1.0 - sigma_t).clamp_min(1e-6))
            pred_xT = raw_xT + residual_xT
            pred_xt = (
                alpha_t * construct_xstart + sigma_t * pred_xT)
            timestep_base = pred_timestep / 999
            return (
                pred_xstart, pred_xt, None, timestep_base,
                pred_timestep, pred_xT)
        if xt_output is None:
            xt_output = self._run_network(
                xt_net,
                self.lq,
                self._xt_prompt(construct_xstart, use_ema_model),
                self.xt_is_nafnet)
        pred_xt, timestep_logits, timestep_base, pred_timestep = xt_output
        alpha_timestep = pred_timestep.clamp(0, 999) if clip_timestep else pred_timestep
        alpha_cumprod = self._alpha_from_timestep(alpha_timestep)
        alpha_cumprod = alpha_cumprod.view(-1, 1, 1, 1)
        pred_xT = (
            pred_xt - torch.sqrt(alpha_cumprod) * construct_xstart
        ) / torch.sqrt((1.0 - alpha_cumprod).clamp_min(1e-6))
        return (
            pred_xstart, pred_xt, timestep_logits, timestep_base,
            pred_timestep, pred_xT)

    def optimize_parameters(self, current_iter, tb_logger):
        if self.xstart_target is None or self.gt is None:
            raise ValueError(
                'Reverse-distillation training requires gt and cond_xstart targets.')
        self.optimizer_xstart.zero_grad()
        self.optimizer_xt.zero_grad()

        (pred_xstart, pred_xt, timestep_logits, timestep_base,
         pred_timestep, pred_xT) = self._predict()
        l_xstart = self.xstart_loss_weight * self.cri_xstart(
            pred_xstart, self.xstart_target)
        l_xT = self.xt_loss_weight * self.cri_xt(pred_xT, self.gt)
        if self.use_afs:
            l_timestep = pred_xT.new_zeros(())
            l_timestep_regression = pred_xT.new_zeros(())
        else:
            (timestep_input, timestep_class,
             timestep_target) = self._sample_timestep_input()
            teacher_timestep_logits, teacher_timestep_base, _ = (
                self.get_bare_model(self.net_xt).predict_timestep(timestep_input))
            if teacher_timestep_logits is not None:
                l_timestep = self.timestep_loss_weight * self.cri_timestep(
                    teacher_timestep_logits, timestep_class)
            else:
                l_timestep = pred_timestep.new_zeros(())
            l_timestep_regression = self.timestep_regression_loss_weight * (
                self.cri_timestep_regression(
                    teacher_timestep_base, timestep_target))
        l_total = l_xstart + l_xT + l_timestep + l_timestep_regression
        l_total.backward()
        self.optimizer_xstart.step()
        self.optimizer_xt.step()

        self.output = pred_xT.detach()
        self.pred_xstart = pred_xstart.detach()
        self.pred_xt = pred_xt.detach()
        self.pred_timestep = pred_timestep.detach()
        self.timestep_base = timestep_base.detach()
        self.update_ema()
        self.log_dict = self.reduce_loss_dict(OrderedDict([
            ('l_xstart', l_xstart),
            ('l_xT', l_xT),
            ('l_timestep', l_timestep),
            ('l_timestep_regression', l_timestep_regression),
            ('timestep_base', timestep_base.mean()),
            ('l_total', l_total),
        ]))

    @torch.no_grad()
    def update_ema(self):
        for ema_param, param in zip(
                self.get_bare_model(self.net_xstart_ema).parameters(),
                self.get_bare_model(self.net_xstart).parameters()):
            ema_param.mul_(self.ema_decay).add_(param.detach(), alpha=1 - self.ema_decay)
        for ema_param, param in zip(
                self.get_bare_model(self.net_xt_ema).parameters(),
                self.get_bare_model(self.net_xt).parameters()):
            ema_param.mul_(self.ema_decay).add_(param.detach(), alpha=1 - self.ema_decay)
        if self.xstart_prompt_tokens is not None:
            for ema_param, param in zip(
                    self.get_bare_model(self.xstart_prompt_tokens_ema).parameters(),
                    self.get_bare_model(self.xstart_prompt_tokens).parameters()):
                ema_param.mul_(self.ema_decay).add_(
                    param.detach(), alpha=1 - self.ema_decay)

    def test(self):
        """Predict x_T from the networks loaded through the configuration."""
        self.get_bare_model(self.net_xstart).eval()
        self.get_bare_model(self.net_xt).eval()
        with torch.no_grad():
            (pred_xstart, pred_xt, timestep_logits, timestep_base,
             pred_timestep, pred_xT) = self._predict(
                use_ema_model=False, clip_timestep=True)
        self.pred_xstart = pred_xstart
        self.pred_xt = pred_xt
        self.timestep_logits = timestep_logits
        self.timestep_base = timestep_base
        self.pred_timestep = pred_timestep
        self.output = pred_xT

    def save(self, epoch, current_iter):
        self.save_network(self.net_xstart, 'net_xstart', current_iter)
        self.save_network(self.net_xt, 'net_xt', current_iter)
        self.save_network(self.net_xstart_ema, 'net_xstart_ema', current_iter)
        self.save_network(self.net_xt_ema, 'net_xt_ema', current_iter)
        if self.xstart_prompt_tokens is not None:
            self.save_network(
                self.xstart_prompt_tokens, 'xstart_prompt_tokens', current_iter)
        self.save_training_state(epoch, current_iter)

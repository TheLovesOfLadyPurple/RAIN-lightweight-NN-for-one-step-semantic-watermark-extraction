import importlib
from collections import OrderedDict
from copy import deepcopy

import torch
from diffusers import DDPMScheduler

from models.archs import define_network
from models.base_model import BaseModel
from utils.adv_distortion import image_distortion, pil_to_tensor, tensor_to_pil

loss_module = importlib.import_module('models.losses')


class ReverseDistillSingleNAFNetModel(BaseModel):
    """Predict terminal noise with one trainable NAFNet."""

    def __init__(self, opt):
        super().__init__(opt)
        self.requires_unconditional_prompt_embeddings = False
        if opt.get('use_prompt_emb', False):
            raise ValueError('The single-NAFNet ablation does not use prompts.')
        self.prediction_type = opt.get('prediction_type', 'direct_xT')
        if self.prediction_type not in ('direct_xT', 'epsilon'):
            raise ValueError('prediction_type must be direct_xT or epsilon.')
        if opt['network_g']['type'] != 'NAFNet':
            raise ValueError('The single-network ablation requires network_g.type NAFNet.')

        self.ema_decay = opt.get('train', {}).get('ema_decay', 0.95)
        self.use_adv_training = opt.get('use_adv_training', False)
        self.adv_training_prob = opt.get('adv_training_prob', 0.5)
        if not 0 <= self.adv_training_prob <= 1:
            raise ValueError('adv_training_prob must be between 0 and 1.')
        self.adv_distortion = opt.get('adv_distortion', {})
        self.adv_vae_batch_size = opt.get('adv_vae_batch_size', 4)
        if (not isinstance(self.adv_vae_batch_size, int)
                or self.adv_vae_batch_size < 1):
            raise ValueError('adv_vae_batch_size must be a positive integer.')
        self.adv_vae = None

        self.alpha_T = None
        self.sigma_T = None
        if self.prediction_type == 'epsilon':
            scheduler = DDPMScheduler.from_pretrained(
                'sd-legacy/stable-diffusion-v1-5', subfolder='scheduler')
            alpha_cumprod_T = scheduler.alphas_cumprod[999].to(
                device=self.device, dtype=torch.float32)
            self.alpha_T = torch.sqrt(alpha_cumprod_T)
            self.sigma_T = torch.sqrt(1.0 - alpha_cumprod_T)

        self.net_g = self.model_to_device(
            define_network(deepcopy(opt['network_g'])))
        self.net_g_ema = self.model_to_device(
            define_network(deepcopy(opt['network_g'])))
        self._load_networks()
        if not self._loaded_ema.get('pretrain_network_g_ema'):
            self.get_bare_model(self.net_g_ema).load_state_dict(
                self.get_bare_model(self.net_g).state_dict())
        self.get_bare_model(self.net_g_ema).requires_grad_(False)
        self.get_bare_model(self.net_g_ema).eval()

        if self.is_train:
            self.init_training_settings()

    def _load_networks(self):
        path_opt = self.opt['path']
        self._loaded_ema = {}
        load_path = path_opt.get('pretrain_network_g')
        if load_path is not None:
            self.load_network(
                self.net_g, load_path, path_opt.get('strict_load_g', True),
                param_key=path_opt.get('param_key', 'params'))
        ema_load_path = path_opt.get('pretrain_network_g_ema')
        if ema_load_path is not None:
            self.load_network(
                self.net_g_ema, ema_load_path,
                path_opt.get('strict_load_g', True),
                param_key=path_opt.get('param_key', 'params'))
            self._loaded_ema['pretrain_network_g_ema'] = True

    def init_training_settings(self):
        self.get_bare_model(self.net_g).train()
        train_opt = self.opt['train']
        loss_opt = deepcopy(train_opt['xt_opt'])
        loss_type = loss_opt.pop('type')
        self.cri_xT = getattr(loss_module, loss_type)(**loss_opt).to(self.device)

        optimizer_opt = deepcopy(train_opt['optim_g'])
        optimizer_type = optimizer_opt.pop('type')
        if optimizer_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(
                self.net_g.parameters(), **optimizer_opt)
        elif optimizer_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(
                self.net_g.parameters(), **optimizer_opt)
        elif optimizer_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(
                self.net_g.parameters(), **optimizer_opt)
        else:
            raise NotImplementedError(
                f'optimizer {optimizer_type} is not supported yet.')
        self.optimizers.append(self.optimizer_g)
        self.setup_schedulers()

    def _load_adv_vae(self):
        if self.adv_vae is None:
            from diffusers import AutoencoderKL
            self.adv_vae = AutoencoderKL.from_pretrained(
                self.opt.get(
                    'sd_model_id', 'sd-legacy/stable-diffusion-v1-5'),
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

    def feed_data(self, data, is_val=False, use_prompt_emb=False):
        del is_val, use_prompt_emb
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

    def _predict(self, use_ema_model=False):
        network = self.net_g_ema if use_ema_model else self.net_g
        network_output = network(self.lq)
        if self.prediction_type == 'direct_xT':
            return network_output, None

        alpha_T = self.alpha_T.to(dtype=self.lq.dtype)
        sigma_T = self.sigma_T.to(dtype=self.lq.dtype)
        pred_xT = alpha_T * self.lq + sigma_T * network_output
        return pred_xT, network_output

    def optimize_parameters(self, current_iter, tb_logger):
        del current_iter, tb_logger
        if self.gt is None:
            raise ValueError('Single-NAFNet training requires a gt target.')
        self.optimizer_g.zero_grad()
        pred_xT, pred_epsilon = self._predict()
        l_xT = self.cri_xT(pred_xT, self.gt)
        l_xT.backward()
        self.optimizer_g.step()

        self.output = pred_xT.detach()
        self.pred_epsilon = (
            pred_epsilon.detach() if pred_epsilon is not None else None)
        self.update_ema()
        self.log_dict = self.reduce_loss_dict(OrderedDict([
            ('l_xT', l_xT),
            ('l_total', l_xT),
        ]))

    @torch.no_grad()
    def update_ema(self):
        for ema_param, param in zip(
                self.get_bare_model(self.net_g_ema).parameters(),
                self.get_bare_model(self.net_g).parameters()):
            ema_param.mul_(self.ema_decay).add_(
                param.detach(), alpha=1 - self.ema_decay)

    def test(self):
        network = self.get_bare_model(self.net_g)
        was_training = network.training
        network.eval()
        with torch.no_grad():
            self.output, self.pred_epsilon = self._predict(
                use_ema_model=False)
        if was_training:
            network.train()

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_g_ema, 'net_g_ema', current_iter)
        self.save_training_state(epoch, current_iter)
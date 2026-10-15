# # ------------------------------------------------------------------------
# # Copyright (c) 2022 megvii-model. All Rights Reserved.
# # ------------------------------------------------------------------------
# # Modified from BasicSR (https://github.com/xinntao/BasicSR)
# # Copyright 2018-2020 BasicSR Authors
# # ------------------------------------------------------------------------

import torch
import pickle
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import numpy as np
from data.data_util import (paired_paths_from_folder,paired_paths_from_lmdb,
                                    tripled_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from data.transforms import augment, paired_random_crop
from .file_client import FileClient
import os
from diffusers import StableDiffusionPipeline, DDPMScheduler
from dpm_solver_v3 import NoiseScheduleVP
import gzip
import random
import functools
from free_lunch_utils import register_free_crossattn_upblock2d, register_free_upblock2d

def _decode_prompt_bytes(value):
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(f'Expected prompt bytes from lmdb, got {type(value)}')
    value = bytes(value)

    # Our LMDB writer stores prompts as gzip-compressed bytes (see utils/lmdb_util.py::read_txt_worker).
    # Gzip streams start with magic bytes 1f 8b.
    if len(value) >= 2 and value[0] == 0x1F and value[1] == 0x8B:
        value = gzip.decompress(value)
    return value.decode('utf-8')


def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True):
    prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
    return {"prompt_embeds": prompt_embeds}

# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def predicted_origin(model_output, timesteps, sample, alphas, sigmas, prediction_type= "epsilon"):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0

def get_noise(pred_x_0, timesteps, x_t, alphas, sigmas):
    sigmas = extract_into_tensor(sigmas, timesteps, x_t.shape)
    alphas = extract_into_tensor(alphas, timesteps, x_t.shape)
    
    noise = (x_t - pred_x_0 * alphas) / sigmas
    return noise

class ComplexOnlineDistillDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """
    
    def append_zero(self, x):
        return torch.cat([x, x.new_zeros([1])])
    
    def sigma_to_t(self, sigma):
        alpha_cumprod_t = 1 - sigma ** 2
        log_alpha_cumprod_t = alpha_cumprod_t.log()
        log_alpha_cumprods = self.alpha_cumprods.log()
        dists = log_alpha_cumprod_t - log_alpha_cumprods[:, None]
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.alphas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.alpha_cumprods[low_idx].log(), self.alpha_cumprods[high_idx].log()
        w = (low - log_alpha_cumprod_t) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)
    
    def vp_sigma_to_t(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma - self.log_sigmas[:, None]
        
        low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
        w = (low - log_sigma) / (low - high)
        w = w.clamp(0, 1)
        t = (1 - w) * low_idx + w * high_idx
        return t.view(sigma.shape)
    
    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7., device='cpu', need_append_zero=True):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return self.append_zero(sigmas).to(device) if need_append_zero else sigmas.to(device)
    
    def get_special_sigmas_with_timesteps(self,timesteps):
        low_idx, high_idx, w = np.minimum(np.floor(timesteps),999), np.minimum(np.ceil(timesteps),999), torch.from_numpy( timesteps - np.floor(timesteps))
        self.alpha_cumprods = self.alpha_cumprods.to('cpu')
        alphas = (1 - w) * self.alpha_cumprods[low_idx] + w * self.alpha_cumprods[high_idx]
        return ((1 - alphas) / alphas) ** 0.5
    
    def get_noise(self, pred_x_0, timesteps, x_t):
        # sigmas = extract_into_tensor(sigmas, timesteps, x_t.shape)
        # alphas = extract_into_tensor(alphas, timesteps, x_t.shape)
        high_idx = torch.ceil(timesteps).int()
        low_idx = torch.floor(timesteps).int()
        # w = (timesteps - low_idx) / (high_idx - low_idx)
        beta_1 = torch.tensor([1e-4],dtype=torch.float32) 
        beta_T = torch.tensor([0.02],dtype=torch.float32)
        ddpm_max_step = torch.tensor([1000.0],dtype=torch.float32)
        beta_t: torch.Tensor = (beta_T - beta_1) / ddpm_max_step  * timesteps + beta_1
        alpha_t = beta_t.new_ones(beta_t.shape[0]) - beta_t
        alpha_cumprod_t_floor = self.alpha_cumprods[low_idx]
        if torch.gt(timesteps - low_idx, torch.tensor([0.1]))[0]:
            alpha_cumprod_t = (alpha_cumprod_t_floor * alpha_t) # the first problems is there, the low idx, in this logic, should be different from timesteps
        else:
            alpha_cumprod_t = alpha_cumprod_t_floor
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sigmas = torch.sqrt(alpha_cumprod_t.new_ones(alpha_cumprod_t.shape[0]) - alpha_cumprod_t)
    
        noise = (x_t - pred_x_0 * sqrt_alpha_cumprod_t) / sigmas
        return noise
    
    def get_real_time_step(self, sigma_min, sigma_max, device = 'cuda'):
        inference_step = 5
        if inference_step == 8:
            sigmas = self.get_sigmas_karras(12, sigma_min, sigma_max, rho=7.0, device=device)
            ct_start, ct_end = self.vp_sigma_to_t(sigmas[0]), self.vp_sigma_to_t(sigmas[9])
            ct = self.get_sigmas_karras(9, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
            sigmas_ct = self.get_special_sigmas_with_timesteps(ct).to(device=device)
            real_ct = [self.vp_sigma_to_t(sigma).to('cpu') for sigma in sigmas_ct]
        elif inference_step == 5:
            sigmas = self.get_sigmas_karras(8, sigma_min, sigma_max, rho=5.0, device=device)
            ct_start, ct_end = self.vp_sigma_to_t(sigmas[0]), self.vp_sigma_to_t(sigmas[6])
            ct = self.get_sigmas_karras(6, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
            sigmas_ct = self.get_special_sigmas_with_timesteps(ct).to(device=device)
            real_ct = [self.vp_sigma_to_t(sigma).to('cpu') for sigma in sigmas_ct]
        elif inference_step == 6:
            sigmas = self.get_sigmas_karras(8, sigma_min, sigma_max, rho=5.0, device=device)
            ct_start, ct_end = self.vp_sigma_to_t(sigmas[0]), self.vp_sigma_to_t(sigmas[6])
            ct = self.get_sigmas_karras(7, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
            sigmas_ct = self.get_special_sigmas_with_timesteps(ct).to(device=device)
            real_ct = [self.vp_sigma_to_t(sigma).to('cpu') for sigma in sigmas_ct]
        return real_ct
    
    def add_noise(self, timesteps, x_0, noise):
        high_idx = torch.ceil(timesteps).int()
        low_idx = torch.floor(timesteps).int()
        w = (timesteps - low_idx) / (high_idx - low_idx)
        beta_1 = torch.tensor([1e-4],dtype=torch.float32) 
        beta_T = torch.tensor([0.02],dtype=torch.float32)
        ddpm_max_step = torch.tensor([1000.0],dtype=torch.float32)
        beta_t: torch.Tensor = (beta_T - beta_1) / ddpm_max_step  * timesteps + beta_1
        alpha_t = beta_t.new_ones(beta_t.shape[0]) - beta_t
        alpha_cumprod_t_floor = self.alpha_cumprods[low_idx]
        if torch.gt(timesteps - low_idx, torch.tensor([0.1]))[0]:
            alpha_cumprod_t = (alpha_cumprod_t_floor * alpha_t) 
        else:
            alpha_cumprod_t = alpha_cumprod_t_floor
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sigmas = torch.sqrt(alpha_cumprod_t.new_ones(alpha_cumprod_t.shape[0]) - alpha_cumprod_t)
        
        # Fix broadcasting
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t#[:, None, None]
        sigmas = sigmas#[:, None, None]
        return x_0 * sqrt_alpha_cumprod_t + sigmas * noise
        
    def get_descrete(self, step):
        if self.total_lcm_iteration == None:
            return None
        # scales = np.ceil(
        #     np.sqrt(
        #         (step / self.total_lcm_iteration) * ((self.end_scales + 1) ** 2 - self.start_scales**2)
        #         + self.start_scales**2
        #     )
        #     - 1
        # ).astype(np.int32)
        # scales = np.maximum(scales, 1)
        # scales = scales + 1
        
        scales = np.ceil(
            np.sqrt(
                (step / self.total_lcm_iteration) * ((self.end_scales + 1) ** 2 - self.start_scales**2)
                + self.start_scales**2
            )
            - 1
        ).astype(np.int32)
        # scales = np.maximum(scales, 1)
        c = -np.log(self.start_ema) * self.start_scales
        target_ema = np.exp(-c / np.maximum(scales, 1))
        scales = scales + 1
        return float(target_ema), int(scales)

    def __init__(self, opt):
        super(ComplexOnlineDistillDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        
        self.batch_size: int = opt['batch_size'] if 'batch_size' in opt else 64
        
        self.data_timestep = opt['timestep']

        # Lazy initialization flag - CUDA models will be loaded on first __getitem__ call
        # This prevents CUDA tensor sharing issues in multiprocessing DataLoader workers
        self._initialized = False
        self.device = "cuda"
        self.pipe = None
        self.alpha_cumprods = None
        self.ns = None
        self.compute_embeddings_fn = None
        self.simgas = None
        self.sqrt_sigma = None
        self.log_sigmas = None
        self.real_time_steps = None
        self.sqrt_alpha_cumprods = None

        self.total_ddpm_step = 1000
        self.total_ddim_step = opt["total_ddim_step"]#20
        self.idx_threshold = 150
        self.is_predict_noise = opt['is_predict_noise'] if 'is_predict_noise' in opt else True
        
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.use_time_embed = opt['use_embed'] if 'use_embed' in opt else False

        self.gt_folder, self.prompt_folder = opt['dataroot_gt'], opt['dataroot_prompt']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder, self.prompt_folder]
            self.io_backend_opt['client_keys'] = ['gt', 'prompt']
            self.paths = paired_paths_from_lmdb(
                [self.gt_folder, self.prompt_folder], ['gt', 'prompt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.gt_folder, self.prompt_folder], ['gt', 'prompt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.gt_folder, self.prompt_folder], ['gt', 'prompt'],
                self.filename_tmpl)
    
    def _lazy_init(self):
        """Lazy initialization of CUDA models and tensors.
        
        This method is called on the first __getitem__ call to ensure CUDA models
        are initialized in the worker process, not in the main process.
        This prevents 'pidfd_getfd: Operation not permitted' errors when using
        DataLoader with num_workers > 0.
        """
        if self._initialized:
            return
        
        DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse
        self.pipe = StableDiffusionPipeline.from_pretrained('sd-legacy/stable-diffusion-v1-5')
        self.pipe.to(device=self.device, torch_dtype=DTYPE)
        self.pipe.scheduler = DDPMScheduler.from_pretrained(
            "sd-legacy/stable-diffusion-v1-5",
            subfolder="scheduler",
            timestep_spacing="trailing",
        )
        if self.opt.get('need_freeU', False):
            register_free_upblock2d(self.pipe, b1=1.2, b2=1.2, s1=0.9, s2=0.2)
            register_free_crossattn_upblock2d(self.pipe, b1=1.2, b2=1.2, s1=0.9, s2=0.2)
            
        alpha_schedule = self.pipe.scheduler.alphas_cumprod.to(device=self.device, dtype=DTYPE)
        
        self.alpha_cumprods = alpha_schedule
        self.ns = NoiseScheduleVP("discrete", alphas_cumprod=self.alpha_cumprods)
        
        self.compute_embeddings_fn = functools.partial(
            compute_embeddings,
            proportion_empty_prompts=0,
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
        )
        
        self.sqrt_alpha_cumprods = torch.sqrt(self.alpha_cumprods)
        self.simgas = self.ns.sigmas.to(device=self.device)
        self.sqrt_sigma = torch.sqrt(self.simgas)
        self.log_sigmas = self.simgas.log()
        self.real_time_steps = torch.tensor(self.get_real_time_step(
            sigma_min=self.ns.sigmas[0].cpu().item(),
            sigma_max=self.ns.sigmas[-1].cpu().item(),
            device=self.simgas[0].device
        ))
        
        self._initialized = True
            
    def __getitem__(self, index):
        # Lazy init CUDA models on first call (in worker process)
        self._lazy_init()
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        # timestep = torch.Tensor([float(int(gt_path.split('_')[1]))]).to(dtype=torch.float32) if self.use_time_embed else torch.Tensor([1000.0]).to(dtype=torch.float32)

        need_load = True
        
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = torch.from_numpy(pickle.loads(img_bytes)).permute(2,0,1) #imfrombytes(img_bytes, float32=True)
            img_gt = img_gt.unsqueeze(0)
            # print("ttttensor size{}".format(img_gt.size()))
        except:
            raise Exception("gt path {} not working".format(gt_path))
        noise = torch.randn_like(img_gt)
        timestep = img_gt.new_ones(img_gt.shape[0]) * self.data_timestep
        timestep = timestep.to(torch.int32)
        img_lq = self.add_noise(timesteps=timestep,x_0=img_gt,noise=noise) #self.pipe.scheduler.add_noise(img_gt, noise, timestep)
        prompt_path = self.paths[index]['prompt_path']
        # print(', lq path', lq_path)
        str_bytes = self.file_client.get(prompt_path, 'prompt')
        try:
            prompt = _decode_prompt_bytes(str_bytes)
        except Exception:
            raise Exception("prompt path {} not working".format(prompt_path))

        y = prompt # category label, which may used prompt embed from clip to replace it later
            
        if isinstance(y, str):
            y = y #+ 'high quality, best quality, masterpiece, 4K, highres, extremely detailed, ultra-detailed'
            y = (y,)
        if isinstance(y, tuple) or isinstance(y, str):
            y = list(y)
        encoded_text = self.compute_embeddings_fn(y)
        c =  encoded_text.pop("prompt_embeds")
        drop_ids = torch.rand(c.shape[0]).cuda() < 0.1
            # uc = None
            # if opt.scale != 1.0:
            # caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        uc = self.compute_embeddings_fn(img_gt.shape[0] * [""])
        uc = uc.pop("prompt_embeds") if uc is not None else None
        uc = uc.to(img_gt.device) # (B, 77,768)
        c = c.to(img_gt.device)   # (B, 77,768)
        drop_mask = drop_ids[:, None, None].to(img_gt.device)
        caption = torch.where(drop_mask, uc, c).to(img_gt.device)
        with torch.no_grad():
            score = self.pipe.unet(img_lq.to(device=self.device), timestep.to(device=self.device), encoder_hidden_states=caption.to(device=self.device)).sample
            score = score.to(device="cpu")
            sigma_schedule = torch.sqrt(1 - self.alpha_cumprods)
            alpha_schedule = torch.sqrt(self.alpha_cumprods)
            pred_x_0 = predicted_origin(
                score,
                timestep,
                img_lq,
                alpha_schedule,
                sigma_schedule,
            )
            if self.is_predict_noise:
                target = score
            else:
                target = pred_x_0
            N = self.total_ddim_step
            indices = (torch.randint(1, N + 1, (1,), device=img_lq.device)).float() #torch.rand( 0, N - 1, img_lq.shape[0], device=img_lq.device )
            progress = indices / (N * 1.0)
            tmp_step = timestep * progress
            if N != 1:
                middle_img_lq = self.add_noise(timesteps=tmp_step,x_0=pred_x_0,noise=score)
            else: 
                middle_img_lq = img_lq
        middle_img_lq = middle_img_lq.squeeze(0)
        target = target.squeeze(0)
        caption = caption.squeeze(0)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq],
        #                             bgr2rgb=True,
        #                             float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(middle_img_lq, self.mean, self.std, inplace=True)
            normalize(target, self.mean, self.std, inplace=True)
        # self.tmp_fetch_feature_count += 1
        # if self.tmp_fetch_feature_count % self.batch_size == 0:
        #     self.tmp_iteration += 1
        #     if self.ema_rate_path is not None:
        #         tmp_value = torch.Tensor(self.tmp_fetch_feature_count)
        #         max_tmp_fetch_feature_count_file = '{}.pth'.format(self.tmp_fetch_feature_count)
        #         torch.save(tmp_value,os.path.join(self.ema_rate_path, max_tmp_fetch_feature_count_file))
                # tmp_value.save(os.path.join(self.ema_rate_path, max_tmp_fetch_feature_count_file))
                # iteration_resume_state = torch.load(os.path.join(ema_rate_path, max_ema_rate_state_file)) 
        # nxt_step = torch.max ( torch.stack([ tmp_timestep - next_step, torch.tensor([0])])) if tmp_timestep is not None else timestep
        # print(nxt_step)                 
        return {
            'lq': middle_img_lq,
            'gt': target,
            'prompt': caption,
            # 'ema_decay_rate': mu if mu is not None else 0.95,
            'timestep': tmp_step,
            'next_timestep': tmp_step,
            'idx': tmp_step,
            # 'lq_path': lq_path,
            'gt_path': gt_path,
            'prompt_path': prompt_path,
            # 'save_tmp_fetch_feature': self.tmp_fetch_feature_count
        }

    def __len__(self):
        return len(self.paths)
    
    def find_real_time_step(self, timestep):
        # self.real_time_steps is sorted in descending order.
        # To find the closest value, we can find the insertion point in the reversed (ascending) tensor.
        
        # Flip for ascending order
        reversed_real_time_steps = torch.flip(self.real_time_steps, [0])
        
        # Find insertion index in ascending tensor
        idx = torch.searchsorted(reversed_real_time_steps, timestep)
        
        # Handle edge cases
        if idx == 0:
            return reversed_real_time_steps[0]
        if idx == len(reversed_real_time_steps):
            return reversed_real_time_steps[-1]
            
        # Compare with neighbors to find the closest
        before = reversed_real_time_steps[idx - 1]
        after = reversed_real_time_steps[idx]
        if (timestep - before) < (after - timestep):
            return before
        else:
            return after

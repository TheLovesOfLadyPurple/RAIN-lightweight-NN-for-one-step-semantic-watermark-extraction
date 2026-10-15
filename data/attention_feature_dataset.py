
import torch
import pickle
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import numpy as np
from .data_util import (paired_paths_from_folder,
                                    tripled_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from .transforms import augment, paired_random_crop
from .file_client import FileClient
import os
from diffusers import DDPMScheduler
from dpm_solver_v3 import NoiseScheduleVP

# def extract_into_tensor(a, t, x_shape):
#     b, *_ = t.shape
#     out = a.gather(-1, t)
#     return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# def predicted_origin(model_output, timesteps, sample, alphas, sigmas, prediction_type= "epsilon"):
#     if prediction_type == "epsilon":
#         sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
#         alphas = extract_into_tensor(alphas, timesteps, sample.shape)
#         pred_x_0 = (sample - sigmas * model_output) / alphas
#     elif prediction_type == "v_prediction":
#         sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
#         alphas = extract_into_tensor(alphas, timesteps, sample.shape)
#         pred_x_0 = alphas * sample - sigmas * model_output
#     else:
#         raise ValueError(f"Prediction type {prediction_type} currently not supported.")

#     return pred_x_0

# def get_noise(pred_x_0, timesteps, x_t, alphas, sigmas):
#     sigmas = extract_into_tensor(sigmas, timesteps, x_t.shape)
#     alphas = extract_into_tensor(alphas, timesteps, x_t.shape)
    
#     noise = (x_t - pred_x_0 * alphas) / sigmas
#     return noise

# class AttentionFeatureDataset(data.Dataset):
#     """Paired image dataset for image restoration.

#     Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
#     GT image pairs.

#     There are three modes:
#     1. 'lmdb': Use lmdb files.
#         If opt['io_backend'] == lmdb.
#     2. 'meta_info_file': Use meta information file to generate paths.
#         If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
#     3. 'folder': Scan folders to generate paths.
#         The rest.

#     Args:
#         opt (dict): Config for train datasets. It contains the following keys:
#             dataroot_gt (str): Data root path for gt.
#             dataroot_lq (str): Data root path for lq.
#             meta_info_file (str): Path for meta information file.
#             io_backend (dict): IO backend type and other kwarg.
#             filename_tmpl (str): Template for each filename. Note that the
#                 template excludes the file extension. Default: '{}'.
#             gt_size (int): Cropped patched size for gt patches.
#             use_flip (bool): Use horizontal flips.
#             use_rot (bool): Use rotation (use vertical flip and transposing h
#                 and w for implementation).

#             scale (bool): Scale, which will be added automatically.
#             phase (str): 'train' or 'val'.
#     """
    
#     def append_zero(self, x):
#         return torch.cat([x, x.new_zeros([1])])
    
#     def sigma_to_t(self, sigma):
#         alpha_cumprod_t = 1 - sigma ** 2
#         log_alpha_cumprod_t = alpha_cumprod_t.log()
#         log_alpha_cumprods = self.alpha_cumprods.log()
#         dists = log_alpha_cumprod_t - log_alpha_cumprods[:, None]
#         low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.alphas.shape[0] - 2)
#         high_idx = low_idx + 1
#         low, high = self.alpha_cumprods[low_idx].log(), self.alpha_cumprods[high_idx].log()
#         w = (low - log_alpha_cumprod_t) / (low - high)
#         w = w.clamp(0, 1)
#         t = (1 - w) * low_idx + w * high_idx
#         return t.view(sigma.shape)
    
#     def vp_sigma_to_t(self, sigma):
#         log_sigma = sigma.log()
#         dists = log_sigma - self.log_sigmas[:, None]
        
#         low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=self.log_sigmas.shape[0] - 2)
#         high_idx = low_idx + 1
#         low, high = self.log_sigmas[low_idx], self.log_sigmas[high_idx]
#         w = (low - log_sigma) / (low - high)
#         w = w.clamp(0, 1)
#         t = (1 - w) * low_idx + w * high_idx
#         return t.view(sigma.shape)
    
#     def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7., device='cpu', need_append_zero=True):
#         """Constructs the noise schedule of Karras et al. (2022)."""
#         ramp = torch.linspace(0, 1, n)
#         min_inv_rho = sigma_min ** (1 / rho)
#         max_inv_rho = sigma_max ** (1 / rho)
#         sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
#         return self.append_zero(sigmas).to(device) if need_append_zero else sigmas.to(device)
    
#     def get_special_sigmas_with_timesteps(self,timesteps):
#         low_idx, high_idx, w = np.minimum(np.floor(timesteps),999), np.minimum(np.ceil(timesteps),999), torch.from_numpy( timesteps - np.floor(timesteps))
#         self.alpha_cumprods = self.alpha_cumprods.to('cpu')
#         alphas = (1 - w) * self.alpha_cumprods[low_idx] + w * self.alpha_cumprods[high_idx]
#         return ((1 - alphas) / alphas) ** 0.5
    
#     def get_noise(self, pred_x_0, timesteps, x_t):
#         # sigmas = extract_into_tensor(sigmas, timesteps, x_t.shape)
#         # alphas = extract_into_tensor(alphas, timesteps, x_t.shape)
#         high_idx = torch.ceil(timesteps).int()
#         low_idx = torch.floor(timesteps).int()
#         # w = (timesteps - low_idx) / (high_idx - low_idx)
#         beta_1 = torch.tensor([1e-4],dtype=torch.float32) 
#         beta_T = torch.tensor([0.02],dtype=torch.float32)
#         ddpm_max_step = torch.tensor([1000.0],dtype=torch.float32)
#         beta_t: torch.Tensor = (beta_T - beta_1) / ddpm_max_step  * timesteps + beta_1
#         alpha_t = beta_t.new_ones(beta_t.shape[0]) - beta_t
#         alpha_cumprod_t_floor = self.alpha_cumprods[low_idx]
#         if torch.gt(timesteps - low_idx, torch.tensor([0.1]))[0]:
#             alpha_cumprod_t = (alpha_cumprod_t_floor * alpha_t) # the first problems is there, the low idx, in this logic, should be different from timesteps
#         else:
#             alpha_cumprod_t = alpha_cumprod_t_floor
#         sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
#         sigmas = torch.sqrt(alpha_cumprod_t.new_ones(alpha_cumprod_t.shape[0]) - alpha_cumprod_t)
    
#         noise = (x_t - pred_x_0 * sqrt_alpha_cumprod_t) / sigmas
#         return noise
    
#     def get_real_time_step(self, sigma_min, sigma_max, device = 'cuda'):
#         inference_step = 5
#         if inference_step == 8:
#             sigmas = self.get_sigmas_karras(12, sigma_min, sigma_max, rho=7.0, device=device)
#             ct_start, ct_end = self.vp_sigma_to_t(sigmas[0]), self.vp_sigma_to_t(sigmas[9])
#             ct = self.get_sigmas_karras(9, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
#             sigmas_ct = self.get_special_sigmas_with_timesteps(ct).to(device=device)
#             real_ct = [self.vp_sigma_to_t(sigma).to('cpu') for sigma in sigmas_ct]
#         elif inference_step == 5:
#             sigmas = self.get_sigmas_karras(8, sigma_min, sigma_max, rho=5.0, device=device)
#             ct_start, ct_end = self.vp_sigma_to_t(sigmas[0]), self.vp_sigma_to_t(sigmas[6])
#             ct = self.get_sigmas_karras(6, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
#             sigmas_ct = self.get_special_sigmas_with_timesteps(ct).to(device=device)
#             real_ct = [self.vp_sigma_to_t(sigma).to('cpu') for sigma in sigmas_ct]
#         elif inference_step == 6:
#             sigmas = self.get_sigmas_karras(8, sigma_min, sigma_max, rho=5.0, device=device)
#             ct_start, ct_end = self.vp_sigma_to_t(sigmas[0]), self.vp_sigma_to_t(sigmas[6])
#             ct = self.get_sigmas_karras(7, ct_end.item(), ct_start.item(),rho=1.2, device='cpu',need_append_zero=False).numpy()
#             sigmas_ct = self.get_special_sigmas_with_timesteps(ct).to(device=device)
#             real_ct = [self.vp_sigma_to_t(sigma).to('cpu') for sigma in sigmas_ct]
#         return real_ct
    
#     def add_noise(self, timesteps, x_0, noise):
#         high_idx = torch.ceil(timesteps).int()
#         low_idx = torch.floor(timesteps).int()
#         w = (timesteps - low_idx) / (high_idx - low_idx)
#         beta_1 = torch.tensor([1e-4],dtype=torch.float32) 
#         beta_T = torch.tensor([0.02],dtype=torch.float32)
#         ddpm_max_step = torch.tensor([1000.0],dtype=torch.float32)
#         beta_t: torch.Tensor = (beta_T - beta_1) / ddpm_max_step  * timesteps + beta_1
#         alpha_t = beta_t.new_ones(beta_t.shape[0]) - beta_t
#         alpha_cumprod_t_floor = self.alpha_cumprods[low_idx]
#         if torch.gt(timesteps - low_idx, torch.tensor([0.1]))[0]:
#             alpha_cumprod_t = (alpha_cumprod_t_floor * alpha_t) 
#         else:
#             alpha_cumprod_t = alpha_cumprod_t_floor
#         sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
#         sigmas = torch.sqrt(alpha_cumprod_t.new_ones(alpha_cumprod_t.shape[0]) - alpha_cumprod_t)
        
#         # Fix broadcasting
#         sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t[:, None, None]
#         sigmas = sigmas[:, None, None]
#         return x_0 * sqrt_alpha_cumprod_t + sigmas * noise
        
#     def get_descrete(self, step):
#         if self.total_lcm_iteration == None:
#             return None
#         # scales = np.ceil(
#         #     np.sqrt(
#         #         (step / self.total_lcm_iteration) * ((self.end_scales + 1) ** 2 - self.start_scales**2)
#         #         + self.start_scales**2
#         #     )
#         #     - 1
#         # ).astype(np.int32)
#         # scales = np.maximum(scales, 1)
#         # scales = scales + 1
        
#         scales = np.ceil(
#             np.sqrt(
#                 (step / self.total_lcm_iteration) * ((self.end_scales + 1) ** 2 - self.start_scales**2)
#                 + self.start_scales**2
#             )
#             - 1
#         ).astype(np.int32)
#         # scales = np.maximum(scales, 1)
#         c = -np.log(self.start_ema) * self.start_scales
#         target_ema = np.exp(-c / np.maximum(scales, 1))
#         scales = scales + 1
#         return float(target_ema), int(scales)

#     def __init__(self, opt):
#         super(AttentionFeatureDataset, self).__init__()
#         self.opt = opt
#         # file client (io backend)
        
#         self.batch_size: int = opt['batch_size'] if 'batch_size' in opt else 64
#         self.ema_rate_path = opt['ema_rate_path'] if 'ema_rate_path' in opt else None
#         self.tmp_fetch_feature_count: int = opt['iteration_resume_state'] if 'iteration_resume_state' in opt else 0
#         self.tmp_iteration: int = self.tmp_fetch_feature_count / self.batch_size if self.tmp_fetch_feature_count is not None else 0
        
#         self.file_client = None
#         self.io_backend_opt = opt['io_backend']
#         self.mean = opt['mean'] if 'mean' in opt else None
#         self.std = opt['std'] if 'std' in opt else None
#         self.use_time_embed = opt['use_embed'] if 'use_embed' in opt else False

#         DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse
#         device = "cuda" 
#         scheduler = DDPMScheduler.from_pretrained(
#             "sd-legacy/stable-diffusion-v1-5",
#             subfolder="scheduler",
#             timestep_spacing="trailing",
#         )
#         alpha_schedule = scheduler.alphas_cumprod.to(device=device, dtype=DTYPE)
#         self.alpha_cumprods = alpha_schedule
#         self.ns = NoiseScheduleVP("discrete", alphas_cumprod=self.alpha_cumprods)

#         self.total_ddpm_step = 1000
#         self.total_ddim_step = 10
#         self.idx_threshold = 150
        
#         self.sqrt_alpha_cumprods = torch.sqrt(self.alpha_cumprods)
#         self.simgas = self.ns.sigmas
#         self.sqrt_sigma = torch.sqrt(self.simgas)
#         self.log_sigmas = self.simgas.log()
#         self.real_time_steps = torch.tensor(self.get_real_time_step(sigma_min = self.ns.sigmas[0].cpu().item()
#                                                        , sigma_max = self.ns.sigmas[-1].cpu().item()
#                                                        , device=self.simgas[0].device))
#         self.is_predict_noise = opt['is_predict_noise'] if 'is_predict_noise' in opt else False
#         self.total_lcm_iteration = opt['total_lcm_iteration'] if 'total_lcm_iteration' in opt else None
#         # self.total_steps = opt['total_steps'] if 'total_steps' in opt else None
#         self.end_scales = opt['end_scales'] if 'end_scales' in opt else self.total_ddim_step
#         self.start_scales = opt['start_scales'] if 'start_scales' in opt else 1
#         self.start_ema = opt['start_ema'] if 'start_ema' in opt else 0.95

#         self.gt_folder, self.lq_folder, self.prompt_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_prompt']
#         if 'filename_tmpl' in opt:
#             self.filename_tmpl = opt['filename_tmpl']
#         else:
#             self.filename_tmpl = '{}'

#         if self.io_backend_opt['type'] == 'lmdb':
#             self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.prompt_folder]
#             self.io_backend_opt['client_keys'] = ['lq', 'gt', 'prompt']
#             self.paths = tripled_paths_from_lmdb(
#                 [self.lq_folder, self.gt_folder, self.prompt_folder], ['lq', 'gt', 'prompt'])
#         elif 'meta_info_file' in self.opt and self.opt[
#                 'meta_info_file'] is not None:
#             self.paths = paired_paths_from_meta_info_file(
#                 [self.lq_folder, self.gt_folder], ['lq', 'gt'],
#                 self.opt['meta_info_file'], self.filename_tmpl)
#         else:
#             self.paths = paired_paths_from_folder(
#                 [self.lq_folder, self.gt_folder], ['lq', 'gt'],
#                 self.filename_tmpl)

#     def __getitem__(self, index):
#         if self.file_client is None:
#             self.file_client = FileClient(
#                 self.io_backend_opt.pop('type'), **self.io_backend_opt)

#         scale = self.opt['scale']

#         # Load gt and lq images. Dimension order: HWC; channel order: BGR;
#         # image range: [0, 1], float32.
#         gt_path = self.paths[index]['gt_path']
#         # timestep = torch.Tensor([float(int(gt_path.split('_')[1]))]).to(dtype=torch.float32) if self.use_time_embed else torch.Tensor([1000.0]).to(dtype=torch.float32)
#         timestep = torch.Tensor([int(gt_path.split('_')[1])]).to(dtype=torch.int) if self.use_time_embed else torch.Tensor([1000]).to(dtype=torch.int)
#         # timestep = self.find_real_time_step(timestep)
        
#         img_bytes = self.file_client.get(gt_path, 'gt')
#         try:
#             img_gt = torch.from_numpy(pickle.loads(img_bytes)).permute(2,0,1) #imfrombytes(img_bytes, float32=True)
#             # print("ttttensor size{}".format(img_gt.size()))
#         except:
#             raise Exception("gt path {} not working".format(gt_path))

#         lq_path = self.paths[index]['lq_path']
#         # print(', lq path', lq_path)
#         img_bytes = self.file_client.get(lq_path, 'lq')
#         try:
#             img_lq = torch.from_numpy(pickle.loads(img_bytes)) #imfrombytes(img_bytes, float32=True)
#             img_lq = img_lq.permute(2,0,1)
#             x_0 = img_gt.clone()
#             next_lq = None
#             mu = None
#             N = None
#             tmp_timestep = None
#             if self.is_predict_noise:
#                 # int_timestep = timestep.to(dtype=torch.int64)
#                 img_gt = self.get_noise(img_gt,timesteps=timestep,x_t=img_lq)
#             if self.total_lcm_iteration is not None:
#                 # int_timestep = timestep.to(dtype=torch.int64)
#                 noise = self.get_noise(img_gt,timesteps=timestep,x_t=img_lq,)
#                 # mu, N = self.get_descrete(self.tmp_iteration)
#                 mu = self.start_ema
#                 N = self.total_ddim_step
#                 next_step = (timestep) / self.total_ddim_step # int( (timestep) / self.total_ddim_step)
#                 indices = (torch.randint(0, N + 1, (1,), device=img_lq.device)).float() #torch.rand( 0, N - 1, img_lq.shape[0], device=img_lq.device )
#                 progress = indices / (N * 1.0)
#                 # tmp_timestep = timestep * progress
#                 nxt_tmp_step = timestep * progress
#                 # if nxt_tmp_step <= self.idx_threshold:
#                 #     nxt_tmp_step = timestep
#                 tmp_timestep, b = torch.min( torch.stack([ torch.ceil(nxt_tmp_step).int() + next_step , torch.ceil(timestep).int()]),dim=0)
#                 # print("b {}".format(b))
#                 # print("timestep {}".format(tmp_timestep))
#                 # print("nxt_tmp_step {}".format(nxt_tmp_step))
#                 if torch.gt(timestep - tmp_timestep, torch.tensor([0.1]))[0]:
#                     img_lq = self.add_noise(timesteps=tmp_timestep,x_0=x_0,noise=noise)
#                 if not torch.gt(tmp_timestep - nxt_tmp_step, torch.tensor([0.1]))[0]:
#                     nxt_tmp_step = torch.max ( torch.stack([ tmp_timestep - next_step, torch.tensor([0])]) ).unsqueeze(0)
#                 # print("nxt_tmp_step {}".format(nxt_tmp_step))
#                 if torch.gt(nxt_tmp_step, torch.tensor([0.1]))[0]:
#                     next_lq = self.add_noise(timesteps= nxt_tmp_step, x_0=x_0, noise=noise)
#                 else:
#                     next_lq = x_0
#                     # print("nxt_tmp_step {}".format(nxt_tmp_step))
#             # print("ttttensor size{}".format(img_lq.size()))
#         except:
#             # print("ttttensor size{}".format(img_lq.size()))
#             raise Exception("lq path {} not working".format(lq_path))
        
#         prompt_path = self.paths[index]['prompt_path']
#         # print(', lq path', lq_path)
#         img_bytes = self.file_client.get(prompt_path, 'prompt')
#         try:
#             prompt = torch.from_numpy(pickle.loads(img_bytes)) #imfrombytes(img_bytes, float32=True)
#             # print("ttttensor size{}".format(img_lq.size()))
#         except:
#             raise Exception("prompt path {} not working".format(prompt_path))

#         # if not self.use_time_embed:
#         #     timestep = torch.Tensor([1000.0])

#         # augmentation for training
#         # if self.opt['phase'] == 'train':
#         #     gt_size = self.opt['gt_size']
#         #     # padding
#         #     img_gt, img_lq = padding(img_gt, img_lq, gt_size)

#         #     # random crop
#         #     img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
#         #                                         gt_path)
#         #     # flip, rotation
#         #     img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
#         #                              self.opt['use_rot'])

#         # TODO: color space transform
#         # BGR to RGB, HWC to CHW, numpy to tensor
#         # img_gt, img_lq = img2tensor([img_gt, img_lq],
#         #                             bgr2rgb=True,
#         #                             float32=True)
#         # normalize
#         if self.mean is not None or self.std is not None:
#             normalize(img_lq, self.mean, self.std, inplace=True)
#             normalize(img_gt, self.mean, self.std, inplace=True)
#         self.tmp_fetch_feature_count += 1
#         if self.tmp_fetch_feature_count % self.batch_size == 0:
#             self.tmp_iteration += 1
#             if self.ema_rate_path is not None:
#                 tmp_value = torch.Tensor(self.tmp_fetch_feature_count)
#                 max_tmp_fetch_feature_count_file = '{}.pth'.format(self.tmp_fetch_feature_count)
#                 torch.save(tmp_value,os.path.join(self.ema_rate_path, max_tmp_fetch_feature_count_file))
#                 # tmp_value.save(os.path.join(self.ema_rate_path, max_tmp_fetch_feature_count_file))
#                 # iteration_resume_state = torch.load(os.path.join(ema_rate_path, max_ema_rate_state_file)) 
#         # nxt_step = torch.max ( torch.stack([ tmp_timestep - next_step, torch.tensor([0])])) if tmp_timestep is not None else timestep
#         # print(nxt_step)                 
#         return {
#             'lq': img_lq,
#             'gt': img_gt,
#             'next_lq': next_lq if tmp_timestep is not None else img_lq,
#             'prompt': prompt,
#             'ema_decay_rate': mu if mu is not None else 0.95,
#             'timestep': tmp_timestep if tmp_timestep is not None else timestep,
#             'next_timestep': nxt_tmp_step if tmp_timestep is not None else timestep,
#             'idx': timestep,
#             'lq_path': lq_path,
#             'gt_path': gt_path,
#             'prompt_path': prompt_path,
#             'save_tmp_fetch_feature': self.tmp_fetch_feature_count
#         }

#     def __len__(self):
#         return len(self.paths)
    
#     def find_real_time_step(self, timestep):
#         # self.real_time_steps is sorted in descending order.
#         # To find the closest value, we can find the insertion point in the reversed (ascending) tensor.
        
#         # Flip for ascending order
#         reversed_real_time_steps = torch.flip(self.real_time_steps, [0])
        
#         # Find insertion index in ascending tensor
#         idx = torch.searchsorted(reversed_real_time_steps, timestep)
        
#         # Handle edge cases
#         if idx == 0:
#             return reversed_real_time_steps[0]
#         if idx == len(reversed_real_time_steps):
#             return reversed_real_time_steps[-1]
            
#         # Compare with neighbors to find the closest
#         before = reversed_real_time_steps[idx - 1]
#         after = reversed_real_time_steps[idx]
#         if (timestep - before) < (after - timestep):
#             return before
#         else:
#             return after

class AttentionFeatureDataset(data.Dataset):
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

    def __init__(self, opt):
        super(AttentionFeatureDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.use_time_embed = opt['use_embed'] if 'use_embed' in opt else False

        self.gt_folder, self.lq_folder, self.prompt_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_prompt']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.prompt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt', 'prompt']
            self.paths = tripled_paths_from_lmdb(
                [self.lq_folder, self.gt_folder, self.prompt_folder], ['lq', 'gt', 'prompt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        timestep = torch.Tensor([(int(gt_path.split('_')[1]))]).to(dtype=torch.int) if self.use_time_embed else torch.Tensor([1000]).to(dtype=torch.int)
        # print('gt path,', torch.Tensor([float(int(gt_path.split('_')[1]))]))
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = torch.from_numpy(pickle.loads(img_bytes)).permute(2,0,1) #imfrombytes(img_bytes, float32=True)
            # print("ttttensor size{}".format(img_gt.size()))
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = torch.from_numpy(pickle.loads(img_bytes)) #imfrombytes(img_bytes, float32=True)
            img_lq = img_lq.permute(2,0,1)
            # print("ttttensor size{}".format(img_lq.size()))
        except:
            # print("ttttensor size{}".format(img_lq.size()))
            raise Exception("lq path {} not working".format(lq_path))
        
        prompt_path = self.paths[index]['prompt_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(prompt_path, 'prompt')
        try:
            prompt = torch.from_numpy(pickle.loads(img_bytes)) #imfrombytes(img_bytes, float32=True)
            # print("ttttensor size{}".format(img_lq.size()))
        except:
            raise Exception("prompt path {} not working".format(prompt_path))


        # augmentation for training
        # if self.opt['phase'] == 'train':
        #     gt_size = self.opt['gt_size']
        #     # padding
        #     img_gt, img_lq = padding(img_gt, img_lq, gt_size)

        #     # random crop
        #     img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
        #                                         gt_path)
        #     # flip, rotation
        #     img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
        #                              self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq],
        #                             bgr2rgb=True,
        #                             float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'prompt': prompt,
            'timestep': timestep,
            'lq_path': lq_path,
            'gt_path': gt_path,
            'prompt_path': prompt_path
        }

    def __len__(self):
        return len(self.paths)

import torch
import pickle
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import numpy as np
from .data_util import (multiple_paths_from_folder, multiple_paths_from_lmdb,
                        multiple_paths_from_meta_info_file)
from .file_client import FileClient
from .online_distill_dataset import _decode_prompt_bytes


def _load_feature_tensor(feature_bytes, feature_name, feature_path):
    """Load a latent feature and normalize its layout to [C, H, W]."""
    try:
        feature = torch.from_numpy(pickle.loads(feature_bytes))
    except Exception as error:
        raise Exception(f'{feature_name} path {feature_path} not working') from error

    if feature.ndim != 3:
        raise ValueError(
            f'{feature_name} at {feature_path} must have three dimensions, '
            f'but got shape {tuple(feature.shape)}.')
    if feature.shape[0] == 4:
        return feature
    if feature.shape[-1] == 4:
        return feature.permute(2, 0, 1)
    raise ValueError(
        f'{feature_name} at {feature_path} must use a four-channel CHW or HWC layout, '
        f'but got shape {tuple(feature.shape)}.')


class ReverseDistillFeatureDataset(data.Dataset):
    """Reverse distillation feature dataset."""

    def __init__(self, opt):
        super(ReverseDistillFeatureDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend'].copy()
        self.io_backend_type = self.io_backend_opt.pop('type')
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.use_time_embed = opt['use_embed'] if 'use_embed' in opt else False

        self.gt_folder, self.cond_xstart_folder = opt['dataroot_gt'], opt['dataroot_cond_xstart']
        self.uncond_xstart_folder, self.final_latents_folder = opt['dataroot_uncond_xstart'], opt['dataroot_final_latents']
        self.prompt_folder = opt['dataroot_prompt']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        folders = [self.gt_folder, self.cond_xstart_folder,
                   self.uncond_xstart_folder, self.final_latents_folder,
                   self.prompt_folder]
        keys = ['gt', 'cond_xstart', 'uncond_xstart', 'final_latents', 'prompt']
        if self.io_backend_type == 'lmdb':
            self.io_backend_opt['db_paths'] = folders
            self.io_backend_opt['client_keys'] = keys
            self.paths = multiple_paths_from_lmdb(folders, keys)
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = multiple_paths_from_meta_info_file(
                folders, keys, self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = multiple_paths_from_folder(
                folders, keys, self.filename_tmpl)

    def __getstate__(self):
        """Do not serialize LMDB environments into spawned loader workers."""
        state = self.__dict__.copy()
        state['file_client'] = None
        return state

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_type, **self.io_backend_opt)

        scale = self.opt['scale']

        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = _load_feature_tensor(img_bytes, 'gt', gt_path)

        cond_xstart_path = self.paths[index]['cond_xstart_path']
        cond_xstart_bytes = self.file_client.get(cond_xstart_path, 'cond_xstart')
        cond_xstart = _load_feature_tensor(
            cond_xstart_bytes, 'cond_xstart', cond_xstart_path)

        uncond_xstart_path = self.paths[index]['uncond_xstart_path']
        uncond_xstart_bytes = self.file_client.get(uncond_xstart_path, 'uncond_xstart')
        uncond_xstart = _load_feature_tensor(
            uncond_xstart_bytes, 'uncond_xstart', uncond_xstart_path)

        final_latents_path = self.paths[index]['final_latents_path']
        final_latents_bytes = self.file_client.get(final_latents_path, 'final_latents')
        final_latents = _load_feature_tensor(
            final_latents_bytes, 'final_latents', final_latents_path)

        prompt_path = self.paths[index]['prompt_path']
        try:
            prompt = _decode_prompt_bytes(self.file_client.get(prompt_path, 'prompt'))
        except Exception:
            raise Exception("prompt path {} not working".format(prompt_path))

        if self.mean is not None or self.std is not None:
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'gt': img_gt,
            'cond_xstart': cond_xstart,
            'uncond_xstart': uncond_xstart,
            'final_latents': final_latents,
            'prompt': prompt,
            'gt_path': gt_path,
            'cond_xstart_path': cond_xstart_path,
            'uncond_xstart_path': uncond_xstart_path,
            'final_latents_path': final_latents_path,
            'prompt_path': prompt_path,
        }

    def __len__(self):
        return len(self.paths)
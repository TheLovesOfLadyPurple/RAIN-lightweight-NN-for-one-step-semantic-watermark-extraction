from torch.utils import data as data

from .data_util import (multiple_paths_from_folder, multiple_paths_from_lmdb,
                        multiple_paths_from_meta_info_file)
from .file_client import FileClient
from .reverse_distill_dataset import _load_feature_tensor


class SingleNAFNetFeatureDataset(data.Dataset):
    """Load terminal-noise targets and final latent inputs."""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend'].copy()
        self.io_backend_type = self.io_backend_opt.pop('type')
        self.gt_folder = opt['dataroot_gt']
        self.final_latents_folder = opt['dataroot_final_latents']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        folders = [self.gt_folder, self.final_latents_folder]
        keys = ['gt', 'final_latents']
        if self.io_backend_type == 'lmdb':
            self.io_backend_opt['db_paths'] = folders
            self.io_backend_opt['client_keys'] = keys
            self.paths = multiple_paths_from_lmdb(folders, keys)
        elif opt.get('meta_info_file') is not None:
            self.paths = multiple_paths_from_meta_info_file(
                folders, keys, opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = multiple_paths_from_folder(
                folders, keys, self.filename_tmpl)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['file_client'] = None
        return state

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_type, **self.io_backend_opt)

        gt_path = self.paths[index]['gt_path']
        final_latents_path = self.paths[index]['final_latents_path']
        gt = _load_feature_tensor(
            self.file_client.get(gt_path, 'gt'), 'gt', gt_path)
        final_latents = _load_feature_tensor(
            self.file_client.get(final_latents_path, 'final_latents'),
            'final_latents', final_latents_path)

        return {
            'gt': gt,
            'final_latents': final_latents,
            'gt_path': gt_path,
            'final_latents_path': final_latents_path,
        }

    def __len__(self):
        return len(self.paths)

import torch
import pickle
import gzip
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import numpy as np
from .data_util import (paired_paths_from_folder,paired_paths_from_lmdb,
                                    tripled_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from .file_client import FileClient


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


class DistillFeatureDataset(data.Dataset):
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
        super(DistillFeatureDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
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

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        # print('gt path,', torch.Tensor([float(int(gt_path.split('_')[1]))]))
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = torch.from_numpy(pickle.loads(img_bytes)).permute(2,0,1) #imfrombytes(img_bytes, float32=True)
            # print("ttttensor size{}".format(img_gt.size()))
        except:
            raise Exception("gt path {} not working".format(gt_path))
        # print(', lq path', lq_path)
        prompt_path = self.paths[index]['prompt_path']
        # print(', lq path', lq_path)
        str_bytes = self.file_client.get(prompt_path, 'prompt')
        try:
            prompt = _decode_prompt_bytes(str_bytes)
        except Exception:
            raise Exception("prompt path {} not working".format(prompt_path))
        

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq],
        #                             bgr2rgb=True,
        #                             float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'gt': img_gt,
            'prompt': prompt,
            'gt_path': gt_path,
            'prompt_path': prompt_path
        }

    def __len__(self):
        return len(self.paths)

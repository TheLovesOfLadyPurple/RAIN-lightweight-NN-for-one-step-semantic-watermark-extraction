# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import argparse
from os import path as osp

from data.misc import scandir
from utils.lmdb_util import make_lmdb_from_imgs, make_lmdb_from_imgs_prev
from utils.lmdb_util import make_lmdb_from_tensor, make_lmdb_from_prompt, make_lmdb_from_txt

def prepare_keys(folder_path, suffix='png', recursive=False, exclude=None):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.
        suffix (str): File suffix to include.
        recursive (bool): Whether to scan subfolders.
        exclude (list[str] | None): Optional list of path patterns to skip.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix=suffix, recursive=recursive)))

    # Normalize and filter any paths that should be skipped (e.g., generated meta files).
    if exclude:
        normalized_exclude = [osp.normpath(pat) for pat in exclude]

        def _should_skip(path):
            npath = osp.normpath(path)
            return any(npath.endswith(pat) or pat in npath for pat in normalized_exclude)

        img_path_list = [p for p in img_path_list if not _should_skip(p)]
    if recursive:
        keys = [img_path.split('.{}'.format(suffix))[0].split('\\')[-1] for img_path in sorted(img_path_list)]
    else:
        keys = [img_path.split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys
  
def create_lmdb_for_distill_feature():
    folder_path = './dataset/nonfixed-pred/Data/gt-12-5.5' #'./datasets/nonfixed-pred/Data/gt'#'./datasets/nonfixed-pred/val/gt'
    lmdb_path = './dataset/nonfixed-pred/Data/gt.lmdb' #'./datasets/nonfixed-pred/Data/gt_crops.lmdb'#'./datasets/nonfixed-pred/val/gt_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './dataset/nonfixed-pred/Data/prompt-12-5.5' 
    lmdb_path = './dataset/nonfixed-pred/Data/prompt.lmdb'
    txt_path_list, keys = prepare_keys(
        folder_path,
        'txt',
        recursive=True
    )
    make_lmdb_from_txt(folder_path, lmdb_path, txt_path_list, keys)
    # folder_path = './dataset/nonfixed-pred/val/gt-12-5.5' #'./datasets/nonfixed-pred/Data/gt'#'./datasets/nonfixed-pred/val/gt'
    # lmdb_path = './dataset/nonfixed-pred/val/gt.lmdb' #'./datasets/nonfixed-pred/Data/gt_crops.lmdb'#'./datasets/nonfixed-pred/val/gt_crops.lmdb'

    # img_path_list, keys = prepare_keys(folder_path, 'pth')
    # make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)

    # folder_path = './dataset/nonfixed-pred/val/prompt-12-5.5' 
    # lmdb_path = './dataset/nonfixed-pred/val/prompt.lmdb'
    # txt_path_list, keys = prepare_keys(
    #     folder_path,
    #     'txt',
    #     recursive=True
    # )
    # make_lmdb_from_txt(folder_path, lmdb_path, txt_path_list, keys)
    
def create_lmdb_for_reverse_distill_feature():
    base_path = './gen_img_val_v15_coco2014_unipc_low'

    tensor_sources = [
        ('x_T', 'noise_gt.lmdb'),
        ('cond_xstart_0', 'cond_xstart_0.lmdb'),
        ('cond_xstart_1', 'cond_xstart_1.lmdb'),
        ('uncond_xstart_0', 'uncond_xstart_0.lmdb'),
        ('uncond_xstart_1', 'uncond_xstart_1.lmdb'),
        ('gt-12-5.5', 'final_latents.lmdb'),
    ]

    for folder_name, lmdb_name in tensor_sources:
        folder_path = osp.join(base_path, folder_name)
        lmdb_path = osp.join(base_path, lmdb_name)

        img_path_list, keys = prepare_keys(folder_path, 'pth')
        make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)

    folder_path = osp.join(base_path, 'prompt-12-5.5')
    lmdb_path = osp.join(base_path, 'prompt.lmdb')
    txt_path_list, keys = prepare_keys(
        folder_path,
        'txt',
        recursive=True
    )
    make_lmdb_from_txt(folder_path, lmdb_path, txt_path_list, keys)


def create_lmdb_for_pd12m():
    folder_path = './pd12m'
    lmdb_path = './pd12m/train_img.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'jpg', recursive=True)
    invalid_log = None#osp.join(folder_path, 'invalid_images.txt')
    invalid_imgs = make_lmdb_from_imgs_prev(
        folder_path,
        lmdb_path,
        img_path_list,
        keys
    ) 

    folder_path = './pd12m'
    lmdb_path = './pd12m/train_txt.lmdb'
    txt_path_list, keys = prepare_keys(
        folder_path,
        'txt',
        recursive=True
    )
    make_lmdb_from_txt(folder_path, lmdb_path, txt_path_list, keys)

    # folder_path = './pd12m'
    # lmdb_path = './pd12m/train_img.lmdb'
    # img_path_list, keys = prepare_keys(folder_path, 'jpg', recursive=True)
    # invalid_log = None#osp.join(folder_path, 'invalid_images.txt')
    # invalid_imgs = make_lmdb_from_imgs(
    #     folder_path,
    #     lmdb_path,
    #     img_path_list,
    #     keys,
    #     resize=262144,
    #     invalid_log_path=invalid_log
    # ) or []

    # folder_path = './pd12m'
    # lmdb_path = './pd12m/train_txt.lmdb'
    # txt_excludes = ['meta_info.txt']
    # if invalid_imgs:
    #     # drop paired txt files for any images that failed decoding
    #     txt_excludes.extend([osp.splitext(img)[0] + '.jpg' for img in invalid_imgs])
    # txt_path_list, keys = prepare_keys(
    #     folder_path,
    #     'txt',
    #     recursive=True,
    #     exclude=txt_excludes
    # )
    # make_lmdb_from_txt(folder_path, lmdb_path, txt_path_list, keys)

def create_lmdb_for_reds():
    # folder_path = './datasets/REDS/val/sharp_300'
    # lmdb_path = './datasets/REDS/val/sharp_300.lmdb'
    # img_path_list, keys = prepare_keys(folder_path, 'png')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    #
    # folder_path = './datasets/REDS/val/blur_300'
    # lmdb_path = './datasets/REDS/val/blur_300.lmdb'
    # img_path_list, keys = prepare_keys(folder_path, 'jpg')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/REDS/train/train_sharp'
    lmdb_path = './datasets/REDS/train/train_sharp.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/REDS/train/train_blur_jpeg'
    lmdb_path = './datasets/REDS/train/train_blur_jpeg.lmdb'
    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)


def create_lmdb_for_gopro():
    folder_path = './datasets/GoPro/train/blur_crops'
    lmdb_path = './datasets/GoPro/train/blur_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/GoPro/train/sharp_crops'
    lmdb_path = './datasets/GoPro/train/sharp_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # folder_path = './datasets/GoPro/test/target'
    # lmdb_path = './datasets/GoPro/test/target.lmdb'

    # img_path_list, keys = prepare_keys(folder_path, 'png')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # folder_path = './datasets/GoPro/test/input'
    # lmdb_path = './datasets/GoPro/test/input.lmdb'

    # img_path_list, keys = prepare_keys(folder_path, 'png')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    
def create_lmdb_for_blured_feature():
    folder_path = './datasets/GoPro/train/blur_crops'
    lmdb_path = './datasets/GoPro/train/blur_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/GoPro/train/sharp_crops'
    lmdb_path = './datasets/GoPro/train/sharp_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)

    # folder_path = './datasets/GoPro/test/target'
    # lmdb_path = './datasets/GoPro/test/target.lmdb'

    # img_path_list, keys = prepare_keys(folder_path, 'png')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    # folder_path = './datasets/GoPro/test/input'
    # lmdb_path = './datasets/GoPro/test/input.lmdb'

    # img_path_list, keys = prepare_keys(folder_path, 'png')
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_rain13k():
    folder_path = './datasets/Rain13k/train/input'
    lmdb_path = './datasets/Rain13k/train/input.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/Rain13k/train/target'
    lmdb_path = './datasets/Rain13k/train/target.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'jpg')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_noisy_feature():
    folder_path = './datasets/nonfixed-pred/Data/input_crops' #'./datasets/nonfixed-pred/Data/input_crops'#'./datasets/nonfixed-pred/val/input_crops'
    lmdb_path = './datasets/nonfixed-pred/Data/input_crops.lmdb'#'./datasets/nonfixed-pred/Data/input_crops.lmdb' #'./datasets/nonfixed-pred/Data/input_crops.lmdb' 

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)
    
    # folder_path = './datasets/nonfixed-pred/val/gt_crops' #'./datasets/nonfixed-pred/Data/gt_crops'#'./datasets/nonfixed-pred/val/gt_crops'
    # lmdb_path = './datasets/nonfixed-pred/val/gt_crops.lmdb' #'./datasets/nonfixed-pred/Data/gt_crops.lmdb'#'./datasets/nonfixed-pred/val/gt_crops.lmdb'

    # img_path_list, keys = prepare_keys(folder_path, 'pth')
    # make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)
    
def create_lmdb_for_attention_feature():
    folder_path = './dataset/nonfixed-pred/Data/intermediate_sd_z' #'./datasets/nonfixed-pred/Data/input'#'./datasets/nonfixed-pred/val/input'
    lmdb_path = './dataset/nonfixed-pred/Data/input.lmdb' #'./datasets/nonfixed-pred/Data/input_crops.lmdb'#'./datasets/nonfixed-pred/val/input_crops.lmdb' 

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './dataset/nonfixed-pred/Data/final_sd_x0' #'./datasets/nonfixed-pred/Data/gt'#'./datasets/nonfixed-pred/val/gt'
    lmdb_path = './dataset/nonfixed-pred/Data/gt.lmdb' #'./datasets/nonfixed-pred/Data/gt_crops.lmdb'#'./datasets/nonfixed-pred/val/gt_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './dataset/nonfixed-pred/Data/sd_prompt' #'./datasets/nonfixed-pred/Data/prompt'#'./datasets/nonfixed-pred/val/prompt'
    lmdb_path = './dataset/nonfixed-pred/Data/prompt.lmdb' #'./datasets/nonfixed-pred/Data/prompt.lmdb'#'./datasets/nonfixed-pred/val/prompt.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_prompt(folder_path, lmdb_path, img_path_list, keys)

    
    folder_path = './dataset/nonfixed-pred/val/intermediate_sd_z' #'./datasets/nonfixed-pred/Data/input'#'./datasets/nonfixed-pred/val/input'
    lmdb_path = './dataset/nonfixed-pred/val/input.lmdb' #'./datasets/nonfixed-pred/Data/input_crops.lmdb'#'./datasets/nonfixed-pred/val/input_crops.lmdb' 

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './dataset/nonfixed-pred/val/final_sd_x0' #'./datasets/nonfixed-pred/Data/gt'#'./datasets/nonfixed-pred/val/gt'
    lmdb_path = './dataset/nonfixed-pred/val/gt.lmdb' #'./datasets/nonfixed-pred/Data/gt_crops.lmdb'#'./datasets/nonfixed-pred/val/gt_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './dataset/nonfixed-pred/val/sd_prompt' #'./datasets/nonfixed-pred/Data/prompt'#'./datasets/nonfixed-pred/val/prompt'
    lmdb_path = './dataset/nonfixed-pred/val/prompt.lmdb' #'./datasets/nonfixed-pred/Data/prompt.lmdb'#'./datasets/nonfixed-pred/val/prompt.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_prompt(folder_path, lmdb_path, img_path_list, keys)
    
def create_lmdb_for_golden_feature():
    folder_path = './datasets/nonfixed-pred/NpnetData/golden_z_gt' #'./datasets/nonfixed-pred/Data/input'#'./datasets/nonfixed-pred/val/input'
    lmdb_path = './datasets/nonfixed-pred/NpnetData/golden_z_gt.lmdb' #'./datasets/nonfixed-pred/Data/input_crops.lmdb'#'./datasets/nonfixed-pred/val/input_crops.lmdb' 

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/nonfixed-pred/NpnetData/golden_z_input' #'./datasets/nonfixed-pred/Data/input'#'./datasets/nonfixed-pred/val/input'
    lmdb_path = './datasets/nonfixed-pred/NpnetData/golden_z_input.lmdb' #'./datasets/nonfixed-pred/Data/gt_crops.lmdb'#'./datasets/nonfixed-pred/val/gt_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)
    
    folder_path = './datasets/nonfixed-pred/NpnetData/normal_z_gt' #'./datasets/nonfixed-pred/Data/input'#'./datasets/nonfixed-pred/val/input'
    lmdb_path = './datasets/nonfixed-pred/NpnetData/normal_z_gt.lmdb' #'./datasets/nonfixed-pred/Data/gt_crops.lmdb'#'./datasets/nonfixed-pred/val/gt_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/nonfixed-pred/NpnetData/normal_z_input' #'./datasets/nonfixed-pred/Data/input'#'./datasets/nonfixed-pred/val/input'
    lmdb_path = './datasets/nonfixed-pred/NpnetData/normal_z_input.lmdb'#'./datasets/nonfixed-pred/Data/gt_crops.lmdb'#'./datasets/nonfixed-pred/val/gt_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_tensor(folder_path, lmdb_path, img_path_list, keys)


    folder_path = './datasets/nonfixed-pred/NpnetData/prompt' #'./datasets/nonfixed-pred/Data/input'#'./datasets/nonfixed-pred/val/input'
    lmdb_path = './datasets/nonfixed-pred/NpnetData/prompt.lmdb' #'./datasets/nonfixed-pred/Data/prompt.lmdb'#'./datasets/nonfixed-pred/val/prompt.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'pth')
    make_lmdb_from_prompt(folder_path, lmdb_path, img_path_list, keys)
    
def create_lmdb_for_SIDD():
    folder_path = './datasets/SIDD/train/input_crops'
    lmdb_path = './datasets/SIDD/train/input_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'PNG')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/SIDD/train/gt_crops'
    lmdb_path = './datasets/SIDD/train/gt_crops.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'PNG')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    #for val
    '''
    
    folder_path = './datasets/SIDD/val/input_crops'
    lmdb_path = './datasets/SIDD/val/input_crops.lmdb'
    mat_path = './datasets/SIDD/ValidationNoisyBlocksSrgb.mat'
    if not osp.exists(folder_path):
        os.makedirs(folder_path)
    assert  osp.exists(mat_path)
    data = scio.loadmat(mat_path)['ValidationNoisyBlocksSrgb']
    N, B, H ,W, C = data.shape
    data = data.reshape(N*B, H, W, C)
    for i in tqdm(range(N*B)):
        cv2.imwrite(osp.join(folder_path, 'ValidationBlocksSrgb_{}.png'.format(i)), cv2.cvtColor(data[i,...], cv2.COLOR_RGB2BGR)) 
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = './datasets/SIDD/val/gt_crops'
    lmdb_path = './datasets/SIDD/val/gt_crops.lmdb'
    mat_path = './datasets/SIDD/ValidationGtBlocksSrgb.mat'
    if not osp.exists(folder_path):
        os.makedirs(folder_path)
    assert  osp.exists(mat_path)
    data = scio.loadmat(mat_path)['ValidationGtBlocksSrgb']
    N, B, H ,W, C = data.shape
    data = data.reshape(N*B, H, W, C)
    for i in tqdm(range(N*B)):
        cv2.imwrite(osp.join(folder_path, 'ValidationBlocksSrgb_{}.png'.format(i)), cv2.cvtColor(data[i,...], cv2.COLOR_RGB2BGR)) 
    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)
    '''

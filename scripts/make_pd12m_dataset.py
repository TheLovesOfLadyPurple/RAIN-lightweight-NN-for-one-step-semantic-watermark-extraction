# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import numpy as np
import os 
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm
import torch
import sys
sys.path.append('E:\\research_project\\DiT')
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from data.misc import scandir_SIDD
from utils.create_lmdb import  create_lmdb_for_attention_feature, create_lmdb_for_pd12m


def main():
    create_lmdb_for_pd12m()


if __name__ == '__main__':
    main()
    # ... make sidd to lmdb
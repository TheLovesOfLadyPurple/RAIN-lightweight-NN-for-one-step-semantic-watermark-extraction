import argparse, os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import accelerate
from pycocotools.coco import COCO
from diffusers import (
    StableDiffusionPipeline,
)
import pandas as pd
import glob
# from huggingface_hub import login
from SVDNoiseUnet import NPNet64
import functools
import random
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import json
import subprocess
import os
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
from sampler import UniPCSampler

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

# New helper to load a list-of-dicts preference JSON
# JSON schema: [ { 'human_preference': [int], 'prompt': str, 'file_path': [str] }, ... ]
def load_preference_json(json_path: str) -> list[dict]:
    """Load records from a JSON file formatted as a list of preference dicts."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# New helper to extract just the prompts from the preference JSON
# Returns a flat list of all 'prompt' values

def extract_prompts_from_pref_json(json_path: str) -> list[str]:
    """Load a JSON of preference records and return only the prompts."""
    records = load_preference_json(json_path)
    return [rec['prompt'] for rec in records]

# Example usage:
# prompts = extract_prompts_from_pref_json("path/to/preference.json")
# print(prompts)

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


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

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def convert_caption_json_to_str(json):
    caption = json["caption"]
    return caption


def model_closure(pipe):
    def model_fn(x, t, c):
        return pipe.unet(x, t, encoder_hidden_states=c).sample

    return model_fn


def save_latent_batch(latents, output_dir, start_idx, visualize, pipe):
    """Save one latent tensor per sample and optionally its VAE-decoded preview."""
    for idx, latent in enumerate(latents):
        torch.save(latent.detach().cpu(), os.path.join(output_dir, f"{start_idx + idx:05}.pth"))

    if not visualize:
        return

    visualization_dir = os.path.join(output_dir, "visualizations")
    with torch.no_grad():
        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)

    for idx, image in enumerate(decoded):
        image = 255.0 * rearrange(image.detach().cpu().numpy(), "c h w -> h w c")
        Image.fromarray(image.astype(np.uint8)).save(
            os.path.join(visualization_dir, f"{start_idx + idx:05}.png")
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./gen_img_val_v15_coco2014_unipc_low"
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        default=True,
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=12,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--stop_steps",
        type=int,
        default=6,
        help="number of stop sampling steps",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=20,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        default='./LAION2',
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--npnet-checkpoint",
        type=str,
        default='./HPSFilterFix.pth',
        help="if specified, load prompts from this file",
    )
    
    parser.add_argument(
        "--use_free_net",
        action='store_true',
        default=True,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--force_not_use_ct",
        action='store_true',
        default=False,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--force_not_use_NPNet",
        action='store_true',
        default=True,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--use_retrain",
        action='store_true',
        default=True,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--use_raw_golden_noise",
        action='store_true',
        default=False,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--inner_lcm_step",
        action='store_true',
        default=4,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--use_8full_trcik",
        action='store_true',
        default=True,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--get_noise",
        action='store_true',
        default=False,
        help="use the free network for inference.",
    )
    parser.add_argument(
        "--visualize_first_step",
        default=False,
        help="save VAE-decoded debug images for x_T and the first x0 predictions.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    
    opt = parser.parse_args()

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    seed_everything(opt.seed)
    seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
    torch.manual_seed(seeds[accelerator.process_index].item())
    
    seed_everything(opt.seed)

    DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse
    device = "cuda" 
    # pipe = StableDiffusionPipeline.from_single_file( "./counterfeit/Counterfeit-V3.0_fp32.safetensors")
    
    # pipe = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4')
    pipe = StableDiffusionPipeline.from_pretrained('sd-legacy/stable-diffusion-v1-5')
    # pipe = StableDiffusionPipeline.from_single_file( "./v1-5-pruned-emaonly.safetensors")
    
    # npn_net = NPNet64('SD1.5', opt.npnet_checkpoint)
    
    pipe.to(device=device, torch_dtype=DTYPE)
    # if opt.use_free_net:
    #     register_free_upblock2d(pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
    #     register_free_crossattn_upblock2d(pipe, b1=1.1, b2=1.1, s1=0.9, s2=0.2)
    sampler = UniPCSampler(pipe
                           , model_closure=model_closure
                           , steps=opt.stop_steps
                           , guidance_scale=opt.scale
                           , is_high_resoulution=False)
    # ts = sampler.unipc_solver.timesteps.cpu().numpy()
    # ts = [t * 999 for t in ts]
    def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, tokenizer, is_train=True):
        prompt_embeds = encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train)
        return {"prompt_embeds": prompt_embeds}
    
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        proportion_empty_prompts=0,
        text_encoder=pipe.text_encoder,
        tokenizer=pipe.tokenizer,
    )


    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    
    
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        # support both COCO json and LAION parquet-like inputs
        input_path = opt.from_file
        _, ext = os.path.splitext(input_path)
        ext = ext.lower()
        print(f"reading prompts from {input_path}")
        if ext in ['.json']:
            # fall back to the original COCO behaviour when a json is provided
            coco_annotation_file_path = input_path
            coco_caption_file_path = './captions_train2014.json'
            coco_annotation = COCO(annotation_file=coco_annotation_file_path)
            coco_caption = COCO(annotation_file=coco_caption_file_path)
            query_names = []
            unselect_names = []

            query_ids = []
            img_ids = coco_annotation.getImgIds()

            unselect_id = []
            unselect_img_ids = []
            for unselect_name in unselect_names:
                unselect_id += coco_annotation.getCatIds(catNms=[unselect_name])
                unselect_img_ids += coco_annotation.getImgIds(catIds=unselect_id)

            real_img_ids = [item for item in img_ids if item not in unselect_img_ids]
            random.shuffle(real_img_ids)
            real_img_ids = real_img_ids[0:60000]

            caption_ids = coco_caption.getAnnIds(imgIds=real_img_ids)
            captions = coco_caption.loadAnns(caption_ids)
            tmp_caption = []
            for idx, caption in enumerate(captions):
                if idx % 5 != 0:
                    continue
                tmp_caption.append(caption)
            captions = tmp_caption
            data = list(map(lambda x: x['caption'], captions))
            data = data[(0):60000]
        elif (ext in ['.parquet', '.snappy', '.snappy.parquet'] or input_path.lower().endswith('.parquet')
              or os.path.isdir(input_path) or any(ch in input_path for ch in ['*', '?', '[', ']']) or ',' in input_path):
            # Accept single parquet, directory of parquets, glob pattern, or comma-separated list
            # We'll collect prompts incrementally from each file until reaching the limit (60k)
            # helper to expand input_path into a list of files
            def expand_paths(path_str):
                paths = []
                if ',' in path_str:
                    parts = [p.strip() for p in path_str.split(',') if p.strip()]
                else:
                    parts = [path_str]
                for p in parts:
                    if os.path.isdir(p):
                        # collect parquet files in directory
                        paths.extend(sorted(glob.glob(os.path.join(p, '*.parquet'))))
                        paths.extend(sorted(glob.glob(os.path.join(p, '*.snappy'))))
                        paths.extend(sorted(glob.glob(os.path.join(p, '*.snappy.parquet'))))
                    elif any(ch in p for ch in ['*', '?', '[', ']']):
                        paths.extend(sorted(glob.glob(p)))
                    else:
                        paths.append(p)
                # dedupe while preserving order
                seen = set()
                out = []
                for p in paths:
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
                return out

            files = expand_paths(input_path)
            if not files:
                raise ValueError(f"No parquet files found for input: {input_path}")

            collected_prompts = []
            start_idx = 0
            prompt_limit = 89600
            for idx, file in enumerate(files):
                if idx < start_idx:
                    continue
                if len(collected_prompts) >= prompt_limit:
                    break
                # try reading parquet
                try:
                    df = pd.read_parquet(file, engine='pyarrow')
                except Exception:
                    try:
                        df = pd.read_parquet(file, engine='fastparquet')
                    except Exception as e:
                        print(f"Warning: failed to read {file}: {e}")
                        continue

                # normalize column names to uppercase for robust access
                cols = {c.upper(): c for c in df.columns}
                text_col = None
                for candidate in ['TEXT', 'CAPTION', 'PROMPT']:
                    if candidate.upper() in cols:
                        text_col = cols[candidate.upper()]
                        break
                if text_col is None:
                    print(f"Warning: Couldn't find a text column in {file}. Available columns: {list(df.columns)}")
                    continue

                prompts = df[text_col].astype(str).fillna("")
                prompts = prompts[prompts.str.strip() != ""].tolist()
                if not prompts:
                    continue

                # optional: shuffle inside each file to mix
                random.shuffle(prompts)
                remaining = prompt_limit - len(collected_prompts)
                if len(prompts) > remaining:
                    prompts = prompts[:remaining]
                collected_prompts.extend(prompts)

            if not collected_prompts:
                raise ValueError("No prompts found in the provided parquet files.")

            # final shuffle and use as data
            random.shuffle(collected_prompts)
            data = collected_prompts
        else:
            raise ValueError(f"Unsupported input file extension: {ext}")

    grouped = [list(t) for t in chunk(data, batch_size)]
    data = grouped

    if opt.stop_steps !=-1:
        folder_name = f"samples-customed-{opt.stop_steps}-unipc"
        if  opt.use_retrain:
            folder_name += "-retrain"
        if opt.use_free_net:
            folder_name += "-free"
        if opt.force_not_use_NPNet:
            folder_name += "-notNPNet"
        if opt.force_not_use_ct:
            folder_name += "-noneCT"
        if opt.use_raw_golden_noise:
            folder_name += "-rawGoldenNoise"
        if opt.use_8full_trcik:
            folder_name += "-full-trick"
        
        folder_name +=f"-{opt.scale}"
        sample_path = os.path.join(outpath, folder_name)
    elif opt.stop_steps == -1:
        folder_name = f"samples-org-{opt.ddim_steps}"
        if opt.use_free_net:
            folder_name += "-free"
        if opt.force_not_use_NPNet:
            folder_name += "-notNPNet"
        if opt.use_raw_golden_noise:
            folder_name += "-rawGoldenNoise"
        sample_path = os.path.join(outpath, folder_name)
    # npn_net = NPNet64('SD1.5', opt.npnet_checkpoint)
    
    gt_path = os.path.join(outpath, f'gt-{opt.ddim_steps}-{opt.scale}')
    prompt_path = os.path.join(outpath, f'prompt-{opt.ddim_steps}-{opt.scale}')
    x_t_path = os.path.join(outpath, 'x_T')
    cond_xstart_0_path = os.path.join(outpath, 'cond_xstart_0')
    cond_xstart_1_path = os.path.join(outpath, 'cond_xstart_1')
    uncond_xstart_0_path = os.path.join(outpath, 'uncond_xstart_0')
    uncond_xstart_1_path = os.path.join(outpath, 'uncond_xstart_1')
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(prompt_path, exist_ok=True)
    for feature_path in (
            x_t_path,
            cond_xstart_0_path,
            cond_xstart_1_path,
            uncond_xstart_0_path,
            uncond_xstart_1_path):
        os.makedirs(feature_path, exist_ok=True)
        if opt.visualize_first_step:
            os.makedirs(os.path.join(feature_path, 'visualizations'), exist_ok=True)

    get_gt_noise = opt.get_noise
    base_count = len(os.listdir(sample_path))
    direct_distill_intermediate_count = 0
    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            tic = time.time()
            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling", disable =not accelerator.is_main_process):
                for prompts in tqdm(data, desc="data", disable=not accelerator.is_main_process):
                    # torch.cuda.empty_cache()
                    intermediate_photos = list()
                    # prompts = prompts[0]
                            
                    # if isinstance(prompts, tuple) or isinstance(prompts, str):
                    #     prompts = list(prompts)
                    if isinstance(prompts, str):
                        prompts = prompts #+ 'high quality, best quality, masterpiece, 4K, highres, extremely detailed, ultra-detailed'
                        prompts = (prompts,)
                    if isinstance(prompts, tuple) or isinstance(prompts, str):
                        prompts = list(prompts)
                    encoded_text = compute_embeddings_fn(prompts)
                    uc = None
                    if opt.scale != 1.0:
                        uc = compute_embeddings_fn(batch_size * [""])
                    uc = uc.pop("prompt_embeds") if uc is not None else None
                    c =  encoded_text.pop("prompt_embeds")
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    
                    
                    x = torch.randn([opt.n_samples, *shape], device=device) 
                    # if (opt.stop_steps != -1 or opt.ddim_steps <= 8) and not opt.force_not_use_NPNet:
                    #     x = npn_net(x,c)
                        
                    grather_feature_dict = {
                        'cond_noise': [] ,
                        'uncond_noise': [],
                        'cond_xstart': [] ,
                        'uncond_xstart': [],
                        'intermediate_x': [] ,
                        'tmp_t':[]
                    }
                    samples, _ = sampler.sample(
                        conditioning=c,
                        batch_size=opt.n_samples,
                        shape=shape,
                        unconditional_conditioning=uc,
                        x_T=x,
                        start_free_u_step=4 if opt.use_free_net else -1,
                        use_corrector=True,
                        grather_feature_dict=grather_feature_dict
                    )
                    start_idx = base_count
                    save_latent_batch(x, x_t_path, start_idx, opt.visualize_first_step, pipe)
                    if len(grather_feature_dict['cond_xstart']) > 0:
                        save_latent_batch(
                            grather_feature_dict['cond_xstart'][0],
                            cond_xstart_0_path,
                            start_idx,
                            opt.visualize_first_step,
                            pipe,
                        )
                    if len(grather_feature_dict['cond_xstart']) > 1:
                        save_latent_batch(
                            grather_feature_dict['cond_xstart'][1],
                            cond_xstart_1_path,
                            start_idx,
                            opt.visualize_first_step,
                            pipe,
                        )
                    if len(grather_feature_dict['uncond_xstart']) > 0:
                        save_latent_batch(
                            grather_feature_dict['uncond_xstart'][0],
                            uncond_xstart_0_path,
                            start_idx,
                            opt.visualize_first_step,
                            pipe,
                        )
                    if len(grather_feature_dict['uncond_xstart']) > 1:
                        save_latent_batch(
                            grather_feature_dict['uncond_xstart'][1],
                            uncond_xstart_1_path,
                            start_idx,
                            opt.visualize_first_step,
                            pipe,
                        )
                    cond_idx = random.randint(0, len(grather_feature_dict["cond_noise"])-1 if len(grather_feature_dict["cond_noise"])>0 else 0)
                    uncond_idx = random.randint(0, len(grather_feature_dict["uncond_noise"])-1 if len(grather_feature_dict["uncond_noise"])>0 else 0)
                    theshold = 0.2
                    record_uncond_noise: bool = random.uniform(0,1) < theshold
                        
                    x_samples_ddim = pipe.vae.decode(samples / pipe.vae.config.scaling_factor).sample
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    if not opt.skip_save:
                        for idx, x_sample in enumerate(x_samples_ddim):
                            filename = f"{start_idx + idx:05}.png"
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(
                                os.path.join(sample_path, filename))

                    # Save prompts outside the image-save condition so they are always recorded
                    if prompts:
                        for idx in range(len(x_samples_ddim)):
                            prompt_text = str(prompts[idx % len(prompts)])
                            prompt_file = os.path.join(prompt_path, f"{start_idx + idx:05}.txt")
                            with open(prompt_file, "w", encoding="utf-8") as pf:
                                pf.write(prompt_text)

                    base_count += len(x_samples_ddim)

                    for idx, img in enumerate(samples):
                        img = img.permute(1,2,0)
                        torch.save(img,os.path.join(gt_path, f"{start_idx + idx:05}.pth"))
                        direct_distill_intermediate_count += 1
                    
                    

            toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()

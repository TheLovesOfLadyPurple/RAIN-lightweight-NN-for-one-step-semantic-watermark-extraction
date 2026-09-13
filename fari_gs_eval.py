"""Evaluate a FARI adapter checkpoint over all Gaussian-Shading attacks."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm

from coco2017_diffusers_unipc_gaussian_shading_adv_eval import ATTACK_NAMES, apply_attack, pil_to_tensor
from coco2017_diffusers_unipc_gaussian_shading_eval import (
    DEFAULT_CAPTIONS, load_coco_captions, restore_watermark_state, watermark_state,
)
from fari_inversion import inject_fari_adapters, load_adapter_state_dict, one_step_inversion
from gaussian_shading_watermark import GaussianShadingWatermark
from reverse_distill_gaussian_shading_eval import latent_metrics


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate FARI on all GS image attacks.')
    parser.add_argument('--checkpoint', default=str(PROJECT_ROOT / 'experiments' / 'FARI' / 'fari_adapters.pth'))
    parser.add_argument('--sd-model-id', default='sd-legacy/stable-diffusion-v1-5')
    parser.add_argument('--prompts', default=str(DEFAULT_CAPTIONS))
    parser.add_argument('--outdir', default=str(PROJECT_ROOT / 'outputs' / 'fari_gs_robustness'))
    parser.add_argument('--generation-cache-dir', default=str(PROJECT_ROOT / 'outputs' / 'fari_generation_cache'))
    parser.add_argument('--refresh-generation-cache', action='store_true')
    parser.add_argument('--num', type=int, default=1000)
    parser.add_argument('--inference-steps', type=int, default=20)
    parser.add_argument('--guidance-scale', type=float, default=7.5)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--channel-copy', type=int, default=1)
    parser.add_argument('--hw-copy', type=int, default=8)
    parser.add_argument('--user-number', type=int, default=1_000_000)
    parser.add_argument('--fpr', type=float, default=1e-6)
    parser.add_argument('--jpeg-ratio', type=int, default=25)
    parser.add_argument('--random-crop-ratio', type=float, default=0.6)
    parser.add_argument('--random-drop-ratio', type=float, default=0.8)
    parser.add_argument('--resize-ratio', type=float, default=0.25)
    parser.add_argument('--gaussian-blur-r', type=float, default=4)
    parser.add_argument('--median-blur-k', type=int, default=7)
    parser.add_argument('--gaussian-std', type=float, default=0.05)
    parser.add_argument('--sp-prob', type=float, default=0.05)
    parser.add_argument('--brightness-factor', type=float, default=6)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def cache_path(cache_dir, caption_record, seed, args):
    key = json.dumps({
        'version': 1, 'caption': caption_record, 'seed': seed,
        'model': args.sd_model_id, 'steps': args.inference_steps,
        'guidance': args.guidance_scale, 'channel_copy': args.channel_copy,
        'hw_copy': args.hw_copy, 'user_number': args.user_number, 'fpr': args.fpr,
    }, sort_keys=True).encode('utf-8')
    return cache_dir / f'fari_{caption_record["image_id"]}_{hashlib.sha256(key).hexdigest()[:16]}.pt'


@torch.no_grad()
def load_or_generate(cache_file, prompt, seed, args, pipe, device):
    if cache_file.is_file() and not args.refresh_generation_cache:
        cached = torch.load(cache_file, map_location='cpu')
        if cached.get('version') == 1:
            watermark = GaussianShadingWatermark(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
            restore_watermark_state(watermark, cached['watermark_state'], device)
            return watermark, cached['noise'].to(device), Image.fromarray(cached['image']), True

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    watermark = GaussianShadingWatermark(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = watermark.create_latent(device=device, dtype=torch.float32, generator=generator)
    image = pipe(
        prompt, latents=noise, guidance_scale=args.guidance_scale,
        num_inference_steps=args.inference_steps, height=512, width=512).images[0].convert('RGB')
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_file.with_suffix('.tmp')
    torch.save({'version': 1, 'watermark_state': watermark_state(watermark),
                'noise': noise.cpu(), 'image': np.asarray(image)}, temporary_path)
    temporary_path.replace(cache_file)
    return watermark, noise, image, False


def aggregate(samples):
    report = {}
    for attack_name in ATTACK_NAMES:
        entries = [sample['attacks'][attack_name] for sample in samples]
        report[attack_name] = {
            'sample_count': len(entries),
            'mean_bit_accuracy': sum(entry['bit_accuracy'] for entry in entries) / len(entries),
            'detection_rate': sum(entry['detected'] for entry in entries) / len(entries),
            'traceability_rate': sum(entry['traceable'] for entry in entries) / len(entries),
            'mean_recovered_noise_mse': sum(entry['noise_metrics']['mean_l2'] for entry in entries) / len(entries),
            'mean_cosine_similarity': sum(entry['noise_metrics']['cosine_similarity'] for entry in entries) / len(entries),
        }
    return report


def save_csv(report, path):
    fields = ['attack', 'sample_count', 'mean_bit_accuracy', 'detection_rate',
              'traceability_rate', 'mean_recovered_noise_mse', 'mean_cosine_similarity']
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for attack_name, values in report.items():
            writer.writerow({'attack': attack_name, **values})


def main():
    args = parse_args()
    if args.device == 'cpu':
        raise ValueError('FARI Gaussian-Shading evaluation requires CUDA.')
    if args.num < 1:
        raise ValueError('--num must be positive.')
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'FARI checkpoint not found: {checkpoint}')

    device = torch.device(args.device)
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_model_id, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    inject_fari_adapters(pipe.unet, args.lora_rank)
    load_adapter_state_dict(pipe.unet, torch.load(checkpoint, map_location='cpu'))
    pipe.unet.eval()
    null_prompt_embedding, _ = pipe.encode_prompt('', device, 1, False)

    cache_dir = Path(args.generation_cache_dir)
    samples = []
    prompts = load_coco_captions(Path(args.prompts), args.num, args.seed)
    for index, caption_record in enumerate(tqdm(prompts, unit='sample', desc='FARI GS evaluation')):
        prompt = caption_record['caption']
        seed = args.seed + index
        watermark, target_noise, image, cache_hit = load_or_generate(
            cache_path(cache_dir, caption_record, seed, args), prompt, seed, args, pipe, device)
        attacks = {}
        for attack_index, attack_name in enumerate(ATTACK_NAMES):
            attacked_image = apply_attack(image, attack_name, args, seed + 10_007 * attack_index)
            with torch.no_grad():
                latent = pipe.vae.encode(
                    (pil_to_tensor(attacked_image, device).unsqueeze(0) * 2 - 1)
                ).latent_dist.mode() * pipe.vae.config.scaling_factor
                prediction = one_step_inversion(pipe, latent, null_prompt_embedding.detach())
            attacks[attack_name] = {
                **watermark.evaluate(prediction).to_dict(),
                'noise_metrics': latent_metrics(prediction, target_noise),
            }
        samples.append({'index': index, **caption_record, 'generation_cache_hit': cache_hit, 'attacks': attacks})

    result = {'configuration': vars(args), 'aggregate': aggregate(samples), 'samples': samples}
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
    save_csv(result['aggregate'], output_dir / 'aggregate.csv')
    print(json.dumps(result['aggregate'], indent=2))


if __name__ == '__main__':
    main()
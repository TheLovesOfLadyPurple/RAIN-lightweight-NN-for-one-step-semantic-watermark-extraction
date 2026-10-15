"""Evaluate the adversarial reverse extractor against all FARI-style attacks."""

import argparse
import csv
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, UniPCMultistepScheduler
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

from coco2017_diffusers_unipc_gaussian_shading_eval import (
    DEFAULT_CAPTIONS,
    generation_cache_path,
    load_coco_captions,
    load_or_generate_sample,
)
from reverse_distill_gaussian_shading_eval import (
    build_model,
    latent_metrics,
    prediction_diagnostics,
    reverse_noise,
)
from reverse_distill_unipc_eval import convert_reverse_model_to_float32, prompt_embeddings
from fari_inversion import (
    inject_fari_adapters,
    load_official_fari_state_dict,
    one_step_inversion as fari_one_step_inversion,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OPTIONS = PROJECT_ROOT / 'options' / 'test' / 'cocoGSReverseDistillUNetAdv.yml'
ATTACK_NAMES = (
    'clean_roundtrip', 'jpeg', 'random_crop', 'random_drop', 'resize',
    'gaussian_blur', 'median_blur', 'gaussian_noise', 'salt_pepper',
    'brightness')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate one adversarial reverse model against all FARI-style attacks.')
    parser.add_argument('--opt', default=str(DEFAULT_OPTIONS))
    parser.add_argument('--sd-model-id', default=None)
    parser.add_argument(
        '--include-fari',
        action='store_true',
        help='Also evaluate the official FARI checkpoint on each attacked image.')
    parser.add_argument(
        '--fari-checkpoint',
        default=str(PROJECT_ROOT / 'fari_weights.pth'),
        help='Official FARI lora_diffusion checkpoint to compare.')
    parser.add_argument(
        '--fari-sd-model-id',
        default='stabilityai/stable-diffusion-2-1-base',
        help='Stable Diffusion backbone used to train the supplied FARI checkpoint.')
    parser.add_argument('--fari-lora-rank', type=int, default=8)
    parser.add_argument('--captions', default=str(DEFAULT_CAPTIONS))
    parser.add_argument(
        '--outdir',
        default=str(PROJECT_ROOT / 'outputs' / 'coco_gs_adversarial_robustness'))
    parser.add_argument('--generation-cache-dir', default=None)
    parser.add_argument('--refresh-generation-cache', action='store_true')
    parser.add_argument('--num', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--guidance-scale', type=float, default=5.5)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--caption-seed', type=int, default=0)
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


def tensor_to_pil(image):
    pixels = (image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
              * 255).round().astype(np.uint8)
    return Image.fromarray(pixels, mode='RGB')


def pil_to_tensor(image, device):
    pixels = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
    return torch.from_numpy(pixels).permute(2, 0, 1).to(device=device)


def apply_attack(image, attack_name, args, seed):
    """Use FARI's ten-condition protocol with reproducible attack randomness."""
    image = image.convert('RGB')
    width, height = image.size
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if attack_name == 'clean_roundtrip':
        return image
    if attack_name == 'jpeg':
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=args.jpeg_ratio)
        buffer.seek(0)
        result = Image.open(buffer).convert('RGB')
        result.load()
        return result
    if attack_name == 'random_crop':
        crop_width = max(1, int(width * args.random_crop_ratio))
        crop_height = max(1, int(height * args.random_crop_ratio))
        left = rng.randrange(width - crop_width + 1)
        top = rng.randrange(height - crop_height + 1)
        source = np.asarray(image)
        result = np.zeros_like(source)
        result[top:top + crop_height, left:left + crop_width] = (
            source[top:top + crop_height, left:left + crop_width])
        return Image.fromarray(result)
    if attack_name == 'random_drop':
        drop_width = max(1, int(width * args.random_drop_ratio))
        drop_height = max(1, int(height * args.random_drop_ratio))
        left = rng.randrange(width - drop_width + 1)
        top = rng.randrange(height - drop_height + 1)
        result = np.asarray(image).copy()
        result[top:top + drop_height, left:left + drop_width] = 0
        return Image.fromarray(result)
    if attack_name == 'resize':
        resized = image.resize(
            (max(1, int(width * args.resize_ratio)),
             max(1, int(height * args.resize_ratio))), Image.Resampling.BILINEAR)
        return resized.resize((width, height), Image.Resampling.BILINEAR)
    if attack_name == 'gaussian_blur':
        return image.filter(ImageFilter.GaussianBlur(args.gaussian_blur_r))
    if attack_name == 'median_blur':
        kernel = max(3, int(args.median_blur_k) | 1)
        return image.filter(ImageFilter.MedianFilter(kernel))
    if attack_name == 'gaussian_noise':
        result = np.asarray(image, dtype=np.float32)
        result += np_rng.normal(0, args.gaussian_std * 255, result.shape)
        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    if attack_name == 'salt_pepper':
        result = np.asarray(image).copy()
        values = np_rng.random(result.shape)
        result[values < args.sp_prob / 2] = 0
        result[values > 1 - args.sp_prob / 2] = 255
        return Image.fromarray(result)
    if attack_name == 'brightness':
        factor = rng.uniform(1 / args.brightness_factor, args.brightness_factor)
        return ImageEnhance.Brightness(image).enhance(factor)
    raise ValueError(f'Unknown attack: {attack_name}')


@torch.no_grad()
def encode_image(pipe, image, dtype):
    vae_input = pil_to_tensor(image, pipe.device).unsqueeze(0) * 2 - 1
    latent = pipe.vae.encode(vae_input.to(dtype=dtype)).latent_dist.mode()
    return latent * pipe.vae.config.scaling_factor


def aggregate_attack_results(results, extractor_name='ours'):
    aggregate = {}
    for attack_name in ATTACK_NAMES:
        entries = [sample['attacks'][attack_name][extractor_name] for sample in results]
        aggregate[attack_name] = {
            'sample_count': len(entries),
            'mean_bit_accuracy': sum(item['bit_accuracy'] for item in entries) / len(entries),
            'detection_rate': sum(item['detected'] for item in entries) / len(entries),
            'traceability_rate': sum(item['traceable'] for item in entries) / len(entries),
            'mean_recovered_noise_mse': sum(item['noise_metrics']['mean_l2'] for item in entries) / len(entries),
            'mean_recovered_noise_l1': sum(item['noise_metrics']['mean_l1'] for item in entries) / len(entries),
            'mean_cosine_similarity': sum(item['noise_metrics']['cosine_similarity'] for item in entries) / len(entries),
            'mean_raw_sign_agreement': sum(
                item['prediction_diagnostics']['raw_sign_agreement'] for item in entries) / len(entries),
        }
    return aggregate


def save_csv(aggregates, path):
    fieldnames = ['extractor', 'attack', 'sample_count', 'mean_bit_accuracy', 'detection_rate',
                  'traceability_rate', 'mean_recovered_noise_mse',
                  'mean_recovered_noise_l1', 'mean_cosine_similarity',
                  'mean_raw_sign_agreement']
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for extractor_name, aggregate in aggregates.items():
            for attack_name, metrics in aggregate.items():
                writer.writerow({'extractor': extractor_name, 'attack': attack_name, **metrics})


def load_fari_pipeline(args, device, dtype):
    checkpoint = Path(args.fari_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'FARI checkpoint not found: {checkpoint}')
    pipe = StableDiffusionPipeline.from_pretrained(args.fari_sd_model_id, torch_dtype=dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    inject_fari_adapters(pipe.unet, args.fari_lora_rank)
    load_official_fari_state_dict(
        pipe.unet, torch.load(checkpoint, map_location='cpu'))
    pipe.unet.eval()
    null_prompt_embedding, _ = pipe.encode_prompt('', device, 1, False)
    return pipe, null_prompt_embedding.detach()


def main():
    args = parse_args()
    if args.num < 1:
        raise ValueError('--num must be positive.')
    if not 12 <= args.steps <= 15:
        raise ValueError('--steps must be between 12 and 15.')
    if args.height != 512 or args.width != 512:
        raise ValueError('Gaussian Shading robustness evaluation requires 512x512.')
    if args.device == 'cpu':
        raise ValueError('Gaussian Shading evaluation requires CUDA.')
    if args.include_fari and args.fari_lora_rank < 1:
        raise ValueError('--fari-lora-rank must be positive.')

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (Path(args.generation_cache_dir) if args.generation_cache_dir
                 else PROJECT_ROOT / 'outputs' / 'coco_generation_cache')
    captions = load_coco_captions(Path(args.captions), args.num, args.caption_seed)
    device = torch.device(args.device)
    dtype = torch.float32
    model, opt = build_model(args)
    convert_reverse_model_to_float32(model)
    pipe = StableDiffusionPipeline.from_pretrained(opt['sd_model_id'], torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    fari_pipe = None
    fari_null_prompt_embedding = None
    if args.include_fari:
        fari_pipe, fari_null_prompt_embedding = load_fari_pipeline(args, device, dtype)

    results = []
    progress_bar = tqdm(captions, unit='sample', desc='COCO adversarial GS evaluation')
    for index, caption_record in enumerate(progress_bar):
        prompt = caption_record['caption']
        prompt_embedding = prompt_embeddings(pipe, [prompt], device, dtype)
        seed = args.seed + index
        cache_path = generation_cache_path(cache_dir, caption_record, seed, args, opt['sd_model_id'])
        (watermark, watermarked_noise, generated_latent, generated_image,
         _, _, _, used_generation_cache) = load_or_generate_sample(
             cache_path, prompt, seed, args, pipe, device, dtype)
        attacks = {}
        base_image = tensor_to_pil(generated_image[0])
        for attack_index, attack_name in enumerate(ATTACK_NAMES):
            attacked_image = apply_attack(base_image, attack_name, args, seed + 10_007 * attack_index)
            attacked_latent = encode_image(pipe, attacked_image, dtype)
            reverse_result = reverse_noise(model, attacked_latent, prompt_embedding)
            watermark_result = watermark.evaluate(reverse_result['predicted_noise'])
            attacks[attack_name] = {'ours': {
                **watermark_result.to_dict(),
                'noise_metrics': latent_metrics(
                    reverse_result['predicted_noise'], watermarked_noise),
                'prediction_diagnostics': prediction_diagnostics(
                    reverse_result['predicted_noise'], watermarked_noise),
            }}
            if fari_pipe is not None:
                fari_latent = encode_image(fari_pipe, attacked_image, dtype)
                fari_prediction = fari_one_step_inversion(
                    fari_pipe, fari_latent, fari_null_prompt_embedding)
                fari_watermark_result = watermark.evaluate(fari_prediction)
                attacks[attack_name]['fari'] = {
                    **fari_watermark_result.to_dict(),
                    'noise_metrics': latent_metrics(fari_prediction, watermarked_noise),
                    'prediction_diagnostics': prediction_diagnostics(
                        fari_prediction, watermarked_noise),
                }
        results.append({
            'index': index,
            'seed': seed,
            **caption_record,
            'generation_cache_path': str(cache_path.resolve()),
            'used_generation_cache': used_generation_cache,
            'attacks': attacks,
        })
        progress_bar.set_postfix_str(
            f'cache={"hit" if used_generation_cache else "miss"}')

    ours_aggregate = aggregate_attack_results(results, 'ours')
    aggregates = {'ours': ours_aggregate}
    if fari_pipe is not None:
        aggregates['fari'] = aggregate_attack_results(results, 'fari')
    report = {
        'configuration': {
            'option': str(Path(args.opt).resolve()),
            'caption_file': str(Path(args.captions).resolve()),
            'sample_count': args.num,
            'steps': args.steps,
            'guidance_scale': args.guidance_scale,
            'generation_cache_dir': str(cache_dir.resolve()),
            'extractors': list(aggregates),
            'fari_checkpoint': (str(Path(args.fari_checkpoint).resolve())
                                if args.include_fari else None),
            'fari_sd_model_id': args.fari_sd_model_id if args.include_fari else None,
            'comparison_note': (
                'FARI is evaluated on the same cached SD v1.5 images and attack '
                'realizations. The supplied official FARI checkpoint has 1024-dimensional '
                'cross-attention and is therefore an SD 2.1-base checkpoint; its backbone '
                'and VAE differ from the SD v1.5 model used to generate these images.'
                if args.include_fari else None),
            'attack_order': ATTACK_NAMES,
            'attack_parameters': {
                'jpeg_ratio': args.jpeg_ratio,
                'random_crop_ratio': args.random_crop_ratio,
                'random_drop_ratio': args.random_drop_ratio,
                'resize_ratio': args.resize_ratio,
                'gaussian_blur_r': args.gaussian_blur_r,
                'median_blur_k': args.median_blur_k,
                'gaussian_std': args.gaussian_std,
                'sp_prob': args.sp_prob,
                'brightness_factor': args.brightness_factor,
            },
        },
        'aggregate': aggregates if fari_pipe is not None else ours_aggregate,
        'samples': results,
    }
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    save_csv(aggregates, output_dir / 'aggregate.csv')
    print(json.dumps(aggregates if fari_pipe is not None else ours_aggregate, indent=2))
    print(f'Saved robustness results to {output_dir}')


if __name__ == '__main__':
    main()
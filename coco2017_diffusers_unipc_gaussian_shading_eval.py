"""Evaluate Gaussian Shading on COCO2014 captions with Diffusers UniPC."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from tqdm import tqdm

from diffusers_unipc_gaussian_shading_eval import (
    generate_latent,
    save_sample_artifacts,
)
from gaussian_shading_watermark import GaussianShadingWatermark
from reverse_distill_gaussian_shading_eval import (
    DEFAULT_OPTIONS,
    build_model,
    latent_metrics,
    mean_metric,
    prediction_diagnostics,
    reverse_noise,
)
from reverse_distill_unipc_eval import (
    convert_reverse_model_to_float32,
    decode_latent,
    prompt_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CAPTIONS = PROJECT_ROOT / 'captions_val2017.json'


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate images from COCO2014 captions with Diffusers UniPC and '
            'test Gaussian-Shading recovery using the OTBD reverse model.'))
    parser.add_argument('--opt', default=str(DEFAULT_OPTIONS))
    parser.add_argument('--sd-model-id', default=None)
    parser.add_argument('--captions', default=str(DEFAULT_CAPTIONS))
    parser.add_argument(
        '--outdir',
        default=str(
            PROJECT_ROOT / 'outputs' / 'coco2014_diffusers_unipc_gaussian_shading'))
    parser.add_argument(
        '--generation-cache-dir',
        default=None,
        help='Directory for generated latents, decoded images, and GS state.')
    parser.add_argument(
        '--refresh-generation-cache',
        action='store_true',
        help='Regenerate and overwrite cached COCO samples.')
    parser.add_argument('--num', type=int, default=1000)
    parser.add_argument(
        '--steps',
        type=int,
        default=15,
        help='Official Diffusers UniPC inference steps; use 12-15.')
    parser.add_argument('--guidance-scale', type=float, default=5.5)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument(
        '--caption-seed',
        type=int,
        default=0,
        help='Seed used to choose captions from distinct COCO images.')
    parser.add_argument('--channel-copy', type=int, default=1)
    parser.add_argument('--hw-copy', type=int, default=8)
    parser.add_argument('--user-number', type=int, default=1_000_000)
    parser.add_argument('--fpr', type=float, default=1e-6)
    parser.add_argument(
        '--use-custom-model',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use the OTBD learned reverse model instead of DDIM inversion.')
    parser.add_argument('--ddim-inverse-steps', type=int, default=50)
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_coco_captions(path, count, seed):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    annotations = data.get('annotations')
    if not isinstance(annotations, list):
        raise ValueError(
            'COCO caption JSON must contain an annotations list.')

    shuffled = list(annotations)
    random.Random(seed).shuffle(shuffled)
    selected = []
    selected_image_ids = set()
    for annotation in shuffled:
        image_id = annotation.get('image_id')
        caption = annotation.get('caption')
        if image_id in selected_image_ids or not isinstance(caption, str):
            continue
        caption = caption.strip()
        if not caption:
            continue
        selected.append({
            'annotation_id': annotation.get('id'),
            'image_id': image_id,
            'caption': caption,
        })
        selected_image_ids.add(image_id)
        if len(selected) == count:
            return selected
    raise ValueError(
        f'Only found {len(selected)} usable captions from distinct images; '
        f'{count} were requested.')


def generation_cache_path(cache_dir, caption_record, seed, args, sd_model_id):
    cache_key = {
        'format_version': 1,
        'caption': caption_record,
        'seed': seed,
        'sd_model_id': sd_model_id,
        'steps': args.steps,
        'guidance_scale': args.guidance_scale,
        'height': args.height,
        'width': args.width,
        'channel_copy': args.channel_copy,
        'hw_copy': args.hw_copy,
        'user_number': args.user_number,
        'fpr': args.fpr,
    }
    encoded_key = json.dumps(cache_key, sort_keys=True).encode('utf-8')
    digest = hashlib.sha256(encoded_key).hexdigest()[:16]
    return cache_dir / f'sample_{caption_record["image_id"]}_{digest}.pt'


def watermark_state(watermark):
    official = watermark._official
    return {
        'key': official.key,
        'nonce': official.nonce,
        'watermark': official.watermark.detach().cpu(),
    }


def restore_watermark_state(watermark, state, device):
    official = watermark._official
    official.key = state['key']
    official.nonce = state['nonce']
    official.watermark = state['watermark'].to(device=device)


@torch.no_grad()
def vae_roundtrip_image(pipe, image):
    quantized_image = (image * 255).round().clamp(0, 255) / 255
    vae_input = quantized_image * 2 - 1
    return (
        pipe.vae.encode(vae_input).latent_dist.mode()
        * pipe.vae.config.scaling_factor)


def load_or_generate_sample(
        cache_path,
        prompt,
        seed,
        args,
        pipe,
        device,
    dtype,
    generated_image_only=False,
    require_cache=False):
    if cache_path.is_file() and not args.refresh_generation_cache:
        cached = torch.load(cache_path, map_location='cpu')
        expected_keys = {
            'watermark_state', 'watermarked_noise', 'generated_latent',
            'generated_image', 'control_noise', 'control_generated_latent',
            'control_generated_image'}
        if cached.get('format_version') == 1 and expected_keys <= cached.keys():
            if generated_image_only:
                return cached['generated_image'].to(device=device, dtype=dtype)
            watermark = GaussianShadingWatermark(
                channel_copy=args.channel_copy,
                hw_copy=args.hw_copy,
                fpr=args.fpr,
                user_number=args.user_number)
            restore_watermark_state(watermark, cached['watermark_state'], device)
            return (
                watermark,
                cached['watermarked_noise'].to(device=device, dtype=dtype),
                cached['generated_latent'].to(device=device, dtype=dtype),
                cached['generated_image'].to(device=device, dtype=dtype),
                cached['control_noise'].to(device=device, dtype=dtype),
                cached['control_generated_latent'].to(device=device, dtype=dtype),
                cached['control_generated_image'].to(device=device, dtype=dtype),
                True)

    if require_cache:
        raise FileNotFoundError(
            f'Required valid generation cache entry not found: {cache_path}')

    generator = torch.Generator(device=device).manual_seed(seed)
    watermark = GaussianShadingWatermark(
        channel_copy=args.channel_copy,
        hw_copy=args.hw_copy,
        fpr=args.fpr,
        user_number=args.user_number)
    watermarked_noise = watermark.create_latent(
        device=device, dtype=dtype, generator=generator)
    generated_latent = generate_latent(
        pipe, prompt, watermarked_noise, args.steps, args.guidance_scale,
        args.height, args.width)
    generated_image = decode_latent(pipe, generated_latent)

    control_generator = torch.Generator(device=device).manual_seed(
        seed + 1_000_000)
    control_noise = torch.randn(
        watermarked_noise.shape,
        generator=control_generator,
        device=device,
        dtype=dtype)
    control_generated_latent = generate_latent(
        pipe, prompt, control_noise, args.steps, args.guidance_scale,
        args.height, args.width)
    control_generated_image = decode_latent(pipe, control_generated_latent)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix('.tmp')
    torch.save({
        'format_version': 1,
        'watermark_state': watermark_state(watermark),
        'watermarked_noise': watermarked_noise.detach().cpu(),
        'generated_latent': generated_latent.detach().cpu(),
        'generated_image': generated_image.detach().cpu(),
        'control_noise': control_noise.detach().cpu(),
        'control_generated_latent': control_generated_latent.detach().cpu(),
        'control_generated_image': control_generated_image.detach().cpu(),
    }, temporary_path)
    temporary_path.replace(cache_path)
    if generated_image_only:
        return generated_image
    return (
        watermark, watermarked_noise, generated_latent, generated_image,
        control_noise, control_generated_latent, control_generated_image, False)


def aggregate_metrics(samples, steps):
    return {
        'sample_count': len(samples),
        'steps': steps,
        'mean_direct_watermark_bit_accuracy': mean_metric(
            samples, ('direct_watermark', 'bit_accuracy')),
        'mean_latent_path_watermark_bit_accuracy': mean_metric(
            samples, ('latent_path_watermark', 'bit_accuracy')),
        'mean_image_roundtrip_watermark_bit_accuracy': mean_metric(
            samples, ('image_roundtrip_watermark', 'bit_accuracy')),
        'latent_path_watermark_detection_rate': mean_metric(
            samples, ('latent_path_watermark', 'detected')),
        'image_roundtrip_watermark_detection_rate': mean_metric(
            samples, ('image_roundtrip_watermark', 'detected')),
        'latent_path_watermark_traceability_rate': mean_metric(
            samples, ('latent_path_watermark', 'traceable')),
        'image_roundtrip_watermark_traceability_rate': mean_metric(
            samples, ('image_roundtrip_watermark', 'traceable')),
        'latent_path_control_false_detection_rate': mean_metric(
            samples, ('latent_path_control', 'detected')),
        'image_roundtrip_control_false_detection_rate': mean_metric(
            samples, ('image_roundtrip_control', 'detected')),
        'mean_direct_control_bit_accuracy': mean_metric(
            samples, ('direct_control', 'bit_accuracy')),
        'mean_latent_path_control_bit_accuracy': mean_metric(
            samples, ('latent_path_control', 'bit_accuracy')),
        'mean_image_roundtrip_control_bit_accuracy': mean_metric(
            samples, ('image_roundtrip_control', 'bit_accuracy')),
        'mean_latent_path_cosine_similarity': mean_metric(
            samples,
            ('latent_path_predicted_vs_watermarked_noise',
             'cosine_similarity')),
        'mean_image_roundtrip_cosine_similarity': mean_metric(
            samples,
            ('image_roundtrip_predicted_vs_watermarked_noise',
             'cosine_similarity')),
        'mean_latent_path_recovered_noise_mse': mean_metric(
            samples,
            ('latent_path_predicted_vs_watermarked_noise', 'mean_l2')),
        'mean_image_roundtrip_recovered_noise_mse': mean_metric(
            samples,
            ('image_roundtrip_predicted_vs_watermarked_noise', 'mean_l2')),
        'mean_latent_path_control_recovered_noise_mse': mean_metric(
            samples,
            ('latent_path_predicted_vs_control_noise', 'mean_l2')),
        'mean_image_roundtrip_control_recovered_noise_mse': mean_metric(
            samples,
            ('image_roundtrip_predicted_vs_control_noise', 'mean_l2')),
    }


def main():
    args = parse_args()
    if args.num < 1:
        raise ValueError('--num must be positive.')
    if not 12 <= args.steps <= 15:
        raise ValueError('--steps must be between 12 and 15.')
    if args.height != 512 or args.width != 512:
        raise ValueError(
            'This Gaussian Shading configuration currently supports only 512x512.')

    captions = load_coco_captions(
        Path(args.captions), args.num, args.caption_seed)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(args.generation_cache_dir)
        if args.generation_cache_dir
        else PROJECT_ROOT / 'outputs' / 'coco_generation_cache')
    device = torch.device(args.device)
    dtype = torch.float32

    model, opt = build_model(args)
    convert_reverse_model_to_float32(model)
    pipe = StableDiffusionPipeline.from_pretrained(
        opt['sd_model_id'], torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    samples = []
    progress_bar = tqdm(
        enumerate(captions),
        total=len(captions),
        unit='sample',
        desc='COCO Gaussian-Shading evaluation')
    for index, caption_record in progress_bar:
        prompt = caption_record['caption']
        prompt_embedding = prompt_embeddings(
            pipe, [prompt], device, dtype)
        seed = args.seed + index
        cache_path = generation_cache_path(
            cache_dir, caption_record, seed, args, opt['sd_model_id'])
        (watermark, watermarked_noise, generated_latent, generated_image,
         control_noise, control_generated_latent, control_generated_image,
         used_generation_cache) = load_or_generate_sample(
             cache_path, prompt, seed, args, pipe, device, dtype)
        progress_bar.set_postfix_str(
            f'cache={"hit" if used_generation_cache else "miss"}')
        latent_reverse_result = reverse_noise(
            model,
            generated_latent,
            prompt_embedding,
            use_custom_model=args.use_custom_model,
            pipe=pipe,
            ddim_inverse_steps=args.ddim_inverse_steps)
        image_roundtrip_input = vae_roundtrip_image(pipe, generated_image)
        image_reverse_result = reverse_noise(
            model,
            image_roundtrip_input,
            prompt_embedding,
            use_custom_model=args.use_custom_model,
            pipe=pipe,
            ddim_inverse_steps=args.ddim_inverse_steps)

        control_latent_reverse_result = reverse_noise(
            model,
            control_generated_latent,
            prompt_embedding,
            use_custom_model=args.use_custom_model,
            pipe=pipe,
            ddim_inverse_steps=args.ddim_inverse_steps)
        control_image_roundtrip_input = vae_roundtrip_image(
            pipe, control_generated_image)
        control_image_reverse_result = reverse_noise(
            model,
            control_image_roundtrip_input,
            prompt_embedding,
            use_custom_model=args.use_custom_model,
            pipe=pipe,
            ddim_inverse_steps=args.ddim_inverse_steps)

        direct_watermark = watermark.evaluate(watermarked_noise)
        latent_path_watermark = watermark.evaluate(
            latent_reverse_result['predicted_noise'])
        image_roundtrip_watermark = watermark.evaluate(
            image_reverse_result['predicted_noise'])
        direct_control = watermark.evaluate(control_noise)
        latent_path_control = watermark.evaluate(
            control_latent_reverse_result['predicted_noise'])
        image_roundtrip_control = watermark.evaluate(
            control_image_reverse_result['predicted_noise'])

        sample_metrics = {
            'index': index,
            'seed': seed,
            **caption_record,
            'steps': args.steps,
            'generation_cache_path': str(cache_path.resolve()),
            'used_generation_cache': used_generation_cache,
            'direct_watermark': direct_watermark.to_dict(),
            'latent_path_watermark': latent_path_watermark.to_dict(),
            'image_roundtrip_watermark': image_roundtrip_watermark.to_dict(),
            'direct_control': direct_control.to_dict(),
            'latent_path_control': latent_path_control.to_dict(),
            'image_roundtrip_control': image_roundtrip_control.to_dict(),
            'latent_path_predicted_vs_watermarked_noise': latent_metrics(
                latent_reverse_result['predicted_noise'], watermarked_noise),
            'image_roundtrip_predicted_vs_watermarked_noise': latent_metrics(
                image_reverse_result['predicted_noise'], watermarked_noise),
            'latent_path_predicted_vs_control_noise': latent_metrics(
                control_latent_reverse_result['predicted_noise'], control_noise),
            'image_roundtrip_predicted_vs_control_noise': latent_metrics(
                control_image_reverse_result['predicted_noise'], control_noise),
            'latent_path_prediction_diagnostics': prediction_diagnostics(
                latent_reverse_result['predicted_noise'], watermarked_noise),
            'image_roundtrip_prediction_diagnostics': prediction_diagnostics(
                image_reverse_result['predicted_noise'], watermarked_noise),
            'latent_path_predicted_timestep': (
                latent_reverse_result['predicted_timestep']
                .float().mean().item()),
            'image_roundtrip_predicted_timestep': (
                image_reverse_result['predicted_timestep']
                .float().mean().item()),
        }
        samples.append(sample_metrics)

        sample_dir = (
            output_dir
            / f'sample_{index:04d}_coco_{caption_record["image_id"]}')
        save_sample_artifacts(
            sample_dir,
            pipe,
            watermarked_noise,
            generated_latent,
            image_roundtrip_input,
            latent_reverse_result,
            image_reverse_result,
            control_noise,
            control_generated_latent,
            control_image_roundtrip_input,
            control_latent_reverse_result,
            control_image_reverse_result,
            generated_image=generated_image,
            control_generated_image=control_generated_image)
        with open(
                sample_dir / 'metrics.json', 'w', encoding='utf-8') as file:
            json.dump(sample_metrics, file, indent=2)

    aggregate = aggregate_metrics(samples, args.steps)
    report = {
        'configuration': {
            'caption_file': str(Path(args.captions).resolve()),
            'caption_selection_seed': args.caption_seed,
            'base_generation_seed': args.seed,
            'scheduler': type(pipe.scheduler).__name__,
            'steps': args.steps,
            'guidance_scale': args.guidance_scale,
            'height': args.height,
            'width': args.width,
            'channel_copy': args.channel_copy,
            'hw_copy': args.hw_copy,
            'fpr': args.fpr,
            'user_number': args.user_number,
            'sd_model_id': opt['sd_model_id'],
            'detector_prompt_mode': 'known COCO generation caption',
            'generation_cache_dir': str(cache_dir.resolve()),
        },
        'aggregate': aggregate,
        'samples': samples,
    }
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    print(json.dumps(aggregate, indent=2))
    print(f'Saved results to {output_dir}')


if __name__ == '__main__':
    main()

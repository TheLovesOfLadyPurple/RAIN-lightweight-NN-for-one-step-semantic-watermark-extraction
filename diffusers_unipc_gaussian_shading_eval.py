"""Test Gaussian Shading with Diffusers' official UniPC scheduler."""

import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

from gaussian_shading_watermark import GaussianShadingWatermark
from reverse_distill_gaussian_shading_eval import (
    DEFAULT_OPTIONS,
    build_model,
    latent_metrics,
    mean_metric,
    prediction_diagnostics,
    reverse_noise,
    vae_roundtrip_latent,
)
from reverse_distill_unipc_eval import (
    convert_reverse_model_to_float32,
    decode_latent,
    prompt_embeddings,
    save_image,
    save_noise_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate Gaussian-Shading images with Diffusers official UniPC '
            'scheduler and test one-step OTBD watermark recovery.'))
    parser.add_argument('--opt', default=str(DEFAULT_OPTIONS))
    parser.add_argument('--sd-model-id', default=None)
    parser.add_argument(
        '--outdir',
        default=str(
            PROJECT_ROOT / 'outputs' / 'diffusers_unipc_gaussian_shading'))
    parser.add_argument('--prompt', default='a smiling woman')
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument(
        '--steps',
        type=int,
        nargs='+',
        default=[12, 15],
        help='Diffusers UniPC inference step counts to evaluate.')
    parser.add_argument('--guidance-scale', type=float, default=5.5)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--channel-copy', type=int, default=1)
    parser.add_argument('--hw-copy', type=int, default=8)
    parser.add_argument('--user-number', type=int, default=1_000_000)
    parser.add_argument('--fpr', type=float, default=1e-6)
    parser.add_argument(
        '--use-custom-model',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Use the OTBD learned reverse model instead of DDIM inversion.')
    parser.add_argument('--ddim-inverse-steps', type=int, default=50)
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


@torch.no_grad()
def generate_latent(pipe, prompt, noise, steps, guidance_scale, height, width):
    return pipe(
        prompt=prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        height=height,
        width=width,
        latents=noise,
        output_type='latent',
    ).images


def save_sample_artifacts(
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
        generated_image=None,
        control_generated_image=None):
    sample_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        'watermarked_noise': watermarked_noise,
        'generated_latent': generated_latent,
        'image_roundtrip_input_latent': image_roundtrip_input,
        'latent_path_predicted_noise': (
            latent_reverse_result['predicted_noise']),
        'image_roundtrip_predicted_noise': (
            image_reverse_result['predicted_noise']),
        'predicted_xstart': image_reverse_result['predicted_xstart'],
        'predicted_xt': image_reverse_result['predicted_xt'],
        'control_noise': control_noise,
        'control_generated_latent': control_generated_latent,
        'control_image_roundtrip_input_latent': control_image_roundtrip_input,
        'control_latent_path_predicted_noise': (
            control_latent_reverse_result['predicted_noise']),
        'control_image_roundtrip_predicted_noise': (
            control_image_reverse_result['predicted_noise']),
    }
    for name, tensor in tensors.items():
        torch.save(tensor.detach().cpu(), sample_dir / f'{name}.pt')
        save_noise_image(tensor, sample_dir / f'{name}_0_255.png')

    if generated_image is None:
        generated_image = decode_latent(pipe, generated_latent)
    if control_generated_image is None:
        control_generated_image = decode_latent(pipe, control_generated_latent)
    save_image(generated_image[0], sample_dir / 'generated_watermarked.png')
    save_image(control_generated_image[0], sample_dir / 'generated_control.png')


def evaluate_step(args, pipe, model, prompt_embedding, device, dtype, steps):
    step_output_dir = Path(args.outdir) / f'step_{steps}'
    step_output_dir.mkdir(parents=True, exist_ok=True)
    samples = []

    for index in range(args.num):
        seed = args.seed + index
        generator = torch.Generator(device=device).manual_seed(seed)
        watermark = GaussianShadingWatermark(
            channel_copy=args.channel_copy,
            hw_copy=args.hw_copy,
            fpr=args.fpr,
            user_number=args.user_number)
        watermarked_noise = watermark.create_latent(
            device=device, dtype=dtype, generator=generator)
        generated_latent = generate_latent(
            pipe,
            args.prompt,
            watermarked_noise,
            steps,
            args.guidance_scale,
            args.height,
            args.width)
        latent_reverse_result = reverse_noise(
            model,
            generated_latent,
            prompt_embedding,
            use_custom_model=args.use_custom_model,
            pipe=pipe,
            ddim_inverse_steps=args.ddim_inverse_steps)
        image_roundtrip_input = vae_roundtrip_latent(pipe, generated_latent)
        image_reverse_result = reverse_noise(
            model,
            image_roundtrip_input,
            prompt_embedding,
            use_custom_model=args.use_custom_model,
            pipe=pipe,
            ddim_inverse_steps=args.ddim_inverse_steps)

        control_generator = torch.Generator(device=device).manual_seed(
            seed + 1_000_000)
        control_noise = torch.randn(
            watermarked_noise.shape,
            generator=control_generator,
            device=device,
            dtype=dtype)
        control_generated_latent = generate_latent(
            pipe,
            args.prompt,
            control_noise,
            steps,
            args.guidance_scale,
            args.height,
            args.width)
        control_latent_reverse_result = reverse_noise(
            model,
            control_generated_latent,
            prompt_embedding,
            use_custom_model=args.use_custom_model,
            pipe=pipe,
            ddim_inverse_steps=args.ddim_inverse_steps)
        control_image_roundtrip_input = vae_roundtrip_latent(
            pipe, control_generated_latent)
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
            'steps': steps,
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

        sample_dir = step_output_dir / f'sample_{index:04d}'
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
            control_image_reverse_result)
        with open(
                sample_dir / 'metrics.json', 'w', encoding='utf-8') as file:
            json.dump(sample_metrics, file, indent=2)

    aggregate = {
        'sample_count': args.num,
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
    }
    with open(
            step_output_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump({'aggregate': aggregate, 'samples': samples}, file, indent=2)
    return aggregate


def main():
    args = parse_args()
    if args.num < 1:
        raise ValueError('--num must be positive.')
    if args.height != 512 or args.width != 512:
        raise ValueError(
            'This Gaussian Shading configuration currently supports only 512x512.')
    if not args.steps or any(steps < 1 for steps in args.steps):
        raise ValueError('--steps must contain positive step counts.')

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    prompt_embedding = prompt_embeddings(
        pipe, [args.prompt], device, dtype)

    step_results = {}
    for steps in sorted(set(args.steps)):
        step_results[str(steps)] = evaluate_step(
            args,
            pipe,
            model,
            prompt_embedding,
            device,
            dtype,
            steps)

    report = {
        'configuration': {
            'prompt': args.prompt,
            'base_seed': args.seed,
            'steps': sorted(set(args.steps)),
            'scheduler': type(pipe.scheduler).__name__,
            'guidance_scale': args.guidance_scale,
            'height': args.height,
            'width': args.width,
            'channel_copy': args.channel_copy,
            'hw_copy': args.hw_copy,
            'fpr': args.fpr,
            'user_number': args.user_number,
            'sd_model_id': opt['sd_model_id'],
            'detector_prompt_mode': 'known generation prompt',
            'generation_path': (
                'StableDiffusionPipeline with Diffusers '
                'UniPCMultistepScheduler and supplied watermarked latents.'),
            'image_roundtrip_path': (
                'Generated latent is VAE-decoded, quantized to 8-bit, and '
                'VAE-encoded before one-step reverse inference.'),
        },
        'steps': step_results,
    }
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    print(json.dumps(step_results, indent=2))
    print(f'Saved results to {output_dir}')


if __name__ == '__main__':
    main()

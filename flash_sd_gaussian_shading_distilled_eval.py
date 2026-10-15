"""Test OTBD watermark recovery on Flash-SD generated images."""

import argparse
import json
from pathlib import Path

import torch
from diffusers import LCMScheduler, StableDiffusionPipeline

from coco2017_diffusers_unipc_gaussian_shading_eval import load_coco_captions
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
DEFAULT_CAPTIONS = PROJECT_ROOT / 'captions_val2014.json'


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate Gaussian-Shading images with Flash-SD and recover the '
            'watermark with the OTBD reverse-distillation model.'))
    parser.add_argument('--opt', default=str(DEFAULT_OPTIONS))
    parser.add_argument('--sd-model-id', default=None)
    parser.add_argument(
        '--base-model-id', default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--adapter-id', default='jasperai/flash-sd')
    parser.add_argument('--captions', default=str(DEFAULT_CAPTIONS))
    parser.add_argument(
        '--outdir',
        default=str(
            PROJECT_ROOT / 'outputs' / 'flash_sd_gaussian_shading_distilled'))
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--lcm-steps', type=int, default=4)
    parser.add_argument('--guidance-scale', type=float, default=0.0)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--caption-seed', type=int, default=0)
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
def generate_flash_latent(pipe, prompt, noise, args):
    return pipe(
        prompt=prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.lcm_steps,
        height=args.height,
        width=args.width,
        latents=noise,
        output_type='latent',
    ).images


def save_sample_artifacts(
        sample_dir,
        pipe,
        watermarked_noise,
        generated_latent,
        image_latent,
        distilled_result,
        control_noise,
        control_generated_latent,
        control_image_latent,
        control_distilled_result):
    sample_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        'watermarked_noise': watermarked_noise,
        'flash_generated_latent': generated_latent,
        'flash_image_latent': image_latent,
        'distilled_inverted_noise': distilled_result['predicted_noise'],
        'distilled_predicted_xstart': distilled_result['predicted_xstart'],
        'distilled_predicted_xt': distilled_result['predicted_xt'],
        'control_noise': control_noise,
        'control_flash_generated_latent': control_generated_latent,
        'control_flash_image_latent': control_image_latent,
        'control_distilled_inverted_noise': (
            control_distilled_result['predicted_noise']),
    }
    for name, tensor in tensors.items():
        torch.save(tensor.detach().cpu(), sample_dir / f'{name}.pt')
        save_noise_image(tensor, sample_dir / f'{name}_0_255.png')

    generated_image = decode_latent(pipe, generated_latent)
    control_image = decode_latent(pipe, control_generated_latent)
    save_image(generated_image[0], sample_dir / 'flash_watermarked.png')
    save_image(control_image[0], sample_dir / 'flash_control.png')


def aggregate_metrics(samples, args):
    return {
        'sample_count': len(samples),
        'lcm_steps': args.lcm_steps,
        'mean_direct_watermark_bit_accuracy': mean_metric(
            samples, ('direct_watermark', 'bit_accuracy')),
        'mean_distilled_watermark_bit_accuracy': mean_metric(
            samples, ('distilled_watermark', 'bit_accuracy')),
        'distilled_watermark_detection_rate': mean_metric(
            samples, ('distilled_watermark', 'detected')),
        'distilled_watermark_traceability_rate': mean_metric(
            samples, ('distilled_watermark', 'traceable')),
        'distilled_control_false_detection_rate': mean_metric(
            samples, ('distilled_control', 'detected')),
        'distilled_control_false_traceability_rate': mean_metric(
            samples, ('distilled_control', 'traceable')),
        'mean_distilled_control_bit_accuracy': mean_metric(
            samples, ('distilled_control', 'bit_accuracy')),
        'mean_distilled_cosine_similarity': mean_metric(
            samples, ('distilled_predicted_vs_watermarked_noise',
                      'cosine_similarity')),
        'mean_distilled_raw_sign_agreement': mean_metric(
            samples, ('distilled_prediction_diagnostics',
                      'raw_sign_agreement')),
    }


def main():
    args = parse_args()
    if args.num < 1:
        raise ValueError('--num must be positive.')
    if not 2 <= args.lcm_steps <= 4:
        raise ValueError('Flash-SD supports --lcm-steps from 2 through 4.')
    if args.height != 512 or args.width != 512:
        raise ValueError(
            'This Gaussian Shading configuration currently supports only 512x512.')

    captions = load_coco_captions(
        Path(args.captions), args.num, args.caption_seed)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float32

    reverse_model, reverse_opt = build_model(args)
    convert_reverse_model_to_float32(reverse_model)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model_id,
        torch_dtype=dtype,
        use_safetensors=True)
    pipe.load_lora_weights(args.adapter_id)
    pipe.fuse_lora()
    pipe.scheduler = LCMScheduler.from_pretrained(
        args.base_model_id,
        subfolder='scheduler',
        timestep_spacing='trailing')
    pipe.safety_checker = None
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    samples = []
    for index, caption_record in enumerate(captions):
        if not args.use_custom_model and index:
            pipe.load_lora_weights(args.adapter_id)
            pipe.fuse_lora()
        prompt = caption_record['caption']
        prompt_embedding = prompt_embeddings(
            pipe, [prompt], device, dtype)
        seed = args.seed + index
        generator = torch.Generator(device=device).manual_seed(seed)
        watermark = GaussianShadingWatermark(
            channel_copy=args.channel_copy,
            hw_copy=args.hw_copy,
            fpr=args.fpr,
            user_number=args.user_number)
        watermarked_noise = watermark.create_latent(
            device=device, dtype=dtype, generator=generator)
        generated_latent = generate_flash_latent(
            pipe, prompt, watermarked_noise, args)
        image_latent = vae_roundtrip_latent(pipe, generated_latent)
        distilled_result = reverse_noise(
            reverse_model,
            image_latent,
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
        if not args.use_custom_model:
            pipe.load_lora_weights(args.adapter_id)
            pipe.fuse_lora()
        control_generated_latent = generate_flash_latent(
            pipe, prompt, control_noise, args)
        control_image_latent = vae_roundtrip_latent(
            pipe, control_generated_latent)
        control_distilled_result = reverse_noise(
            reverse_model,
            control_image_latent,
            prompt_embedding,
            use_custom_model=args.use_custom_model,
            pipe=pipe,
            ddim_inverse_steps=args.ddim_inverse_steps)

        direct_watermark = watermark.evaluate(watermarked_noise)
        distilled_watermark = watermark.evaluate(
            distilled_result['predicted_noise'])
        distilled_control = watermark.evaluate(
            control_distilled_result['predicted_noise'])
        sample_metrics = {
            'index': index,
            'seed': seed,
            **caption_record,
            'direct_watermark': direct_watermark.to_dict(),
            'distilled_watermark': distilled_watermark.to_dict(),
            'distilled_control': distilled_control.to_dict(),
            'distilled_predicted_vs_watermarked_noise': latent_metrics(
                distilled_result['predicted_noise'], watermarked_noise),
            'distilled_prediction_diagnostics': prediction_diagnostics(
                distilled_result['predicted_noise'], watermarked_noise),
            'distilled_predicted_timestep': (
                distilled_result['predicted_timestep']
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
            image_latent,
            distilled_result,
            control_noise,
            control_generated_latent,
            control_image_latent,
            control_distilled_result)
        with open(
                sample_dir / 'metrics.json', 'w', encoding='utf-8') as file:
            json.dump(sample_metrics, file, indent=2)

    aggregate = aggregate_metrics(samples, args)
    report = {
        'configuration': {
            'caption_file': str(Path(args.captions).resolve()),
            'caption_selection_seed': args.caption_seed,
            'base_generation_seed': args.seed,
            'base_model_id': args.base_model_id,
            'flash_sd_adapter_id': args.adapter_id,
            'generation_scheduler': type(pipe.scheduler).__name__,
            'lcm_steps': args.lcm_steps,
            'guidance_scale': args.guidance_scale,
            'height': args.height,
            'width': args.width,
            'channel_copy': args.channel_copy,
            'hw_copy': args.hw_copy,
            'fpr': args.fpr,
            'user_number': args.user_number,
            'reverse_model_id': reverse_opt['sd_model_id'],
            'detector_prompt_mode': 'known COCO generation caption',
            'image_path': (
                'Flash-SD LCM latent is VAE-decoded, quantized to 8-bit, and '
                'VAE-encoded before OTBD one-step reverse inference.'),
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

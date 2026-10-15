"""Evaluate Gaussian Shading recovery with the OTBD one-step reverse model."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
from diffusers import DDIMInverseScheduler, StableDiffusionPipeline

from data.options import parse
from gaussian_shading_watermark import GaussianShadingWatermark
from models import create_model
from reverse_distill_unipc_eval import (
    convert_reverse_model_to_float32,
    decode_latent,
    generate_latent_with_custom_unipc,
    model_closure,
    prompt_embeddings,
    save_image,
    save_noise_image,
)
from sampler import UniPCSampler


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OPTIONS = (
    PROJECT_ROOT / 'options' / 'test' / 'cocoGSSingleNAFNetEpsilonXT.yml')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Test whether OTBD one-step reverse inference recovers a '
            'Gaussian-Shading watermarked terminal latent.'))
    parser.add_argument('--opt', default=str(DEFAULT_OPTIONS))
    parser.add_argument('--sd-model-id', default=None)
    parser.add_argument(
        '--outdir',
        default=str(PROJECT_ROOT / 'outputs' / 'reverse_distill_gaussian_shading'))
    parser.add_argument('--prompt', default='a smiling woman')
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--guidance-scale', type=float, default=5.5)
    parser.add_argument('--unipc-steps', type=int, default=6)
    parser.add_argument('--start-free-u-step', type=int, default=4)
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
        help=(
            'Use the OTBD learned reverse model. Disable to run Diffusers '\
            'DDIM inversion with the Stable Diffusion UNet.'))
    parser.add_argument('--ddim-inverse-steps', type=int, default=50)
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def resolve_checkpoint_paths(opt):
    for key, value in opt['path'].items():
        if value is None or not key.startswith('pretrain_'):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        opt['path'][key] = str(path.resolve())


def build_model(args):
    opt = parse(args.opt, is_train=False)
    resolve_checkpoint_paths(opt)
    opt['dist'] = False
    opt['num_gpu'] = 0 if args.device == 'cpu' else 1
    if args.sd_model_id:
        opt['sd_model_id'] = args.sd_model_id
    checkpoint_keys = (
        ('pretrain_network_g', 'pretrain_network_g_ema')
        if opt['model_type'] == 'ReverseDistillSingleNAFNetModel'
        else (
            'pretrain_network_xstart',
            'pretrain_network_xt',
            'pretrain_xstart_prompt_tokens'))
    for key in checkpoint_keys:
        checkpoint = opt['path'].get(key)
        if checkpoint and not os.path.isfile(checkpoint):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    return create_model(opt), opt


def latent_metrics(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    predicted_flat = predicted.float().flatten(1)
    target_flat = target.float().flatten(1)
    difference = predicted_flat - target_flat
    target_min = target_flat.amin(dim=1, keepdim=True)
    target_range = (
        target_flat.amax(dim=1, keepdim=True) - target_min).clamp_min(1e-12)
    normalized_mean_l2 = (
        difference / target_range).square().mean(dim=1)
    return {
        'mean_l1': difference.abs().mean().item(),
        'mean_l2': difference.square().mean().item(),
        'psnr_db': (
            -10 * torch.log10(normalized_mean_l2.clamp_min(1e-12))
        ).mean().item(),
        'cosine_similarity': torch.nn.functional.cosine_similarity(
            predicted_flat, target_flat, dim=1).mean().item(),
    }


def prediction_diagnostics(
        predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, object]:
    predicted_float = predicted.float()
    target_float = target.float()
    return {
        'raw_sign_agreement': (
            (predicted_float > 0) == (target_float > 0)
        ).float().mean().item(),
        'mean': predicted_float.mean().item(),
        'std': predicted_float.std().item(),
        'channel_mean': predicted_float.mean(dim=(0, 2, 3)).tolist(),
        'channel_std': predicted_float.std(dim=(0, 2, 3)).tolist(),
    }


@torch.no_grad()
def reverse_noise(
        model,
        final_latent,
        prompt_embedding,
        use_custom_model=True,
        pipe=None,
        ddim_inverse_steps=50):
    if not use_custom_model:
        if pipe is None:
            raise ValueError(
                'pipe is required when use_custom_model is False.')
        if ddim_inverse_steps < 1:
            raise ValueError('--ddim-inverse-steps must be positive.')
        pipe.unfuse_lora()
        pipe.unload_lora_weights()
        inverse_scheduler = DDIMInverseScheduler.from_config(
            pipe.scheduler.config)
        inverse_scheduler.set_timesteps(
            ddim_inverse_steps, device=final_latent.device)
        inverted_latent = final_latent.float()
        predicted_xstart = inverted_latent
        predicted_timestep = inverse_scheduler.timesteps[-1]
        for timestep in inverse_scheduler.timesteps:
            noise_prediction = pipe.unet(
                inverted_latent,
                timestep,
                encoder_hidden_states=prompt_embedding.float()).sample
            step_output = inverse_scheduler.step(
                noise_prediction, timestep, inverted_latent)
            inverted_latent = step_output.prev_sample
            predicted_xstart = step_output.pred_original_sample
            predicted_timestep = timestep
        return {
            'predicted_noise': inverted_latent.detach().clone(),
            'predicted_xstart': predicted_xstart.detach().clone(),
            'predicted_xt': final_latent.detach().clone(),
            'predicted_timestep': torch.full(
                (final_latent.shape[0],),
                predicted_timestep.item(),
                device=final_latent.device,
                dtype=final_latent.dtype),
        }

    model.feed_data({
        'final_latents': final_latent.float(),
        # 'prompt': prompt_embedding.float(),
    })
    model.test()
    if model.opt['model_type'] == 'ReverseDistillSingleNAFNetModel':
        return {
            'predicted_noise': model.output.detach().clone(),
            'predicted_xstart': final_latent.detach().clone(),
            'predicted_xt': model.output.detach().clone(),
            'predicted_timestep': torch.full(
                (final_latent.shape[0],),
                999,
                device=final_latent.device,
                dtype=final_latent.dtype),
        }
    return {
        'predicted_noise': model.output.detach().clone(),
        'predicted_xstart': model.pred_xstart.detach().clone(),
        'predicted_xt': model.pred_xt.detach().clone(),
        'predicted_timestep': model.pred_timestep.detach().clone(),
    }


@torch.no_grad()
def vae_roundtrip_latent(pipe, latent):
    image = decode_latent(pipe, latent)
    quantized_image = (image * 255).round().clamp(0, 255) / 255
    vae_input = quantized_image * 2 - 1
    return (
        pipe.vae.encode(vae_input).latent_dist.mode()
        * pipe.vae.config.scaling_factor)


def save_sample_artifacts(
        sample_dir,
        pipe,
        sampler,
        cond_embeds,
        uncond_embeds,
        args,
        watermarked_noise,
        network_input,
        reverse_result,
        image_roundtrip_input,
        image_reverse_result,
        control_noise,
        control_input,
        control_reverse_result,
        control_image_roundtrip_input,
        control_image_reverse_result):
    sample_dir.mkdir(parents=True, exist_ok=True)
    predicted_noise = reverse_result['predicted_noise']
    regenerated_latent = generate_latent_with_custom_unipc(
        sampler,
        predicted_noise,
        cond_embeds,
        uncond_embeds,
        args.start_free_u_step)
    generated_image = decode_latent(pipe, network_input)
    regenerated_image = decode_latent(pipe, regenerated_latent)

    tensors = {
        'watermarked_noise': watermarked_noise,
        'input_latent': network_input,
        'predicted_xstart': reverse_result['predicted_xstart'],
        'predicted_xt': reverse_result['predicted_xt'],
        'predicted_noise': predicted_noise,
        'image_roundtrip_input_latent': image_roundtrip_input,
        'image_roundtrip_predicted_noise': (
            image_reverse_result['predicted_noise']),
        'control_noise': control_noise,
        'control_input_latent': control_input,
        'control_predicted_noise': control_reverse_result['predicted_noise'],
        'control_image_roundtrip_input_latent': control_image_roundtrip_input,
        'control_image_roundtrip_predicted_noise': (
            control_image_reverse_result['predicted_noise']),
    }
    for name, tensor in tensors.items():
        torch.save(tensor.detach().cpu(), sample_dir / f'{name}.pt')
        save_noise_image(tensor, sample_dir / f'{name}_0_255.png')
    save_image(generated_image[0], sample_dir / 'generated_watermarked.png')
    save_image(regenerated_image[0], sample_dir / 'regenerated_from_prediction.png')
    control_image = decode_latent(pipe, control_input)
    save_image(control_image[0], sample_dir / 'generated_control.png')


def mean_metric(samples, metric_path):
    values = []
    for sample in samples:
        value = sample
        for key in metric_path:
            value = value[key]
        values.append(float(value))
    return sum(values) / len(values)


def main():
    args = parse_args()
    if args.num < 1:
        raise ValueError('--num must be positive.')
    if args.ddim_inverse_steps < 1:
        raise ValueError('--ddim-inverse-steps must be positive.')
    if args.height != 512 or args.width != 512:
        raise ValueError(
            'This Gaussian Shading configuration currently supports only 512x512.')

    device = torch.device(args.device)
    dtype = torch.float32
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_custom_model:
        model, opt = build_model(args)
        convert_reverse_model_to_float32(model)
    else:
        opt = parse(args.opt, is_train=False)
        resolve_checkpoint_paths(opt)
        if args.sd_model_id:
            opt['sd_model_id'] = args.sd_model_id
        model = None
    pipe = StableDiffusionPipeline.from_pretrained(
        opt['sd_model_id'], torch_dtype=dtype)
    pipe.to(device=device, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    sampler = UniPCSampler(
        pipe,
        model_closure=model_closure,
        steps=args.unipc_steps,
        guidance_scale=args.guidance_scale,
        is_high_resoulution=False)
    cond_embeds = prompt_embeddings(pipe, [args.prompt], device, dtype)
    uncond_embeds = prompt_embeddings(pipe, [''], device, dtype)

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
        network_input = generate_latent_with_custom_unipc(
            sampler,
            watermarked_noise,
            cond_embeds,
            uncond_embeds,
            args.start_free_u_step)
        reverse_result = reverse_noise(
            model,
            network_input,
            cond_embeds,
            use_custom_model=args.use_custom_model,
            pipe=pipe,
            ddim_inverse_steps=args.ddim_inverse_steps)
        image_roundtrip_input = vae_roundtrip_latent(pipe, network_input)
        image_reverse_result = reverse_noise(
            model,
            image_roundtrip_input,
            cond_embeds,
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
        control_input = generate_latent_with_custom_unipc(
            sampler,
            control_noise,
            cond_embeds,
            uncond_embeds,
            args.start_free_u_step)
        control_reverse_result = reverse_noise(
            model,
            control_input,
            cond_embeds,
            use_custom_model=args.use_custom_model,
            pipe=pipe,
            ddim_inverse_steps=args.ddim_inverse_steps)
        control_image_roundtrip_input = vae_roundtrip_latent(
            pipe, control_input)
        control_image_reverse_result = reverse_noise(
            model,
            control_image_roundtrip_input,
            cond_embeds,
            use_custom_model=args.use_custom_model,
            pipe=pipe,
            ddim_inverse_steps=args.ddim_inverse_steps)

        direct_watermark = watermark.evaluate(watermarked_noise)
        latent_path_watermark = watermark.evaluate(
            reverse_result['predicted_noise'])
        image_roundtrip_watermark = watermark.evaluate(
            image_reverse_result['predicted_noise'])
        direct_control = watermark.evaluate(control_noise)
        latent_path_control = watermark.evaluate(
            control_reverse_result['predicted_noise'])
        image_roundtrip_control = watermark.evaluate(
            control_image_reverse_result['predicted_noise'])
        sample_metrics = {
            'index': index,
            'seed': seed,
            'direct_watermark': direct_watermark.to_dict(),
            'latent_path_watermark': latent_path_watermark.to_dict(),
            'image_roundtrip_watermark': image_roundtrip_watermark.to_dict(),
            'direct_control': direct_control.to_dict(),
            'latent_path_control': latent_path_control.to_dict(),
            'image_roundtrip_control': image_roundtrip_control.to_dict(),
            'latent_path_predicted_vs_watermarked_noise': latent_metrics(
                reverse_result['predicted_noise'], watermarked_noise),
            'image_roundtrip_predicted_vs_watermarked_noise': latent_metrics(
                image_reverse_result['predicted_noise'], watermarked_noise),
            'latent_path_control_predicted_vs_original_noise': latent_metrics(
                control_reverse_result['predicted_noise'], control_noise),
            'image_roundtrip_control_predicted_vs_original_noise': latent_metrics(
                control_image_reverse_result['predicted_noise'], control_noise),
            'latent_path_prediction_diagnostics': prediction_diagnostics(
                reverse_result['predicted_noise'], watermarked_noise),
            'image_roundtrip_prediction_diagnostics': prediction_diagnostics(
                image_reverse_result['predicted_noise'], watermarked_noise),
            'latent_path_predicted_timestep': (
                reverse_result['predicted_timestep'].float().mean().item()),
            'image_roundtrip_predicted_timestep': (
                image_reverse_result['predicted_timestep'].float().mean().item()),
            'latent_path_control_predicted_timestep': (
                control_reverse_result['predicted_timestep'].float().mean().item()),
            'image_roundtrip_control_predicted_timestep': (
                control_image_reverse_result['predicted_timestep']
                .float().mean().item()),
        }
        samples.append(sample_metrics)

        sample_dir = output_dir / f'sample_{index:04d}'
        save_sample_artifacts(
            sample_dir,
            pipe,
            sampler,
            cond_embeds,
            uncond_embeds,
            args,
            watermarked_noise,
            network_input,
            reverse_result,
            image_roundtrip_input,
            image_reverse_result,
            control_noise,
            control_input,
            control_reverse_result,
            control_image_roundtrip_input,
            control_image_reverse_result)
        with open(
                sample_dir / 'metrics.json', 'w', encoding='utf-8') as file:
            json.dump(sample_metrics, file, indent=2)

    aggregate = {
        'sample_count': args.num,
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
        'mean_latent_path_l1': mean_metric(
            samples, ('latent_path_predicted_vs_watermarked_noise', 'mean_l1')),
        'mean_image_roundtrip_l1': mean_metric(
            samples,
            ('image_roundtrip_predicted_vs_watermarked_noise', 'mean_l1')),
        'mean_latent_path_cosine_similarity': mean_metric(
            samples,
            ('latent_path_predicted_vs_watermarked_noise', 'cosine_similarity')),
        'mean_image_roundtrip_cosine_similarity': mean_metric(
            samples,
            ('image_roundtrip_predicted_vs_watermarked_noise',
             'cosine_similarity')),
    }
    report = {
        'configuration': {
            'prompt': args.prompt,
            'base_seed': args.seed,
            'height': args.height,
            'width': args.width,
            'guidance_scale': args.guidance_scale,
            'unipc_steps': args.unipc_steps,
            'start_free_u_step': args.start_free_u_step,
            'use_custom_model': args.use_custom_model,
            'ddim_inverse_steps': args.ddim_inverse_steps,
            'channel_copy': args.channel_copy,
            'hw_copy': args.hw_copy,
            'fpr': args.fpr,
            'user_number': args.user_number,
            'detection_threshold': watermark.detection_threshold,
            'traceability_threshold': watermark.traceability_threshold,
            'detector_prompt_mode': 'known generation prompt',
            'latent_path_description': (
                'Reverse model receives the sampler output latent directly.'),
            'image_roundtrip_description': (
                'Sampler output is VAE-decoded, quantized to 8-bit, normalized, '
                'and VAE-encoded before reverse inference.'),
            'control_note': (
                'Control detection rates are sanity checks, not empirical '
                'estimates of the configured false-positive rate.'),
            'sd_model_id': opt['sd_model_id'],
            'options': str(Path(args.opt).resolve()),
            # 'xstart_checkpoint': opt['path']['pretrain_network_xstart'],
            # 'xt_checkpoint': opt['path']['pretrain_network_xt'],
            # 'xstart_prompt_checkpoint': opt['path'].get(
            #     'pretrain_xstart_prompt_tokens'),
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

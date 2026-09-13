"""Benchmark NAFNet extractor inference on cached clean COCO images."""

import argparse
import json
import statistics
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from coco2017_diffusers_unipc_gaussian_shading_eval import (
    DEFAULT_CAPTIONS,
    generation_cache_path,
    load_coco_captions,
    load_or_generate_sample,
)
from reverse_distill_gaussian_shading_eval import build_model
from reverse_distill_unipc_eval import convert_reverse_model_to_float32


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = 'sd-legacy/stable-diffusion-v1-5'
DEFAULT_OPTIONS = {
    'single_direct_xt': PROJECT_ROOT / 'options' / 'test' / 'cocoGSSingleNAFNetDirectXT.yml',
    'single_epsilon_xt': PROJECT_ROOT / 'options' / 'test' / 'cocoGSSingleNAFNetEpsilonXT.yml',
    'two_branch': PROJECT_ROOT / 'options' / 'test' / 'cocoGSReverseDistillUNetNoneADV.yml',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Time extractor-only inference on cached clean SD v1.5 COCO '
            'images, generating missing cache entries when needed, with the '
            'two-branch model split across two GPUs.'))
    parser.add_argument('--single-direct-opt', default=str(DEFAULT_OPTIONS['single_direct_xt']))
    parser.add_argument('--single-epsilon-opt', default=str(DEFAULT_OPTIONS['single_epsilon_xt']))
    parser.add_argument('--two-branch-opt', default=str(DEFAULT_OPTIONS['two_branch']))
    parser.add_argument('--sd-model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--captions', default=str(DEFAULT_CAPTIONS))
    parser.add_argument(
        '--generation-cache-dir',
        default=str(PROJECT_ROOT / 'outputs' / 'coco_generation_cache'))
    parser.add_argument(
        '--refresh-generation-cache', action='store_true',
        help='Regenerate and overwrite all selected cached COCO samples.')
    parser.add_argument(
        '--outdir',
        default=str(PROJECT_ROOT / 'outputs' / 'nafnet_extraction_latency_1000'))
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--warmup-batches', type=int, default=3)
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
    parser.add_argument(
        '--local-files-only', action='store_true',
        help='Load Stable Diffusion only from the local Hugging Face cache.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--xt-device', default='cuda:1',
        help='Second CUDA device used by the xT branch of the two-branch model.')
    return parser.parse_args()


def resolve_cache_paths(args):
    captions = load_coco_captions(
        Path(args.captions), args.num, args.caption_seed)
    cache_dir = Path(args.generation_cache_dir)
    entries = []
    for index, caption in enumerate(captions):
        seed = args.seed + index
        path = generation_cache_path(
            cache_dir, caption, seed, args, args.sd_model_id)
        entries.append((caption, seed, path))
    return entries


@torch.inference_mode()
def prepare_latent_batches(
        cache_entries, pipe, args, inference_batch_size, device, dtype):
    """Encode one cached image at a time, then form CPU inference batches."""
    prepared_latents = []
    watermarks = []
    for caption, seed, cache_path in tqdm(
            cache_entries,
            desc='Preparing cached clean images', unit='image'):
        (watermark, _, _, generated_image,
         _, _, _, _) = load_or_generate_sample(
            cache_path,
            caption['caption'],
            seed,
            args,
            pipe,
            device,
            dtype,
            require_cache=False,
        )
        quantized = (generated_image * 255).round().clamp(0, 255) / 255
        latent = pipe.vae.encode(quantized * 2 - 1).latent_dist.mode()
        latent = latent * pipe.vae.config.scaling_factor
        prepared_latents.append(latent.cpu())
        watermarks.append(watermark)
    latent_batches = [
        torch.cat(prepared_latents[start:start + inference_batch_size])
        for start in range(0, len(prepared_latents), inference_batch_size)
    ]
    watermark_batches = [
        watermarks[start:start + inference_batch_size]
        for start in range(0, len(watermarks), inference_batch_size)
    ]
    return latent_batches, watermark_batches


def run_extractor(model, latent_batch):
    model.feed_data({'final_latents': latent_batch}, use_prompt_emb=False)
    model.test()
    return model.output


def configure_two_gpu_model(model, primary_device, xt_device):
    if not model.xstart_is_nafnet or not model.xt_is_nafnet:
        raise ValueError('Two-GPU inference requires two NAFNet branches.')
    model.net_xstart.to(primary_device)
    model.net_xt.to(xt_device)
    model._benchmark_xstart_stream = torch.cuda.Stream(device=primary_device)
    model._benchmark_xt_stream = torch.cuda.Stream(device=xt_device)


def configure_two_gpu_single_model(model, primary_device, secondary_device):
    primary_network = model.get_bare_model(model.net_g).to(primary_device).eval()
    model.net_g = primary_network
    model._benchmark_secondary_net_g = deepcopy(primary_network).to(
        secondary_device).eval()
    if model.prediction_type == 'epsilon':
        model._benchmark_secondary_alpha_t = model.alpha_T.to(secondary_device)
        model._benchmark_secondary_sigma_t = model.sigma_T.to(secondary_device)
    model._benchmark_primary_stream = torch.cuda.Stream(device=primary_device)
    model._benchmark_secondary_stream = torch.cuda.Stream(
        device=secondary_device)


@torch.inference_mode()
def run_two_gpu_single_extractor(
        model, primary_input, secondary_input, primary_device):
    primary_stream = model._benchmark_primary_stream
    secondary_stream = model._benchmark_secondary_stream
    primary_stream.wait_stream(torch.cuda.current_stream(primary_device))
    secondary_stream.wait_stream(
        torch.cuda.current_stream(secondary_input.device))

    with torch.cuda.stream(primary_stream):
        primary_output = model.net_g(primary_input)
        if model.prediction_type == 'epsilon':
            primary_output = (
                model.alpha_T.to(dtype=primary_output.dtype) * primary_input
                + model.sigma_T.to(dtype=primary_output.dtype) * primary_output)
    with torch.cuda.stream(secondary_stream):
        secondary_output = model._benchmark_secondary_net_g(secondary_input)
        if model.prediction_type == 'epsilon':
            secondary_output = (
                model._benchmark_secondary_alpha_t.to(
                    dtype=secondary_output.dtype) * secondary_input
                + model._benchmark_secondary_sigma_t.to(
                    dtype=secondary_output.dtype) * secondary_output)

    torch.cuda.current_stream(primary_device).wait_stream(primary_stream)
    torch.cuda.synchronize(secondary_input.device)
    secondary_output = secondary_output.to(primary_device, non_blocking=True)
    network_output = torch.cat([primary_output, secondary_output], dim=0)
    model.output = network_output
    return model.output


@torch.inference_mode()
def run_two_gpu_extractor(model, primary_input, xt_input, primary_device):
    xstart_stream = model._benchmark_xstart_stream
    xt_stream = model._benchmark_xt_stream
    xstart_stream.wait_stream(torch.cuda.current_stream(primary_device))
    xt_stream.wait_stream(torch.cuda.current_stream(xt_input.device))

    with torch.cuda.stream(xstart_stream):
        pred_xstart = model._run_network(
            model.net_xstart, primary_input, None, model.xstart_is_nafnet)
    with torch.cuda.stream(xt_stream):
        xt_output = model._run_network(
            model.net_xt, xt_input, None, model.xt_is_nafnet)

    torch.cuda.current_stream(primary_device).wait_stream(xstart_stream)
    torch.cuda.synchronize(xt_input.device)
    xt_output = tuple(
        output.to(primary_device, non_blocking=True)
        if torch.is_tensor(output) else output
        for output in xt_output)
    pred_xt, _, _, pred_timestep = xt_output
    alpha_cumprod = model._alpha_from_timestep(
        pred_timestep.clamp(0, 999)).view(-1, 1, 1, 1)
    return (
        pred_xt - torch.sqrt(alpha_cumprod) * pred_xstart.detach()
    ) / torch.sqrt((1.0 - alpha_cumprod).clamp_min(1e-6))


def synchronize_devices(devices):
    for device in devices:
        torch.cuda.synchronize(device)


def benchmark_model(
        model, latent_batches, watermark_batches, device, warmup_batches,
        secondary_device, parallel_mode):
    warmup_batch = latent_batches[0]
    if parallel_mode == 'data':
        warmup_primary, warmup_secondary = torch.tensor_split(
            warmup_batch, 2, dim=0)
    else:
        warmup_primary = warmup_secondary = warmup_batch
    warmup_primary = warmup_primary.to(device=device, non_blocking=True)
    warmup_secondary = warmup_secondary.to(
        device=secondary_device, non_blocking=True)
    for _ in range(warmup_batches):
        if parallel_mode == 'data':
            run_two_gpu_single_extractor(
                model, warmup_primary, warmup_secondary, device)
        else:
            run_two_gpu_extractor(
                model, warmup_primary, warmup_secondary, device)
    devices = (device, secondary_device)
    synchronize_devices(devices)

    batch_times = []
    sample_count = 0
    detected_count = 0
    bit_accuracies = []
    for cpu_latent_batch, batch_watermarks in zip(
            latent_batches, watermark_batches):
        if parallel_mode == 'data':
            primary_batch, secondary_batch = torch.tensor_split(
                cpu_latent_batch, 2, dim=0)
        else:
            primary_batch = secondary_batch = cpu_latent_batch
        primary_batch = primary_batch.to(device=device, non_blocking=True)
        secondary_batch = secondary_batch.to(
            device=secondary_device, non_blocking=True)
        synchronize_devices(devices)
        start_time = time.perf_counter()
        if parallel_mode == 'data':
            output = run_two_gpu_single_extractor(
                model, primary_batch, secondary_batch, device)
        else:
            output = run_two_gpu_extractor(
                model, primary_batch, secondary_batch, device)
        synchronize_devices(devices)
        batch_times.append((time.perf_counter() - start_time) * 1000)
        sample_count += cpu_latent_batch.shape[0]
        for sample_output, watermark in zip(output, batch_watermarks):
            evaluation = watermark.evaluate(sample_output.unsqueeze(0))
            detected_count += int(evaluation.detected)
            bit_accuracies.append(evaluation.bit_accuracy)

    total_ms = sum(batch_times)
    return {
        'samples': sample_count,
        'batches': len(batch_times),
        'batch_size': latent_batches[0].shape[0],
        'parallel_mode': parallel_mode,
        'total_inference_ms': total_ms,
        'mean_batch_ms': statistics.fmean(batch_times),
        'median_batch_ms': statistics.median(batch_times),
        'mean_sample_ms': total_ms / sample_count,
        'throughput_images_per_second': sample_count / (total_ms / 1000),
        'detection_rate': detected_count / sample_count,
        'mean_bit_accuracy': statistics.fmean(bit_accuracies),
        'per_batch_ms': batch_times,
    }


def main():
    args = parse_args()
    if args.device == 'cpu':
        raise ValueError('CUDA is required for synchronized inference timing.')
    device = torch.device(args.device)
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    xt_device = torch.device(args.xt_device)
    if xt_device.type != 'cuda' or xt_device.index is None:
        raise ValueError('--xt-device must identify a CUDA device, such as cuda:1.')
    if xt_device.index >= torch.cuda.device_count():
        raise ValueError(
            f'--xt-device {xt_device} is unavailable; found '
            f'{torch.cuda.device_count()} CUDA device(s).')
    if xt_device == device:
        raise ValueError('--xt-device must differ from the primary --device.')
    if args.num % args.batch_size:
        raise ValueError('--num must be divisible by --batch-size.')
    if args.batch_size < 2:
        raise ValueError('--batch-size must be at least 2 for two-GPU data parallelism.')
    if args.warmup_batches < 0:
        raise ValueError('--warmup-batches cannot be negative.')
    if not 12 <= args.steps <= 15:
        raise ValueError('--steps must be between 12 and 15 to match the cache key.')
    if args.height != 512 or args.width != 512:
        raise ValueError('The cached Gaussian-Shading images must be 512x512.')

    dtype = torch.float32
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_entries = resolve_cache_paths(args)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_id,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    latent_batches, watermark_batches = prepare_latent_batches(
        cache_entries, pipe, args, args.batch_size, device, dtype)
    del pipe
    torch.cuda.empty_cache()

    option_paths = {
        'single_direct_xt': Path(args.single_direct_opt),
        'single_epsilon_xt': Path(args.single_epsilon_opt),
        'two_branch_parallel': Path(args.two_branch_opt),
    }
    results = {}
    for name, option_path in option_paths.items():
        model_args = SimpleNamespace(
            opt=str(option_path), device=args.device, sd_model_id=args.sd_model_id)
        model, opt = build_model(model_args)
        if opt['sd_model_id'] != args.sd_model_id:
            raise ValueError(
                f'{name} uses {opt["sd_model_id"]!r}, expected {args.sd_model_id!r}.')
        convert_reverse_model_to_float32(model)
        if name == 'two_branch_parallel':
            configure_two_gpu_model(model, device, xt_device)
            parallel_mode = 'branch_parallel'
        else:
            configure_two_gpu_single_model(model, device, xt_device)
            parallel_mode = 'data_parallel'
        results[name] = benchmark_model(
            model, latent_batches, watermark_batches, device,
            args.warmup_batches,
            secondary_device=xt_device,
            parallel_mode=('data' if parallel_mode == 'data_parallel'
                           else 'branch'))
        results[name]['allocation'] = parallel_mode
        results[name]['option'] = str(option_path.resolve())
        results[name]['model_type'] = opt['model_type']
        print(
            f'{name}: {results[name]["total_inference_ms"]:.3f} ms total, '
            f'{results[name]["mean_sample_ms"]:.3f} ms/image, '
            f'{results[name]["throughput_images_per_second"]:.2f} images/s, '
            f'detection rate {results[name]["detection_rate"]:.2%}')
        del model
        for cache_device in (device, xt_device):
            with torch.cuda.device(cache_device):
                torch.cuda.empty_cache()

    report = {
        'configuration': {
            'sd_model_id': args.sd_model_id,
            'generation_cache_dir': str(Path(args.generation_cache_dir).resolve()),
            'samples': args.num,
            'cache_load_batch_size': 1,
            'batch_size': args.batch_size,
            'timed_batches': args.num // args.batch_size,
            'warmup_batches': args.warmup_batches,
            'device': torch.cuda.get_device_name(device),
            'secondary_device': torch.cuda.get_device_name(xt_device),
            'two_branch_devices': {
                'xstart': str(device),
                'xt': str(xt_device),
            },
            'allocation_strategy': {
                'single_direct_xt': 'data parallel across both GPUs',
                'single_epsilon_xt': 'data parallel across both GPUs',
                'two_branch_parallel': 'x-start and xT branch model parallelism',
            },
            'input_scope': (
                f'Exactly {args.num} clean images are loaded from cache or '
                'generated when missing, then VAE-encoded one at a time before '
                'timing. The resulting CPU '
                f'latents are grouped into batches of {args.batch_size} only for '
                'extractor inference. Cache loading, initial host-to-device '
                'transfer, image perturbation, VAE encoding, and Gaussian-Shading '
                'decoding are excluded. Detection rate and mean bit accuracy are '
                'computed after each timed inference interval. Each single NAFNet '
                'splits every batch '
                f'across {device} and {xt_device}. For the two-branch model, '
                f'x-start runs on {device} and xT runs on {xt_device}; all timed '
                'regions include cross-device output gathering and reconstruction.'),
        },
        'results': results,
    }
    report_path = output_dir / 'latency.json'
    with open(report_path, 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    print(f'Saved latency report to {report_path}')


if __name__ == '__main__':
    main()

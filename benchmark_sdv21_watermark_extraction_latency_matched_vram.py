"""Benchmark FARI, OSI, and Ours latency at one shared VRAM budget.

Batch size is inversely proportional to the calflops parameter count, so
parameters x batch_size matches across methods. The timed input is an fp32
latent of shape (batch, 4, 64, 64). FARI and OSI are timed data-parallel.
Ours is timed branch-parallel. Image encoding and Gaussian-Shading decoding
stay outside the timed region.

At the default reference of one latent per GPU, the matched batches are
FARI 2, OSI 2, and Ours 112. OSI lands on the same integer as FARI because
865.911 M is 99.7 percent of 868.469 M.
"""

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

from benchmark_nafnet_watermark_extraction_latency import (
    configure_two_gpu_model,
    configure_two_gpu_single_model,
    run_two_gpu_extractor,
    run_two_gpu_single_extractor,
    synchronize_devices,
)
from benchmark_sdv21_watermark_extraction_latency import (
    EXTRACTOR_NAMES,
    FARILatentToNoise,
    OSILatentToNoise,
    SingleExtractorHandle,
    batch_items,
    batch_tensors,
    empty_cuda_cache,
    encode_osi_latents,
    prepare_cached_inputs,
    require_checkpoint,
    resolve_cache_paths,
)
from coco2017_diffusers_unipc_gaussian_shading_adv_eval_sdv21 import (
    DEFAULT_MODEL_ID,
    DEFAULT_OPTIONS,
    check_ours_backbone,
    configure_fari,
    load_osi_model,
)
from coco2017_diffusers_unipc_gaussian_shading_eval import DEFAULT_CAPTIONS
from fari_inversion import FARIMode, fari_mode
from reverse_distill_gaussian_shading_eval import build_model
from reverse_distill_unipc_eval import convert_reverse_model_to_float32


PROJECT_ROOT = Path(__file__).resolve().parent
# calflops parameter counts for one latent-to-noise extraction, in thousandths
# of a million. 868_469 means 868.469 M parameters.
EXTRACTOR_PARAMS_MILLI_M = {
    'fari': 868_469,
    'osi': 865_911,
    'ours': 15_468,
}
LATENT_CHANNELS = 4
LATENT_HEIGHT = 64
LATENT_WIDTH = 64
BYTES_PER_PARAMETER = 4  # fp32
MATCH_TOLERANCE = 0.01


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Time FARI, OSI, and Ours extraction on cached clean SD 2.1 COCO '
            'images. Each method uses the batch size that matches '
            'parameters x batch size, the shared fp32 VRAM budget.'))
    parser.add_argument('--opt', default=str(DEFAULT_OPTIONS))
    parser.add_argument('--sd-model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--fari-checkpoint', default=str(PROJECT_ROOT / 'fari_weights.pth'))
    parser.add_argument('--fari-lora-rank', type=int, default=8)
    parser.add_argument(
        '--osi-unet-checkpoint', default=str(PROJECT_ROOT / 'osi_sd21_unet.pth'))
    parser.add_argument(
        '--osi-encoder-checkpoint', default=str(PROJECT_ROOT / 'osi_sd21_encoder.pth'))
    parser.add_argument(
        '--skip-extractors', nargs='*', choices=EXTRACTOR_NAMES, default=(),
        help='Extractors to omit, for example: --skip-extractors osi.')
    parser.add_argument('--captions', default=str(DEFAULT_CAPTIONS))
    parser.add_argument(
        '--generation-cache-dir',
        default=str(PROJECT_ROOT / 'outputs' / 'coco_generation_cache_sdv21'))
    parser.add_argument(
        '--refresh-generation-cache', action='store_true',
        help='Regenerate and overwrite all selected cached COCO samples.')
    parser.add_argument(
        '--outdir',
        default=str(PROJECT_ROOT / 'outputs' / 'sdv21_extraction_latency_matched_vram'))
    parser.add_argument(
        '--num', type=int, default=224,
        help=(
            'Number of clean images. Planned batches are reduced to a divisor '
            'of this count so none is larger than the dataset. 112 divides '
            'the default batches 2, 2, and 112.'))
    parser.add_argument(
        '--reference-batch-size', type=int, default=2,
        help=(
            'FARI batch size that sets the shared VRAM budget. OSI and Ours '
            'scale by the inverse parameter ratio. Must be an even integer '
            '>= 2. The default gives FARI 2, OSI 2, and Ours 112.'))
    parser.add_argument(
        '--fari-batch-size', type=int, default=None,
        help='Override the parameter-matched FARI batch size.')
    parser.add_argument(
        '--osi-batch-size', type=int, default=None,
        help='Override the parameter-matched OSI batch size.')
    parser.add_argument(
        '--ours-batch-size', type=int, default=None,
        help='Override the parameter-matched Ours batch size.')
    parser.add_argument('--warmup-batches', type=int, default=3)
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--guidance-scale', type=float, default=5.5)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--seed', type=int, default=25)
    parser.add_argument('--caption-seed', type=int, default=8)
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
        help=(
            'Second CUDA device. FARI and OSI place the second data-parallel '
            'replica here. Ours places the xT branch here.'))
    return parser.parse_args()


def rounded_div(numerator, denominator):
    """Round a positive integer division, with halves going to the even quotient."""
    quotient, remainder = divmod(numerator, denominator)
    twice_remainder = remainder * 2
    if twice_remainder > denominator or (twice_remainder == denominator and quotient % 2):
        return quotient + 1
    return quotient


def nearest_even_batch(size, params_milli_m, target_product):
    """Pick an even batch of at least 2 whose parameter product is closest."""
    if size >= 2 and size % 2 == 0:
        return size
    lower = size - 1 if size % 2 else size
    upper = lower + 2
    if lower < 2:
        return upper
    lower_gap = abs(lower * params_milli_m - target_product)
    upper_gap = abs(upper * params_milli_m - target_product)
    return lower if lower_gap <= upper_gap else upper


def iso_vram_batch_sizes(reference_batch_size):
    """Scale batch size by the inverse parameter count.

    FARI and OSI each keep a full replica on both GPUs and split the batch.
    Ours keeps one branch on each GPU and forwards the full batch to both.
    Those two factors cancel, so matching ``parameters x batch_size`` also
    matches the per-GPU parameter-batch product when the Ours branches are
    treated as an even split.
    """
    if reference_batch_size < 2 or reference_batch_size % 2:
        raise ValueError('--reference-batch-size must be an even integer >= 2.')
    reference_params = EXTRACTOR_PARAMS_MILLI_M['fari']
    target_product = reference_batch_size * reference_params
    sizes = {}
    for name, params_milli_m in EXTRACTOR_PARAMS_MILLI_M.items():
        size = rounded_div(target_product, params_milli_m)
        if name != 'ours':
            size = nearest_even_batch(size, params_milli_m, target_product)
        elif size < 1:
            size = 1
        product = size * params_milli_m
        gap = abs(product - target_product) / target_product
        if gap > MATCH_TOLERANCE:
            raise RuntimeError(
                f'{name} batch {size} misses the FARI VRAM budget by {gap:.2%}. '
                'Choose another --reference-batch-size.')
        sizes[name] = size
    return sizes


def positive_divisors(num_samples):
    divisors = set()
    factor = 1
    while factor * factor <= num_samples:
        if num_samples % factor == 0:
            divisors.add(factor)
            divisors.add(num_samples // factor)
        factor += 1
    return divisors


def batch_size_within_dataset(matched_size, num_samples, even):
    """Pick the dataset divisor closest to the parameter-matched batch.

    Every candidate is at most ``num_samples``, so a batch never asks for more
    images than exist. FARI and OSI stay on an even divisor of at least 2.
    """
    if num_samples < 1:
        raise ValueError('--num must be positive.')
    candidates = positive_divisors(num_samples)
    if even:
        candidates = {size for size in candidates if size >= 2 and size % 2 == 0}
    if not candidates:
        raise ValueError(
            f'--num {num_samples} has no even divisor of at least 2. '
            'FARI and OSI need an even batch so it splits across the two GPUs.')
    return min(candidates, key=lambda size: (abs(size - matched_size), -size))


def build_batch_plan(reference_batch_size, overrides, num_samples):
    derived = iso_vram_batch_sizes(reference_batch_size)
    target_product = reference_batch_size * EXTRACTOR_PARAMS_MILLI_M['fari']
    plan = {}
    for name in EXTRACTOR_NAMES:
        params_milli_m = EXTRACTOR_PARAMS_MILLI_M[name]
        fitted = batch_size_within_dataset(
            derived[name], num_samples, even=(name != 'ours'))
        batch_size = fitted if overrides[name] is None else overrides[name]
        product = params_milli_m * batch_size
        plan[name] = {
            'params_m': params_milli_m / 1000,
            'parameter_count': params_milli_m * 1000,
            'derived_batch_size': derived[name],
            'batch_size': batch_size,
            'parameter_batch_product_m': product / 1000,
            'parameter_batch_bytes': params_milli_m * 1000 * batch_size * BYTES_PER_PARAMETER,
            'relative_gap': abs(product - target_product) / target_product,
        }
    return plan


def latent_shape(batch_size):
    return (batch_size, LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH)


def regroup_latents(single_latents, batch_size):
    expected = latent_shape(1)
    for latent in single_latents:
        if tuple(latent.shape) != expected:
            raise ValueError(
                f'Each cached latent must have shape {expected}, got {tuple(latent.shape)}.')
    batches = batch_tensors(single_latents, batch_size)
    expected_batch = latent_shape(batch_size)
    for batch in batches:
        if tuple(batch.shape) != expected_batch:
            raise ValueError(
                f'Expected latent batch shape {expected_batch}, got {tuple(batch.shape)}.')
    return batches


def release_pipeline_modules(pipe, names, devices):
    """Move unused pipeline modules off CUDA and drop them before timing."""
    released = []
    for name in names:
        module = getattr(pipe, name, None)
        setattr(pipe, name, None)
        if module is None:
            continue
        module.to('cpu')
        released.append(module)
    del released
    gc.collect()
    empty_cuda_cache(devices)


def reset_cuda_peak_memory(devices):
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_memory(devices):
    """Peak resident memory after the weights are loaded, including the forward."""
    snapshot = {}
    for device in devices:
        snapshot[str(device)] = {
            'max_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'max_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'max_allocated_mib': torch.cuda.max_memory_allocated(device) / (1024 ** 2),
            'max_reserved_mib': torch.cuda.max_memory_reserved(device) / (1024 ** 2),
        }
    return snapshot


def cuda_current_memory(devices):
    """Tensor memory and caching-allocator reservation currently held on each GPU."""
    snapshot = {}
    for device in devices:
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        snapshot[str(device)] = {
            'allocated_bytes': allocated,
            'reserved_bytes': reserved,
        }
    return snapshot


def summarize_memory_samples(samples):
    """Average the per-batch snapshots. Total is the sum across both GPUs."""
    devices = list(samples[0])
    summary = {'batches': len(samples)}
    total_allocated = []
    total_reserved = []
    for device in devices:
        allocated = [sample[device]['allocated_bytes'] for sample in samples]
        reserved = [sample[device]['reserved_bytes'] for sample in samples]
        summary[device] = {
            'mean_allocated_bytes': statistics.fmean(allocated),
            'mean_reserved_bytes': statistics.fmean(reserved),
            'mean_allocated_mib': statistics.fmean(allocated) / (1024 ** 2),
            'mean_reserved_mib': statistics.fmean(reserved) / (1024 ** 2),
        }
    for sample in samples:
        total_allocated.append(sum(sample[device]['allocated_bytes'] for device in devices))
        total_reserved.append(sum(sample[device]['reserved_bytes'] for device in devices))
    summary['total'] = {
        'mean_allocated_bytes': statistics.fmean(total_allocated),
        'mean_reserved_bytes': statistics.fmean(total_reserved),
        'mean_allocated_mib': statistics.fmean(total_allocated) / (1024 ** 2),
        'mean_reserved_mib': statistics.fmean(total_reserved) / (1024 ** 2),
    }
    summary['per_batch_allocated_mib'] = {
        device: [sample[device]['allocated_bytes'] / (1024 ** 2) for sample in samples]
        for device in devices
    }
    summary['per_batch_allocated_mib']['total'] = [
        value / (1024 ** 2) for value in total_allocated]
    return summary


def time_latent_batches(
        model, latent_batches, watermark_batches, device, secondary_device,
        warmup_batches, parallel_mode):
    """Time the same forwards as benchmark_model and sample VRAM after each batch."""
    warmup_batch = latent_batches[0]
    if parallel_mode == 'data':
        warmup_primary, warmup_secondary = torch.tensor_split(warmup_batch, 2, dim=0)
    else:
        warmup_primary = warmup_secondary = warmup_batch
    warmup_primary = warmup_primary.to(device=device, non_blocking=True)
    warmup_secondary = warmup_secondary.to(device=secondary_device, non_blocking=True)
    for _ in range(warmup_batches):
        if parallel_mode == 'data':
            run_two_gpu_single_extractor(
                model, warmup_primary, warmup_secondary, device)
        else:
            run_two_gpu_extractor(model, warmup_primary, warmup_secondary, device)
    devices = (device, secondary_device)
    synchronize_devices(devices)

    batch_times = []
    memory_samples = []
    sample_count = 0
    detected_count = 0
    bit_accuracies = []
    for cpu_latent_batch, batch_watermarks in zip(latent_batches, watermark_batches):
        if parallel_mode == 'data':
            primary_batch, secondary_batch = torch.tensor_split(
                cpu_latent_batch, 2, dim=0)
        else:
            primary_batch = secondary_batch = cpu_latent_batch
        primary_batch = primary_batch.to(device=device, non_blocking=True)
        secondary_batch = secondary_batch.to(device=secondary_device, non_blocking=True)
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
        memory_samples.append(cuda_current_memory(devices))
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
        'peak_memory': cuda_peak_memory(devices),
        'mean_memory': summarize_memory_samples(memory_samples),
    }


def time_extractor(
        network, latent_batches, watermark_batches, device, secondary_device,
        warmup_batches, parallel_mode):
    if parallel_mode == 'data':
        model = SingleExtractorHandle(network)
        configure_two_gpu_single_model(model, device, secondary_device)
    else:
        model = network
        configure_two_gpu_model(model, device, secondary_device)
    devices = (device, secondary_device)
    empty_cuda_cache(devices)
    reset_cuda_peak_memory(devices)
    result = time_latent_batches(
        model, latent_batches, watermark_batches, device, secondary_device,
        warmup_batches, parallel_mode)
    result['allocation'] = (
        'data_parallel' if parallel_mode == 'data' else 'branch_parallel')
    return result


def annotate_result(result, plan_entry):
    result['params_m'] = plan_entry['params_m']
    result['parameter_count'] = plan_entry['parameter_count']
    result['parameter_batch_product_m'] = plan_entry['parameter_batch_product_m']
    result['parameter_batch_bytes'] = plan_entry['parameter_batch_bytes']
    result['latent_shape'] = list(latent_shape(plan_entry['batch_size']))
    if result['batch_size'] != plan_entry['batch_size']:
        raise RuntimeError(
            f'Timed batch {result["batch_size"]} does not match the VRAM plan '
            f'batch {plan_entry["batch_size"]}.')
    return result


def validate_args(args, device, secondary_device, plan, enabled):
    if args.device == 'cpu':
        raise ValueError('CUDA is required for synchronized inference timing.')
    if secondary_device.type != 'cuda' or secondary_device.index is None:
        raise ValueError('--xt-device must identify a CUDA device, such as cuda:1.')
    if secondary_device.index >= torch.cuda.device_count():
        raise ValueError(
            f'--xt-device {secondary_device} is unavailable; found '
            f'{torch.cuda.device_count()} CUDA device(s).')
    if secondary_device == device:
        raise ValueError('--xt-device must differ from the primary --device.')
    if args.warmup_batches < 0:
        raise ValueError('--warmup-batches cannot be negative.')
    if not 12 <= args.steps <= 15:
        raise ValueError('--steps must be between 12 and 15 to match the cache key.')
    if args.height != 512 or args.width != 512:
        raise ValueError('The cached Gaussian-Shading images must be 512x512.')
    if not enabled:
        raise ValueError('At least one extractor must remain enabled.')
    enabled_batches = []
    for name in enabled:
        batch_size = plan[name]['batch_size']
        if batch_size > args.num:
            raise ValueError(
                f'{name} batch {batch_size} exceeds --num {args.num}. '
                'A batch cannot contain more images than the dataset.')
        if name == 'ours':
            if batch_size < 1:
                raise ValueError('--ours-batch-size must be positive.')
        elif batch_size < 2 or batch_size % 2:
            raise ValueError(
                f'--{name}-batch-size must be an even integer >= 2 so the batch '
                'splits evenly across the two GPUs.')
        enabled_batches.append(batch_size)
    if any(args.num % batch_size for batch_size in enabled_batches):
        suggested = math.lcm(*enabled_batches)
        raise ValueError(
            '--num must be divisible by each enabled batch size '
            f'{ {name: plan[name]["batch_size"] for name in enabled} }. '
            f'The least such positive count is {suggested}.')
    if 'fari' in enabled:
        if args.fari_lora_rank < 1:
            raise ValueError('--fari-lora-rank must be positive.')
        require_checkpoint(args.fari_checkpoint, 'FARI')
    if 'osi' in enabled:
        require_checkpoint(args.osi_encoder_checkpoint, 'OSI encoder')
        require_checkpoint(args.osi_unet_checkpoint, 'OSI U-Net')
    if 'ours' in enabled:
        check_ours_backbone(args)
    return enabled


def format_gib(num_bytes):
    return num_bytes / (1024 ** 3)


def print_batch_plan(plan, num_samples):
    print(
        f'Matched VRAM batches for {num_samples} fp32 latents '
        '(N, 4, 64, 64):')
    for name in EXTRACTOR_NAMES:
        entry = plan[name]
        limited = ''
        if entry['batch_size'] != entry['derived_batch_size']:
            limited = (
                f', parameter match {entry["derived_batch_size"]} '
                f'limited by {num_samples} images')
        print(
            f'  {name}: batch {entry["batch_size"]}{limited}, '
            f'{entry["params_m"]:.3f} M params, '
            f'budget {format_gib(entry["parameter_batch_bytes"]):.3f} GiB, '
            f'gap {entry["relative_gap"]:.2%}')


def format_memory_mib(snapshot, field):
    return ', '.join(
        f'{device} {stats[field]:.0f} MiB'
        for device, stats in snapshot.items()
        if device != 'total' and isinstance(stats, dict) and field in stats)


def print_result(name, result):
    peak_text = format_memory_mib(result.get('peak_memory', {}), 'max_allocated_mib')
    mean_memory = result.get('mean_memory', {})
    mean_text = format_memory_mib(mean_memory, 'mean_allocated_mib')
    total_mean = mean_memory.get('total', {}).get('mean_allocated_mib')
    total_text = f', total {total_mean:.0f} MiB' if total_mean is not None else ''
    print(
        f'{name}: batch {result["batch_size"]}, '
        f'{result["total_inference_ms"]:.3f} ms total, '
        f'{result["mean_sample_ms"]:.3f} ms/image, '
        f'{result["throughput_images_per_second"]:.2f} images/s, '
        f'detection rate {result["detection_rate"]:.2%}, '
        f'peak allocated {peak_text}, '
        f'mean allocated {mean_text}{total_text}')


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.index is None and device.type == 'cuda':
        device = torch.device('cuda', torch.cuda.current_device())
    secondary_device = torch.device(args.xt_device)
    enabled = tuple(name for name in EXTRACTOR_NAMES if name not in args.skip_extractors)
    plan = build_batch_plan(args.reference_batch_size, {
        'fari': args.fari_batch_size,
        'osi': args.osi_batch_size,
        'ours': args.ours_batch_size,
    }, args.num)
    enabled = validate_args(args, device, secondary_device, plan, enabled)
    print_batch_plan(plan, args.num)
    devices = (device, secondary_device)

    dtype = torch.float32
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_entries = resolve_cache_paths(args)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_id,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    alpha_terminal = pipe.scheduler.alphas_cumprod[-1].detach().float().cpu()
    latent_ones, image_ones, watermark_ones = prepare_cached_inputs(
        cache_entries, pipe, args, 1, device, dtype, keep_images='osi' in enabled)
    watermark_list = [batch[0] for batch in watermark_ones]

    osi_model = None
    osi_prompt = None
    osi_latent_ones = None
    if 'osi' in enabled:
        # Copy the clean U-Net before FARI injects adapters into pipe.unet.
        osi_model, osi_prompt = load_osi_model(pipe, args, device, dtype)
        osi_prompt = osi_prompt.detach().cpu()
        osi_latent_ones = encode_osi_latents(osi_model, image_ones, device, dtype)
        image_ones = []
        del osi_model.encoder
        del osi_model.quant_conv
        osi_model.unet.to('cpu')
        empty_cuda_cache(devices)

    results = {}
    if 'fari' in enabled:
        fari_prompt = configure_fari(pipe, args, device).detach()
        # The null prompt is already encoded. Drop the text encoder and VAE so
        # the FARI measurement contains only the UNet replica on each GPU.
        release_pipeline_modules(pipe, ('text_encoder', 'vae'), devices)
        fari_latents = regroup_latents(latent_ones, plan['fari']['batch_size'])
        fari_watermarks = batch_items(watermark_list, plan['fari']['batch_size'])
        with fari_mode(pipe.unet, FARIMode.INVERSION):
            results['fari'] = time_extractor(
                FARILatentToNoise(pipe.unet, fari_prompt, alpha_terminal),
                fari_latents, fari_watermarks, device, secondary_device,
                args.warmup_batches, parallel_mode='data')
        annotate_result(results['fari'], plan['fari'])
        print_result('fari', results['fari'])
        del fari_prompt
        empty_cuda_cache(devices)

    del pipe
    empty_cuda_cache(devices)

    if osi_model is not None:
        osi_model.unet.to(device)
        osi_latents = regroup_latents(osi_latent_ones, plan['osi']['batch_size'])
        osi_watermarks = batch_items(watermark_list, plan['osi']['batch_size'])
        results['osi'] = time_extractor(
            OSILatentToNoise(osi_model.unet, osi_prompt.to(dtype=dtype)),
            osi_latents, osi_watermarks, device, secondary_device,
            args.warmup_batches, parallel_mode='data')
        annotate_result(results['osi'], plan['osi'])
        print_result('osi', results['osi'])
        del osi_model, osi_prompt, osi_latent_ones
        empty_cuda_cache(devices)

    if 'ours' in enabled:
        model_args = SimpleNamespace(
            opt=str(Path(args.opt)), device=args.device, sd_model_id=args.sd_model_id)
        model, opt = build_model(model_args)
        if opt['sd_model_id'] != args.sd_model_id:
            raise ValueError(
                f'Ours uses {opt["sd_model_id"]!r}, expected {args.sd_model_id!r}.')
        convert_reverse_model_to_float32(model)
        ours_latents = regroup_latents(latent_ones, plan['ours']['batch_size'])
        ours_watermarks = batch_items(watermark_list, plan['ours']['batch_size'])
        results['ours'] = time_extractor(
            model, ours_latents, ours_watermarks, device, secondary_device,
            args.warmup_batches, parallel_mode='branch')
        results['ours']['option'] = str(Path(args.opt).resolve())
        results['ours']['model_type'] = opt['model_type']
        annotate_result(results['ours'], plan['ours'])
        print_result('ours', results['ours'])
        del model
        empty_cuda_cache(devices)

    ordered_results = {name: results[name] for name in EXTRACTOR_NAMES if name in results}
    batch_sizes = {name: plan[name]['batch_size'] for name in EXTRACTOR_NAMES}
    report = {
        'configuration': {
            'sd_model_id': args.sd_model_id,
            'generation_cache_dir': str(Path(args.generation_cache_dir).resolve()),
            'samples': args.num,
            'cache_load_batch_size': 1,
            'latent_shape': [LATENT_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH],
            'dtype': 'float32',
            'bytes_per_parameter': BYTES_PER_PARAMETER,
            'vram_policy': (
                'batch_size is inversely proportional to the calflops parameter '
                'count, so parameters x batch_size matches. The matched quantity '
                'is reported as parameter_batch_bytes.'
            ),
            'reference_method': 'fari',
            'reference_batch_size': args.reference_batch_size,
            'batch_plan': plan,
            'batch_sizes': batch_sizes,
            'warmup_batches': args.warmup_batches,
            'device': torch.cuda.get_device_name(device),
            'secondary_device': torch.cuda.get_device_name(secondary_device),
            'enabled_extractors': enabled,
            'allocation_strategy': {
                'fari': 'data parallel across both GPUs',
                'osi': 'data parallel across both GPUs',
                'ours': 'x-start and xT branch model parallelism',
            },
            'input_scope': (
                f'Exactly {args.num} clean images are loaded from cache or '
                'generated when missing, then encoded one at a time before '
                'timing. Each encoded sample is an fp32 latent of shape '
                f'(1, {LATENT_CHANNELS}, {LATENT_HEIGHT}, {LATENT_WIDTH}). '
                'FARI and Ours use the SD 2.1 VAE latent. OSI uses the latent '
                'from its own encoder and quantizer; that encoder pass is '
                'preparation and is excluded, as is the shared VAE encode. '
                'The SD text encoder and VAE are moved off CUDA and dropped '
                'after FARI encodes its null prompt, before FARI timing. '
                f'FARI is timed at batch {batch_sizes["fari"]}, OSI at batch '
                f'{batch_sizes["osi"]}, and Ours at batch {batch_sizes["ours"]}. '
                'Those batches equalize parameters x batch_size. Cache loading, '
                'host-to-device transfer, and Gaussian-Shading decoding are '
                'excluded. Detection rate and mean bit accuracy are computed '
                f'after each timed interval. FARI and OSI split every batch across '
                f'{device} and {secondary_device}. Ours runs x-start on {device} '
                f'and xT on {secondary_device}; the timed region includes '
                'cross-device output gathering and reconstruction. All prompts '
                'are null. Peak CUDA memory is the high-water mark from warmup '
                'through timing, with the resident weights included. Mean CUDA '
                'memory averages memory_allocated and memory_reserved on both '
                'GPUs after every timed batch, outside the timed interval.'),
        },
        'latency_comparison': [
            {
                'model': name,
                'batch_size': result['batch_size'],
                'params_m': result['params_m'],
                'parameter_batch_product_m': result['parameter_batch_product_m'],
                'parameter_batch_gib': format_gib(result['parameter_batch_bytes']),
                'mean_sample_ms': result['mean_sample_ms'],
                'throughput_images_per_second': result['throughput_images_per_second'],
                'detection_rate': result['detection_rate'],
                'mean_bit_accuracy': result['mean_bit_accuracy'],
                'allocation': result['allocation'],
                'peak_memory': result['peak_memory'],
                'mean_allocated_mib': result['mean_memory']['total']['mean_allocated_mib'],
                'mean_reserved_mib': result['mean_memory']['total']['mean_reserved_mib'],
                'mean_memory': result['mean_memory'],
            }
            for name, result in ordered_results.items()
        ],
        'results': ordered_results,
    }
    print('Latency comparison:')
    for item in report['latency_comparison']:
        print(
            f'  {item["model"]}: batch {item["batch_size"]}, '
            f'{item["mean_sample_ms"]:.3f} ms/image, '
            f'{item["throughput_images_per_second"]:.2f} images/s, '
            f'detection rate {item["detection_rate"]:.2%}, '
            f'mean allocated {item["mean_allocated_mib"]:.0f} MiB, '
            f'mean reserved {item["mean_reserved_mib"]:.0f} MiB')
    report_path = output_dir / 'latency.json'
    with open(report_path, 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    print(f'Saved latency report to {report_path}')


if __name__ == '__main__':
    main()

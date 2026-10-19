"""Benchmark FARI, OSI, and Ours extraction latency on cached clean SD 2.1 images.

Running time is measured by benchmark_model from the NAFNet latency script.
Cached images are encoded before that call. FARI and OSI are exposed as one
network and timed data-parallel. Ours is timed branch-parallel.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from torch import nn
from tqdm import tqdm

from benchmark_nafnet_watermark_extraction_latency import (
    benchmark_model,
    configure_two_gpu_model,
    configure_two_gpu_single_model,
)
from coco2017_diffusers_unipc_gaussian_shading_adv_eval_sdv21 import (
    DEFAULT_MODEL_ID,
    DEFAULT_OPTIONS,
    check_ours_backbone,
    configure_fari,
    load_osi_model,
)
from coco2017_diffusers_unipc_gaussian_shading_eval import (
    DEFAULT_CAPTIONS,
    generation_cache_path,
    load_coco_captions,
    load_or_generate_sample,
)
from fari_inversion import FARIMode, fari_mode
from reverse_distill_gaussian_shading_eval import build_model
from reverse_distill_unipc_eval import convert_reverse_model_to_float32


PROJECT_ROOT = Path(__file__).resolve().parent
EXTRACTOR_NAMES = ('fari', 'osi', 'ours')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Time FARI, OSI, and Ours extraction on cached clean SD 2.1 COCO '
            'images. FARI and OSI split each batch across two GPUs. Ours runs '
            'its x-start and xT branches on separate GPUs.'))
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
        default=str(PROJECT_ROOT / 'outputs' / 'sdv21_extraction_latency'))
    parser.add_argument('--num', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=10)
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


def batch_tensors(tensors, batch_size):
    return [
        torch.cat(tensors[start:start + batch_size])
        for start in range(0, len(tensors), batch_size)
    ]


def batch_items(items, batch_size):
    return [
        items[start:start + batch_size]
        for start in range(0, len(items), batch_size)
    ]


@torch.inference_mode()
def prepare_cached_inputs(cache_entries, pipe, args, batch_size, device, dtype, keep_images):
    """Encode one cached image at a time, then form CPU inference batches."""
    prepared_latents = []
    prepared_images = []
    watermarks = []
    for caption, seed, cache_path in tqdm(
            cache_entries,
            desc='Preparing cached clean images', unit='image'):
        watermark, _, _, generated_image, _, _, _, _ = load_or_generate_sample(
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
        vae_input = quantized * 2 - 1
        latent = pipe.vae.encode(vae_input.to(dtype=dtype)).latent_dist.mode()
        latent = latent * pipe.vae.config.scaling_factor
        prepared_latents.append(latent.cpu())
        if keep_images:
            prepared_images.append(vae_input.cpu())
        watermarks.append(watermark)
    return (
        batch_tensors(prepared_latents, batch_size),
        batch_tensors(prepared_images, batch_size) if keep_images else [],
        batch_items(watermarks, batch_size),
    )


@torch.inference_mode()
def encode_osi_latents(osi_model, image_batches, device, dtype):
    """Run the OSI encoder before timing, matching the excluded VAE encode."""
    encoded_batches = []
    for image_batch in tqdm(image_batches, desc='Preparing OSI latents', unit='batch'):
        image = image_batch.to(device=device, dtype=dtype)
        latent = osi_model.quant_conv(osi_model.encoder(image))[:, :4]
        latent = latent * osi_model.vae_scaling_factor
        encoded_batches.append(latent.cpu())
    return encoded_batches


def empty_cuda_cache(devices):
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


class FARILatentToNoise(nn.Module):
    """FARI one-step inversion, exposed as the single-network timed forward."""

    def __init__(self, unet, prompt, alpha_terminal):
        super().__init__()
        self.unet = unet
        self.register_buffer('prompt', prompt.detach().float())
        self.register_buffer(
            'alpha_terminal', alpha_terminal.detach().float().reshape(()))

    def forward(self, latent):
        prompt = self.prompt
        if prompt.shape[0] != latent.shape[0]:
            prompt = prompt.repeat(latent.shape[0], 1, 1)
        noise_prediction = self.unet(
            latent, 0, encoder_hidden_states=prompt, return_dict=False)[0]
        alpha_terminal = self.alpha_terminal.to(dtype=latent.dtype)
        return (
            alpha_terminal.sqrt() * latent
            + (1 - alpha_terminal).sqrt() * noise_prediction)


class OSILatentToNoise(nn.Module):
    """OSI U-Net call, exposed as the single-network timed forward."""

    def __init__(self, unet, prompt):
        super().__init__()
        self.unet = unet
        self.register_buffer('prompt', prompt.detach().float())

    def forward(self, latent):
        prompt = self.prompt
        if prompt.shape[0] != latent.shape[0]:
            prompt = prompt.repeat(latent.shape[0], 1, 1)
        timestep = torch.full(
            (latent.shape[0],), 999, device=latent.device, dtype=torch.long)
        return self.unet(
            latent, timestep, encoder_hidden_states=prompt, return_dict=False)[0]


class SingleExtractorHandle:
    """Give one network the interface benchmark_model uses for a single NAFNet."""

    def __init__(self, network):
        self.net_g = network
        self.prediction_type = 'direct_xT'

    def get_bare_model(self, net):
        return net


def time_single_extractor(
        network, latent_batches, watermark_batches, primary_device,
        secondary_device, warmup_batches):
    handle = SingleExtractorHandle(network)
    configure_two_gpu_single_model(handle, primary_device, secondary_device)
    result = benchmark_model(
        handle, latent_batches, watermark_batches, primary_device,
        warmup_batches, secondary_device, parallel_mode='data')
    result['allocation'] = 'data_parallel'
    return result


def require_checkpoint(path, name):
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'{name} checkpoint not found: {checkpoint}')
    return checkpoint


def validate_args(args, device, secondary_device):
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
    enabled = tuple(name for name in EXTRACTOR_NAMES if name not in args.skip_extractors)
    if not enabled:
        raise ValueError('At least one extractor must remain enabled.')
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


def print_result(name, result):
    print(
        f'{name}: {result["total_inference_ms"]:.3f} ms total, '
        f'{result["mean_sample_ms"]:.3f} ms/image, '
        f'{result["throughput_images_per_second"]:.2f} images/s, '
        f'detection rate {result["detection_rate"]:.2%}')


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.index is None and device.type == 'cuda':
        device = torch.device('cuda', torch.cuda.current_device())
    secondary_device = torch.device(args.xt_device)
    enabled = validate_args(args, device, secondary_device)
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
    latent_batches, image_batches, watermark_batches = prepare_cached_inputs(
        cache_entries, pipe, args, args.batch_size, device, dtype,
        keep_images='osi' in enabled)

    osi_model = None
    osi_prompt = None
    osi_latent_batches = None
    if 'osi' in enabled:
        # Copy the clean U-Net before FARI injects adapters into pipe.unet.
        osi_model, osi_prompt = load_osi_model(pipe, args, device, dtype)
        osi_prompt = osi_prompt.detach().cpu()
        osi_latent_batches = encode_osi_latents(
            osi_model, image_batches, device, dtype)
        image_batches = []
        del osi_model.encoder
        del osi_model.quant_conv
        osi_model.unet.to('cpu')
        empty_cuda_cache(devices)

    results = {}
    if 'fari' in enabled:
        fari_prompt = configure_fari(pipe, args, device).detach()
        with fari_mode(pipe.unet, FARIMode.INVERSION):
            results['fari'] = time_single_extractor(
                FARILatentToNoise(pipe.unet, fari_prompt, alpha_terminal),
                latent_batches, watermark_batches, device, secondary_device,
                args.warmup_batches)
        print_result('fari', results['fari'])
        del fari_prompt
        empty_cuda_cache(devices)

    del pipe
    empty_cuda_cache(devices)

    if osi_model is not None:
        osi_model.unet.to(device)
        results['osi'] = time_single_extractor(
            OSILatentToNoise(osi_model.unet, osi_prompt.to(dtype=dtype)),
            osi_latent_batches, watermark_batches, device, secondary_device,
            args.warmup_batches)
        print_result('osi', results['osi'])
        del osi_model, osi_prompt, osi_latent_batches
        empty_cuda_cache(devices)

    if 'ours' in enabled:
        model_args = SimpleNamespace(
            opt=str(Path(args.opt)), device=args.device, sd_model_id=args.sd_model_id)
        model, opt = build_model(model_args)
        if opt['sd_model_id'] != args.sd_model_id:
            raise ValueError(
                f'Ours uses {opt["sd_model_id"]!r}, expected {args.sd_model_id!r}.')
        convert_reverse_model_to_float32(model)
        configure_two_gpu_model(model, device, secondary_device)
        results['ours'] = benchmark_model(
            model, latent_batches, watermark_batches, device,
            args.warmup_batches, secondary_device, parallel_mode='branch')
        results['ours']['allocation'] = 'branch_parallel'
        results['ours']['option'] = str(Path(args.opt).resolve())
        results['ours']['model_type'] = opt['model_type']
        print_result('ours', results['ours'])
        del model
        empty_cuda_cache(devices)

    ordered_results = {name: results[name] for name in EXTRACTOR_NAMES if name in results}
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
                'timing. FARI and Ours use the SD 2.1 VAE latent. OSI uses the '
                'latent from its own encoder and quantizer; that encoder pass is '
                'preparation and is excluded, as is the shared VAE encode. The '
                f'CPU tensors are grouped into batches of {args.batch_size} only '
                'for extractor inference. Cache loading, host-to-device transfer, '
                'and Gaussian-Shading decoding are excluded. Detection rate and '
                'mean bit accuracy are computed after each timed interval. FARI '
                f'and OSI split every batch across {device} and {secondary_device}. '
                f'Ours runs x-start on {device} and xT on {secondary_device}; the '
                'timed region includes cross-device output gathering and '
                'reconstruction. All prompts are null.'),
        },
        'latency_comparison': [
            {
                'model': name,
                'mean_sample_ms': result['mean_sample_ms'],
                'throughput_images_per_second': result['throughput_images_per_second'],
                'detection_rate': result['detection_rate'],
                'mean_bit_accuracy': result['mean_bit_accuracy'],
                'allocation': result['allocation'],
            }
            for name, result in ordered_results.items()
        ],
        'results': ordered_results,
    }
    print('Latency comparison:')
    for item in report['latency_comparison']:
        print(
            f'  {item["model"]}: {item["mean_sample_ms"]:.3f} ms/image, '
            f'{item["throughput_images_per_second"]:.2f} images/s, '
            f'detection rate {item["detection_rate"]:.2%}')
    report_path = output_dir / 'latency.json'
    with open(report_path, 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    print(f'Saved latency report to {report_path}')


if __name__ == '__main__':
    main()

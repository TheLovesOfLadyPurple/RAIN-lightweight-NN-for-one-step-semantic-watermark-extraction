"""Profile NAFNet watermark extractors with calflops.

Counts GFLOPs, GMACs, and parameters for one encoded clean latent. Inference
timing lives in benchmark_nafnet_watermark_extraction_latency.py.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from calflops import calculate_flops
from diffusers import StableDiffusionPipeline
from torch import nn

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
    'two_branch_parallel': PROJECT_ROOT / 'options' / 'test' / 'cocoGSReverseDistillUNetNoneADV.yml',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Profile one NAFNet watermark extraction on a cached clean SD v1.5 '
            'COCO image with calflops.'))
    parser.add_argument('--single-direct-opt', default=str(DEFAULT_OPTIONS['single_direct_xt']))
    parser.add_argument('--single-epsilon-opt', default=str(DEFAULT_OPTIONS['single_epsilon_xt']))
    parser.add_argument('--two-branch-opt', default=str(DEFAULT_OPTIONS['two_branch_parallel']))
    parser.add_argument('--sd-model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--captions', default=str(DEFAULT_CAPTIONS))
    parser.add_argument(
        '--generation-cache-dir',
        default=str(PROJECT_ROOT / 'outputs' / 'coco_generation_cache'))
    parser.add_argument(
        '--refresh-generation-cache', action='store_true',
        help='Regenerate and overwrite the cached COCO sample.')
    parser.add_argument(
        '--outdir',
        default=str(PROJECT_ROOT / 'outputs' / 'nafnet_extraction_complexity'))
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
    return parser.parse_args()


class SingleNAFNetProfileAdapter(nn.Module):
    """Expose the single-NAFNet latent-to-noise call to calflops."""

    def __init__(self, model):
        super().__init__()
        self.net_g = model.net_g
        # The model stays callable for _predict without becoming a second parent
        # of net_g, which would make calflops add the same forward twice.
        object.__setattr__(self, 'reverse_model', model)
        self.last_output = None

    def forward(self, image_latent):
        self.reverse_model.lq = image_latent.float()
        output = self.reverse_model._predict(use_ema_model=False)[0]
        self.last_output = output.detach()
        return output


class TwoBranchProfileAdapter(nn.Module):
    """Expose the same reverse-model call as OursProfileAdapter."""

    def __init__(self, reverse_model):
        super().__init__()
        self.net_xstart = reverse_model.net_xstart
        self.net_xt = reverse_model.net_xt
        if reverse_model.xstart_prompt_tokens is not None:
            self.xstart_prompt_tokens = reverse_model.xstart_prompt_tokens
        object.__setattr__(self, 'reverse_model', reverse_model)
        self.last_output = None

    def forward(self, image_latent):
        self.reverse_model.lq = image_latent.float()
        self.reverse_model.prompt_emb = None
        self.reverse_model.use_prompt_emb = False
        output = self.reverse_model._predict(
            use_ema_model=False, clip_timestep=True)[-1]
        self.last_output = output.detach()
        return output


def profile_with_calflops(name, extractor, image_latent):
    """Count one real image latent with calflops."""
    extractor.eval()
    trainable_parameters = sum(
        parameter.numel() for parameter in extractor.parameters()
        if parameter.requires_grad)
    example = image_latent.detach().to(
        device=next(extractor.parameters()).device,
        dtype=torch.float32)
    if tuple(example.shape) != (1, 4, 64, 64):
        raise ValueError(
            'calflops expects one encoded latent of shape (1, 4, 64, 64), '
            f'got {tuple(example.shape)}.')
    with torch.inference_mode():
        flops, macs, calflops_parameters = calculate_flops(
            model=extractor,
            args=[example],
            print_results=False,
            print_detailed=False,
            output_as_string=False,
        )
    if calflops_parameters != trainable_parameters:
        raise RuntimeError(
            f'calflops parameter count for {name} does not match the explicit '
            f'trainable count: {calflops_parameters} != {trainable_parameters}.')
    if extractor.last_output is None:
        raise RuntimeError(f'calflops did not execute the {name} extractor.')
    return {
        'input_shape': [1, 4, 64, 64],
        'params': calflops_parameters,
        'params_m': calflops_parameters / 1e6,
        'macs': macs,
        'gmacs': macs / 1e9,
        'flops': flops,
        'gflops': flops / 1e9,
    }


def extractor_complexity(model, name, image_latent):
    """Profile one _predict call, excluding EMA copies and GPU replicas."""
    if name == 'two_branch_parallel':
        extractor = TwoBranchProfileAdapter(model)
    else:
        extractor = SingleNAFNetProfileAdapter(model)
    return profile_with_calflops(name, extractor, image_latent)


@torch.inference_mode()
def encode_cached_latent(args, pipe, device, dtype):
    """Encode the first cached clean image, matching the latency script's VAE path."""
    captions = load_coco_captions(Path(args.captions), 1, args.caption_seed)
    caption = captions[0]
    seed = args.seed
    cache_path = generation_cache_path(
        Path(args.generation_cache_dir), caption, seed, args, args.sd_model_id)
    (_, _, _, generated_image, _, _, _, _) = load_or_generate_sample(
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
    return latent.cpu().contiguous(), str(cache_path.resolve())


def main():
    args = parse_args()
    if args.device == 'cpu':
        raise ValueError('CUDA is required to profile the NAFNet extractors.')
    if not 12 <= args.steps <= 15:
        raise ValueError('--steps must be between 12 and 15 to match the cache key.')
    if args.height != 512 or args.width != 512:
        raise ValueError('The cached Gaussian-Shading images must be 512x512.')

    device = torch.device(args.device)
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    dtype = torch.float32
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_id,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    image_latent, cache_path = encode_cached_latent(args, pipe, device, dtype)
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
        complexity = extractor_complexity(model, name, image_latent)
        results[name] = {
            'option': str(option_path.resolve()),
            'model_type': opt['model_type'],
            **complexity,
        }
        print(
            f'{name}: {complexity["gflops"]:.3f} GFLOPs, '
            f'{complexity["gmacs"]:.3f} GMACs, '
            f'{complexity["params_m"]:.3f} M params')
        del model
        torch.cuda.empty_cache()

    report = {
        'configuration': {
            'sd_model_id': args.sd_model_id,
            'generation_cache_dir': str(Path(args.generation_cache_dir).resolve()),
            'cache_path': cache_path,
            'seed': args.seed,
            'caption_seed': args.caption_seed,
            'device': torch.cuda.get_device_name(device),
            'complexity_counting': (
                'GFLOPs, GMACs, and params come from calflops.calculate_flops on '
                'one cached image latent, shape (1, 4, 64, 64), using the same '
                'VAE encoding as timed inference. The single-NAFNet adapter calls '
                '_predict(use_ema_model=False). The two-branch adapter uses the '
                'same call as OursProfileAdapter: '
                '_predict(use_ema_model=False, clip_timestep=True). EMA copies '
                'and the second-GPU data-parallel replica are not registered. '
                'The two-branch count includes both NAFNet branches, the xT '
                'timestep head, and any loaded x-start prompt tokens. Image '
                'loading, VAE encoding, and Gaussian-Shading decoding are excluded.'),
        },
        'complexity_comparison': [
            {'model': name, **results[name]} for name in results
        ],
        'results': results,
    }
    print('Complexity per 1x4x64x64 latent:')
    for item in report['complexity_comparison']:
        print(
            f'  {item["model"]}: {item["gflops"]:.3f} GFLOPs, '
            f'{item["gmacs"]:.3f} GMACs, {item["params"]:,} params')
    report_path = output_dir / 'complexity.json'
    with open(report_path, 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    print(f'Saved complexity report to {report_path}')


if __name__ == '__main__':
    main()

"""Evaluate six SD v1.5 NAFNet extraction ablations on shared cached attacks."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from tqdm import tqdm

from coco2017_diffusers_unipc_gaussian_shading_adv_eval import (
    ATTACK_NAMES,
    aggregate_attack_results,
    apply_attack,
    encode_image,
    save_csv,
    tensor_to_pil,
)
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
)
from reverse_distill_unipc_eval import convert_reverse_model_to_float32


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = 'sd-legacy/stable-diffusion-v1-5'
MODEL_OPTIONS = {
    'single_epsilon_no_adv': (
        PROJECT_ROOT / 'options' / 'test' / 'cocoGSSingleNAFNetEpsilonXT.yml'),
    'single_epsilon_adv': (
        PROJECT_ROOT / 'options' / 'test' / 'cocoGSSingleNAFNetEpsilonXTAdv.yml'),
    'single_direct_no_adv': (
        PROJECT_ROOT / 'options' / 'test' / 'cocoGSSingleNAFNetDirectXT.yml'),
    'single_direct_adv': (
        PROJECT_ROOT / 'options' / 'test' / 'cocoGSSingleNAFNetDirectXTAdv.yml'),
    'two_branch_no_adv': (
        PROJECT_ROOT / 'options' / 'test' / 'cocoGSReverseDistillUNetNoneADV.yml'),
    'two_branch_adv': (
        PROJECT_ROOT / 'options' / 'test' / 'cocoGSReverseDistillUNetAdv.yml'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate six NAFNet extraction ablations on the same cached SD v1.5 '
            'COCO images and deterministic FARI-style attacks.'))
    parser.add_argument('--sd-model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--captions', default=str(DEFAULT_CAPTIONS))
    parser.add_argument(
        '--generation-cache-dir',
        default=str(PROJECT_ROOT / 'outputs' / 'coco_generation_cache'),
        help='Existing shared SD v1.5 GS generation cache.')
    parser.add_argument(
        '--outdir',
        default=str(PROJECT_ROOT / 'outputs' / 'coco_gs_nafnet_ablation_v15'))
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
    parser.add_argument(
        '--local-files-only', action='store_true',
        help='Load Stable Diffusion only from the local Hugging Face cache.')
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def resolve_cache_entries(args, captions):
    cache_dir = Path(args.generation_cache_dir)
    entries = []
    for index, caption_record in enumerate(captions):
        seed = args.seed + index
        cache_path = generation_cache_path(
            cache_dir, caption_record, seed, args, args.sd_model_id)
        entries.append((caption_record, seed, cache_path))
    missing = [cache_path for _, _, cache_path in entries
               if not cache_path.is_file()]
    if missing:
        examples = ', '.join(str(path) for path in missing[:3])
        raise FileNotFoundError(
            f'{len(missing)} of {len(entries)} required cache entries are '
            f'missing. This evaluator never regenerates samples. Examples: '
            f'{examples}')
    return entries


def discard_unused_ema_networks(model):
    for name in (
            'net_g_ema', 'net_xstart_ema', 'net_xt_ema',
            'xstart_prompt_tokens', 'xstart_prompt_tokens_ema'):
        if hasattr(model, name):
            delattr(model, name)


def load_models(args):
    models = {}
    options = {}
    for name, option_path in tqdm(
            MODEL_OPTIONS.items(), desc='Loading ablation models', unit='model'):
        model_args = SimpleNamespace(
            opt=str(option_path), device=args.device,
            sd_model_id=args.sd_model_id)
        model, option = build_model(model_args)
        if option['sd_model_id'] != args.sd_model_id:
            raise ValueError(
                f'{name} uses {option["sd_model_id"]!r}; expected '
                f'{args.sd_model_id!r}.')
        convert_reverse_model_to_float32(model)
        discard_unused_ema_networks(model)
        models[name] = model
        options[name] = option
    return models, options


@torch.inference_mode()
def predict_noise(model, attacked_latent):
    model.feed_data(
        {'final_latents': attacked_latent.float()}, use_prompt_emb=False)
    model.test()
    return model.output.detach()


def extraction_result(watermark, predicted_noise, target_noise):
    return {
        **watermark.evaluate(predicted_noise).to_dict(),
        'noise_metrics': latent_metrics(predicted_noise, target_noise),
        'prediction_diagnostics': prediction_diagnostics(
            predicted_noise, target_noise),
    }


def checkpoint_metadata(option):
    keys = (
        ('pretrain_network_g',)
        if option['model_type'] == 'ReverseDistillSingleNAFNetModel'
        else ('pretrain_network_xstart', 'pretrain_network_xt'))
    return {
        key: option['path'].get(key)
        for key in keys
    }


def main():
    args = parse_args()
    if args.num < 1:
        raise ValueError('--num must be positive.')
    if not 12 <= args.steps <= 15:
        raise ValueError('--steps must be between 12 and 15.')
    if args.height != 512 or args.width != 512:
        raise ValueError('Gaussian Shading evaluation requires 512x512.')
    if args.device == 'cpu':
        raise ValueError('Gaussian Shading evaluation requires CUDA.')
    if args.sd_model_id != DEFAULT_MODEL_ID:
        raise ValueError(f'This ablation requires {DEFAULT_MODEL_ID}.')

    device = torch.device(args.device)
    dtype = torch.float32
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    captions = load_coco_captions(
        Path(args.captions), args.num, args.caption_seed)
    cache_entries = resolve_cache_entries(args, captions)
    args.refresh_generation_cache = False

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_id,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    models, options = load_models(args)

    results = []
    total_extractions = len(cache_entries) * len(ATTACK_NAMES) * len(models)
    progress = tqdm(
        total=total_extractions,
        desc='COCO SD 1.5 NAFNet ablation',
        unit='extraction')
    for index, (caption_record, seed, cache_path) in enumerate(cache_entries):
        (watermark, watermarked_noise, _, generated_image,
         _, _, _, used_generation_cache) = load_or_generate_sample(
             cache_path,
             caption_record['caption'],
             seed,
             args,
             pipe,
             device,
             dtype,
             require_cache=True,
         )
        base_image = tensor_to_pil(generated_image[0])
        attacks = {}
        for attack_index, attack_name in enumerate(ATTACK_NAMES):
            attacked_image = apply_attack(
                base_image,
                attack_name,
                args,
                seed + 10_007 * attack_index,
            )
            attacked_latent = encode_image(pipe, attacked_image, dtype)
            attacks[attack_name] = {}
            for model_name, model in models.items():
                predicted_noise = predict_noise(model, attacked_latent)
                attacks[attack_name][model_name] = extraction_result(
                    watermark, predicted_noise, watermarked_noise)
                progress.update(1)
                progress.set_postfix(
                    sample=f'{index + 1}/{len(cache_entries)}',
                    attack=attack_name,
                    model=model_name,
                    cache='hit' if used_generation_cache else 'miss',
                    refresh=False,
                )
        results.append({
            'index': index,
            'seed': seed,
            **caption_record,
            'generation_cache_path': str(cache_path.resolve()),
            'used_generation_cache': used_generation_cache,
            'attacks': attacks,
        })
    progress.close()

    aggregates = {
        name: aggregate_attack_results(results, name)
        for name in models
    }
    report = {
        'configuration': {
            'sd_model_id': args.sd_model_id,
            'caption_file': str(Path(args.captions).resolve()),
            'sample_count': args.num,
            'steps': args.steps,
            'guidance_scale': args.guidance_scale,
            'generation_cache_dir': str(
                Path(args.generation_cache_dir).resolve()),
            'cache_policy': 'existing valid entries required; no regeneration',
            'attack_order': ATTACK_NAMES,
            'models': {
                name: {
                    'option': str(MODEL_OPTIONS[name].resolve()),
                    'model_type': option['model_type'],
                    'prediction_type': option.get('prediction_type'),
                    'adversarial_training': name.endswith('_adv'),
                    'checkpoints': checkpoint_metadata(option),
                }
                for name, option in options.items()
            },
            'protocol_note': (
                'All six extractors use the same cached SD v1.5 image, '
                'Gaussian-Shading state, deterministic attack realization, and '
                'VAE-encoded attacked latent. Each attacked image is encoded once '
                'and that latent is reused by every extractor.'),
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
        'aggregate': aggregates,
        'samples': results,
    }
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    save_csv(aggregates, output_dir / 'aggregate.csv')
    print(json.dumps(aggregates, indent=2))
    print(f'Saved six-model ablation results to {output_dir}')


if __name__ == '__main__':
    main()

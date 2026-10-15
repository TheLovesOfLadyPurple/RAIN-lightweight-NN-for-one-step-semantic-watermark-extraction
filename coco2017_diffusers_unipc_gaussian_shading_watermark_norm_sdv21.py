"""L2 norm of the watermark latent recovered by each SD 2.1 GS extractor."""

import argparse
import csv
import json
from pathlib import Path

import torch
from diffusers import DDIMInverseScheduler, StableDiffusionPipeline, UniPCMultistepScheduler
from tqdm import tqdm

from coco2017_diffusers_unipc_gaussian_shading_adv_eval import (
    ATTACK_NAMES,
    apply_attack,
    encode_image,
    tensor_to_pil,
)
from coco2017_diffusers_unipc_gaussian_shading_adv_eval_sdv21 import (
    DDIM_INVERSE_GUIDANCE_SCALE,
    DDIM_INVERSE_STEPS,
    EXTRACTOR_NAMES,
    check_ours_backbone,
    configure_fari,
    ddim_inverse_noise,
    load_osi_model,
    osi_one_step_inversion,
)
from coco2017_diffusers_unipc_gaussian_shading_eval import (
    DEFAULT_CAPTIONS,
    generation_cache_path,
    load_coco_captions,
    load_or_generate_sample,
)
from fari_inversion import one_step_inversion
from reverse_distill_gaussian_shading_eval import build_model, reverse_noise
from reverse_distill_unipc_eval import convert_reverse_model_to_float32


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = 'sd2-community/stable-diffusion-2-1-base'
DEFAULT_OPTIONS = PROJECT_ROOT / 'options' / 'test' / 'cocoGSReverseDistillUNetAdvSDV21.yml'
NORM_FIELDS = (
    'sample_count', 'mean_l2', 'std_l2', 'min_l2', 'max_l2',
    'mean_rms', 'std_rms', 'mean_target_l2', 'mean_l2_ratio')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure the L2 norm of SD 2.1 GS latents recovered under each attack.')
    parser.add_argument(
        '--opt', default=str(DEFAULT_OPTIONS),
        help='SD 2.1 reverse-extractor option used for the Ours comparison.')
    parser.add_argument('--fari-checkpoint', default=str(PROJECT_ROOT / 'fari_weights.pth'))
    parser.add_argument('--sd-model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--fari-lora-rank', type=int, default=8)
    parser.add_argument('--osi-unet-checkpoint', default='./osi_sd21_unet.pth')
    parser.add_argument('--osi-encoder-checkpoint', default='./osi_sd21_encoder.pth')
    parser.add_argument(
        '--skip-extractors', nargs='*', choices=EXTRACTOR_NAMES, default=(),
        help='Extractors to omit. For example: --skip-extractors fari ours evaluates OSI only.')
    parser.add_argument('--captions', default=str(DEFAULT_CAPTIONS))
    parser.add_argument(
        '--outdir', default=str(PROJECT_ROOT / 'outputs' / 'coco_gs_sdv21_watermark_norm'))
    parser.add_argument(
        '--generation-cache-dir',
        default=str(PROJECT_ROOT / 'outputs' / 'coco_generation_cache_sdv21'),
        help='Separate SD 2.1 generation cache; never reuses the SD 1.5 cache.')
    parser.add_argument('--refresh-generation-cache', action='store_true')
    parser.add_argument('--num', type=int, default=1000)
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


def noise_norm(latent):
    """L2 norm of a recovered or original Gaussian Shading latent."""
    values = latent.detach().float().flatten()
    l2 = values.norm(p=2).item()
    return {
        'l2': l2,
        'rms': l2 / values.numel() ** 0.5,
    }


def extracted_norm(predicted_noise, target_noise):
    extracted = noise_norm(predicted_noise)
    target = noise_norm(target_noise)
    return {
        **extracted,
        'target_l2': target['l2'],
        'l2_ratio': extracted['l2'] / target['l2'],
    }


def summarize_norms(records):
    def column(name):
        return torch.tensor([item[name] for item in records], dtype=torch.float64)

    l2 = column('l2')
    rms = column('rms')
    target_l2 = column('target_l2')
    ratio = column('l2_ratio')
    return {
        'sample_count': len(records),
        'mean_l2': l2.mean().item(),
        'std_l2': l2.std(correction=0).item(),
        'min_l2': l2.min().item(),
        'max_l2': l2.max().item(),
        'mean_rms': rms.mean().item(),
        'std_rms': rms.std(correction=0).item(),
        'mean_target_l2': target_l2.mean().item(),
        'mean_l2_ratio': ratio.mean().item(),
    }


def save_norm_csv(summary, path):
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['extractor', 'attack', *NORM_FIELDS])
        writer.writeheader()
        for extractor_name, attacks in summary.items():
            for attack_name, metrics in attacks.items():
                writer.writerow({'extractor': extractor_name, 'attack': attack_name, **metrics})


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
    enabled_extractors = tuple(name for name in EXTRACTOR_NAMES if name not in args.skip_extractors)
    if not enabled_extractors:
        raise ValueError('At least one extractor must remain enabled.')
    if 'fari' in enabled_extractors and args.fari_lora_rank < 1:
        raise ValueError('--fari-lora-rank must be positive.')
    if 'osi' in enabled_extractors and (
            not args.osi_encoder_checkpoint or not args.osi_unet_checkpoint):
        raise ValueError(
            'OSI evaluation requires --osi-encoder-checkpoint and --osi-unet-checkpoint. '
            'Use --skip-extractors osi to omit it.')
    if 'ours' in enabled_extractors:
        check_ours_backbone(args)

    device = torch.device(args.device)
    dtype = torch.float32
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.generation_cache_dir)
    captions = load_coco_captions(Path(args.captions), args.num, args.caption_seed)
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_model_id, torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    cache_files = []
    used_generation_cache = []
    generation_bar = tqdm(captions, unit='sample', desc='SD 2.1 generation cache')
    for index, caption_record in enumerate(generation_bar):
        seed = args.seed + index
        cache_file = generation_cache_path(
            cache_dir, caption_record, seed, args, args.sd_model_id)
        *_, cache_hit = load_or_generate_sample(
            cache_file, caption_record['caption'], seed, args, pipe, device, dtype)
        cache_files.append(cache_file)
        used_generation_cache.append(cache_hit)
        generation_bar.set_postfix_str(f'cache={"hit" if cache_hit else "miss"}')
    args.refresh_generation_cache = False

    osi_model = osi_null_prompt_embedding = None
    if 'osi' in enabled_extractors:
        osi_model, osi_null_prompt_embedding = load_osi_model(pipe, args, device, dtype)
    fari_null_prompt_embedding = None
    if 'fari' in enabled_extractors:
        fari_null_prompt_embedding = configure_fari(pipe, args, device)
    model = opt = None
    if 'ours' in enabled_extractors:
        model, opt = build_model(args)
        convert_reverse_model_to_float32(model)
    null_prompt_embedding = None
    if 'ours' in enabled_extractors:
        null_prompt_embedding, _ = pipe.encode_prompt('', device, 1, False)
        null_prompt_embedding = null_prompt_embedding.detach()
    ddim_inverse_scheduler = None
    extractors_before_ddim = tuple(name for name in enabled_extractors if name != 'ddim')

    def extract_noise(extractor_name, image, latent):
        if extractor_name == 'fari':
            return one_step_inversion(pipe, latent, fari_null_prompt_embedding)
        if extractor_name == 'osi':
            return osi_one_step_inversion(osi_model, image, osi_null_prompt_embedding, device, dtype)
        if extractor_name == 'ddim':
            return ddim_inverse_noise(
                pipe, latent, null_prompt_embedding,
                ddim_inverse_scheduler, DDIM_INVERSE_GUIDANCE_SCALE)
        if extractor_name == 'ours':
            return reverse_noise(model, latent, null_prompt_embedding)['predicted_noise']
        raise ValueError(f'Unknown extractor: {extractor_name}')

    norm_records = {
        name: {attack_name: [] for attack_name in ATTACK_NAMES}
        for name in enabled_extractors}
    samples = []

    def measure_attacks(base_image, watermarked_noise, seed, extractor_names):
        attack_norms = {attack_name: {} for attack_name in ATTACK_NAMES}
        for attack_index, attack_name in enumerate(ATTACK_NAMES):
            attacked_image = apply_attack(
                base_image, attack_name, args, seed + 10_007 * attack_index)
            attacked_latent = encode_image(pipe, attacked_image, dtype)
            for extractor_name in extractor_names:
                predicted_noise = extract_noise(extractor_name, attacked_image, attacked_latent)
                record = extracted_norm(predicted_noise, watermarked_noise)
                attack_norms[attack_name][extractor_name] = record
                norm_records[extractor_name][attack_name].append(record)
        return attack_norms

    progress_bar = tqdm(captions, unit='sample', desc='SD 2.1 watermark norm')
    for index, caption_record in enumerate(progress_bar):
        prompt = caption_record['caption']
        seed = args.seed + index
        cache_file = cache_files[index]
        (_, watermarked_noise, _, generated_image, _, _, _, _) = load_or_generate_sample(
            cache_file, prompt, seed, args, pipe, device, dtype, require_cache=True)
        base_image = tensor_to_pil(generated_image[0])
        attacks = measure_attacks(base_image, watermarked_noise, seed, extractors_before_ddim)
        samples.append({
            'index': index,
            'seed': seed,
            **caption_record,
            'generation_cache_path': str(cache_file.resolve()),
            'used_generation_cache': used_generation_cache[index],
            'watermark_noise': noise_norm(watermarked_noise),
            'attacks': attacks,
        })

    if 'ddim' in enabled_extractors:
        del pipe
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        pipe = StableDiffusionPipeline.from_pretrained(args.sd_model_id, torch_dtype=dtype)
        ddim_inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
        ddim_inverse_scheduler.set_timesteps(DDIM_INVERSE_STEPS, device=device)
        pipe.safety_checker = None
        pipe.to(device=device, dtype=dtype)
        pipe.set_progress_bar_config(disable=True)
        pipe.unet.eval()
        null_prompt_embedding, _ = pipe.encode_prompt('', device, 1, False)
        null_prompt_embedding = null_prompt_embedding.detach()
        ddim_bar = tqdm(captions, unit='sample', desc='SD 2.1 DDIM watermark norm')
        for index, caption_record in enumerate(ddim_bar):
            prompt = caption_record['caption']
            seed = args.seed + index
            (_, watermarked_noise, _, generated_image, _, _, _, _) = load_or_generate_sample(
                cache_files[index], prompt, seed, args, pipe, device, dtype, require_cache=True)
            base_image = tensor_to_pil(generated_image[0])
            attacks = measure_attacks(base_image, watermarked_noise, seed, ('ddim',))
            for attack_name, records in attacks.items():
                samples[index]['attacks'][attack_name].update(records)

    summary = {
        extractor_name: {
            attack_name: summarize_norms(norm_records[extractor_name][attack_name])
            for attack_name in ATTACK_NAMES
        }
        for extractor_name in enabled_extractors
    }
    watermark_l2 = torch.tensor(
        [sample['watermark_noise']['l2'] for sample in samples], dtype=torch.float64)
    report = {
        'configuration': {
            'enabled_extractors': enabled_extractors,
            'skipped_extractors': tuple(args.skip_extractors),
            'fari_checkpoint': (str(Path(args.fari_checkpoint).resolve())
                                if 'fari' in enabled_extractors else None),
            'osi_encoder_checkpoint': (str(Path(args.osi_encoder_checkpoint).resolve())
                                       if 'osi' in enabled_extractors else None),
            'osi_unet_checkpoint': (str(Path(args.osi_unet_checkpoint).resolve())
                                    if 'osi' in enabled_extractors else None),
            'sd_model_id': args.sd_model_id,
            'ours_option': str(Path(args.opt).resolve()) if opt else None,
            'ours_xstart_checkpoint': opt['path']['pretrain_network_xstart'] if opt else None,
            'ours_xt_checkpoint': opt['path']['pretrain_network_xt'] if opt else None,
            'sample_count': args.num,
            'steps': args.steps,
            'guidance_scale': args.guidance_scale,
            'ddim_inverse_steps': DDIM_INVERSE_STEPS if 'ddim' in enabled_extractors else None,
            'ddim_guidance_scale': (
                DDIM_INVERSE_GUIDANCE_SCALE if 'ddim' in enabled_extractors else None),
            'generation_cache_dir': str(cache_dir.resolve()),
            'attack_order': ATTACK_NAMES,
            'watermark_noise_mean_l2': watermark_l2.mean().item(),
            'watermark_noise_std_l2': watermark_l2.std(correction=0).item(),
            'protocol_note': (
                'The extracted watermark is the latent recovered by each extractor from the '
                'attacked watermarked image. l2 is its Euclidean norm and rms is l2 divided by '
                'sqrt(4*64*64). target_l2 is the norm of the original Gaussian Shading noise, '
                'and l2_ratio is extracted l2 divided by that target. Attacks and seeds match '
                'the SD 2.1 robustness evaluation. DDIM inversion reloads a fresh pipeline.'),
        },
        'summary': summary,
        'samples': samples,
    }
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    save_norm_csv(summary, output_dir / 'watermark_norm.csv')
    print(json.dumps(summary, indent=2))
    print(f'Saved extracted-watermark norms to {output_dir}')


if __name__ == '__main__':
    main()

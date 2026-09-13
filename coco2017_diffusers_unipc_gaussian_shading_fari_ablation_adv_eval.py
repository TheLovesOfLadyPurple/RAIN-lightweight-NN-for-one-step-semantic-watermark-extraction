"""Evaluate four SD v1.5 FARI LoRA ablations on identical GS attack inputs."""

import argparse
import json
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, UniPCMultistepScheduler
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
from fari_inversion import (
    FARIMode,
    adapter_state_dict,
    fari_mode,
    inject_fari_adapters,
    one_step_inversion,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = 'sd-legacy/stable-diffusion-v1-5'
ABLATIONS = {
    'adv_direct': 'FARI-adv-direct-v15',
    'adv_fari': 'FARI-adv-v15',
    'no_adv_direct': 'FARI-noneADV-direct-v15',
    'no_adv_fari': 'FARI-normal-v15',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare four FARI SD v1.5 training ablations under all GS attacks.')
    parser.add_argument('--sd-model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--captions', default=str(DEFAULT_CAPTIONS))
    parser.add_argument(
        '--outdir', default=str(PROJECT_ROOT / 'outputs' / 'coco_gs_fari_ablation_v15'))
    parser.add_argument(
        '--generation-cache-dir',
        default=str(PROJECT_ROOT / 'outputs' / 'coco_generation_cache'),
        help='Shared SD v1.5 GS generation cache; all ablations use these same samples.')
    parser.add_argument('--refresh-generation-cache', action='store_true')
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
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_ablation_specs():
    specs = {}
    for name, directory_name in ABLATIONS.items():
        directory = PROJECT_ROOT / 'experiments' / directory_name
        checkpoint = directory / 'fari_adapters.pth'
        settings_file = directory / 'training_settings.json'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'{name} FARI adapter checkpoint not found: {checkpoint}')
        if not settings_file.is_file():
            raise FileNotFoundError(f'{name} training settings not found: {settings_file}')
        with open(settings_file, encoding='utf-8') as file:
            settings = json.load(file)
        if settings.get('sd_model_id') != DEFAULT_MODEL_ID:
            raise ValueError(f'{name} was not trained with {DEFAULT_MODEL_ID}.')
        prediction_type = settings.get('prediction_type')
        if prediction_type not in ('fari', 'direct_xt'):
            raise ValueError(f'{name} has unsupported prediction type: {prediction_type!r}')
        specs[name] = {
            'experiment_dir': directory,
            'checkpoint': checkpoint,
            'prediction_type': prediction_type,
            'use_adv_training': settings.get('use_adv_training'),
        }
    return specs


def load_adapter_checkpoint(unet, checkpoint):
    state_dict = torch.load(checkpoint, map_location='cpu')
    expected_keys = set(adapter_state_dict(unet))
    actual_keys = set(state_dict)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f'Adapter checkpoint mismatch for {checkpoint}: '
            f'{len(missing)} missing and {len(unexpected)} unexpected keys.')
    unet.load_state_dict(state_dict, strict=False)


@torch.no_grad()
def predict_terminal_noise(pipe, image_latent, null_prompt_embedding, prediction_type):
    if prediction_type == 'fari':
        return one_step_inversion(pipe, image_latent, null_prompt_embedding)
    if prediction_type == 'direct_xt':
        model_input = pipe.scheduler.scale_model_input(image_latent, 0)
        with fari_mode(pipe.unet, FARIMode.INVERSION):
            return pipe.unet(
                model_input, 0, encoder_hidden_states=null_prompt_embedding,
                return_dict=False)[0]
    raise ValueError(f'Unsupported prediction type: {prediction_type}')


def extraction_result(watermark, predicted_noise, target_noise):
    from reverse_distill_gaussian_shading_eval import latent_metrics, prediction_diagnostics

    return {
        **watermark.evaluate(predicted_noise).to_dict(),
        'noise_metrics': latent_metrics(predicted_noise, target_noise),
        'prediction_diagnostics': prediction_diagnostics(predicted_noise, target_noise),
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
        raise ValueError(f'This ablation evaluator requires {DEFAULT_MODEL_ID}.')

    specs = load_ablation_specs()
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
    inject_fari_adapters(pipe.unet, rank=8)
    pipe.unet.eval()
    null_prompt_embedding, _ = pipe.encode_prompt('', device, 1, False)
    null_prompt_embedding = null_prompt_embedding.detach()

    results = []
    progress_bar = tqdm(captions, unit='sample', desc='COCO SD 1.5 FARI ablation')
    for index, caption_record in enumerate(progress_bar):
        prompt = caption_record['caption']
        seed = args.seed + index
        cache_file = generation_cache_path(cache_dir, caption_record, seed, args, args.sd_model_id)
        (watermark, watermarked_noise, _, generated_image,
         _, _, _, used_generation_cache) = load_or_generate_sample(
             cache_file, prompt, seed, args, pipe, device, dtype)
        base_image = tensor_to_pil(generated_image[0])
        attacks = {}
        for attack_index, attack_name in enumerate(ATTACK_NAMES):
            attacked_image = apply_attack(
                base_image, attack_name, args, seed + 10_007 * attack_index)
            attacked_latent = encode_image(pipe, attacked_image, dtype)
            attacks[attack_name] = {}
            for name, spec in specs.items():
                load_adapter_checkpoint(pipe.unet, spec['checkpoint'])
                predicted_noise = predict_terminal_noise(
                    pipe, attacked_latent, null_prompt_embedding, spec['prediction_type'])
                attacks[attack_name][name] = extraction_result(
                    watermark, predicted_noise, watermarked_noise)
        results.append({
            'index': index,
            'seed': seed,
            **caption_record,
            'generation_cache_path': str(cache_file.resolve()),
            'used_generation_cache': used_generation_cache,
            'attacks': attacks,
        })
        progress_bar.set_postfix_str(f'cache={"hit" if used_generation_cache else "miss"}')

    aggregates = {name: aggregate_attack_results(results, name) for name in specs}
    report = {
        'configuration': {
            'sd_model_id': args.sd_model_id,
            'sample_count': args.num,
            'steps': args.steps,
            'guidance_scale': args.guidance_scale,
            'generation_cache_dir': str(cache_dir.resolve()),
            'attack_order': ATTACK_NAMES,
            'ablation_extractors': {
                name: {
                    'checkpoint': str(spec['checkpoint'].resolve()),
                    'prediction_type': spec['prediction_type'],
                    'use_adv_training': spec['use_adv_training'],
                }
                for name, spec in specs.items()
            },
            'protocol_note': (
                'All four extractors use the same cached SD v1.5 GS images, VAE '
                'encoding, watermark state, and deterministic attack realizations. '
                'Only the FARI LoRA adapter checkpoint and its recorded prediction type differ.'),
        },
        'aggregate': aggregates,
        'samples': results,
    }
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    save_csv(aggregates, output_dir / 'aggregate.csv')
    print(json.dumps(aggregates, indent=2))
    print(f'Saved SD 1.5 FARI ablation results to {output_dir}')


if __name__ == '__main__':
    main()

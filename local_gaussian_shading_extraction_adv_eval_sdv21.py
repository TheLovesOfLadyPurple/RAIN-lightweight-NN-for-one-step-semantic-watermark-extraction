"""Compare FARI, OSI, and Ours on clean SD 2.1 Gaussian-Shading images."""

import argparse
import copy
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from tqdm import tqdm

from local_gaussian_shading_test_eval import (
    DEFAULT_PROMPTS,
    generation_cache_path,
    load_or_generate_sample,
    vae_roundtrip_image,
)
from fari_inversion import (
    inject_fari_adapters,
    load_official_fari_state_dict,
    one_step_inversion,
)
from reverse_distill_gaussian_shading_eval import (
    build_model,
    latent_metrics,
    mean_metric,
    prediction_diagnostics,
    reverse_noise,
)
from reverse_distill_unipc_eval import convert_reverse_model_to_float32, prompt_embeddings


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = 'sd2-community/stable-diffusion-2-1'
DEFAULT_LOCAL_TEST_OPTIONS = (
    PROJECT_ROOT / 'options' / 'test' / 'local_test_GS_reverse_distill_uet_adv_sdV21.yml')
EXTRACTOR_NAMES = ('fari', 'osi', 'ours')


class OSIModel(torch.nn.Module):
    """The SD v2.1 OSI image encoder, quantizer, and one-step U-Net."""

    def __init__(self, unet, encoder, quant_conv, vae_scaling_factor):
        super().__init__()
        self.unet = unet
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.vae_scaling_factor = vae_scaling_factor

    def forward(self, image, timestep, prompt_embeds):
        latent = self.quant_conv(self.encoder(image))[:, :4] * self.vae_scaling_factor
        noise_prediction = self.unet(
            latent, timestep, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
        return noise_prediction


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare FARI, OSI, and Ours on clean SD 2.1 GS images.')
    parser.add_argument(
        '--opt',
        default=str(DEFAULT_LOCAL_TEST_OPTIONS),
        help='SD 2.1 reverse-extractor option used for the Ours comparison.')
    parser.add_argument('--fari-checkpoint', default=str(PROJECT_ROOT / 'fari_weights.pth'))
    parser.add_argument('--sd-model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--fari-lora-rank', type=int, default=8)
    parser.add_argument(
        '--osi-unet-checkpoint',
        help='OSI SD v2.1 U-Net checkpoint from the run_osi_sd21 reference implementation.',
        default="./osi_sd21_unet.pth"
        )
    parser.add_argument(
        '--osi-encoder-checkpoint',
        help='OSI SD v2.1 encoder and quant_conv checkpoint from the run_osi_sd21 reference implementation.',
        default="./osi_sd21_encoder.pth"
        )
    parser.add_argument(
        '--skip-extractors', nargs='*', choices=EXTRACTOR_NAMES, default=('fari', 'osi',),
        help='Extractors to omit. For example: --skip-extractors fari ours evaluates OSI only.')
    parser.add_argument(
        '--prompts',
        nargs='+',
        default=DEFAULT_PROMPTS,
        metavar='PROMPT',
        help=(
            'One or more image prompts. Quote each multi-word prompt. '
            'Defaults to ten built-in prompts.'))
    parser.add_argument(
        '--outdir', default=str(PROJECT_ROOT / 'outputs' / 'prompted_gs_fari_sdv21'))
    parser.add_argument(
        '--generation-cache-dir',
        default=str(PROJECT_ROOT / 'outputs' / 'prompt_generation_cache_sdv21'),
        help='Separate SD 2.1 generation cache; never reuses the SD 1.5 cache.')
    parser.add_argument('--refresh-generation-cache', action='store_true', help='Regenerate and overwrite cached prompt samples.')
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--guidance-scale', type=float, default=5.5)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--channel-copy', type=int, default=1)
    parser.add_argument('--hw-copy', type=int, default=8)
    parser.add_argument('--user-number', type=int, default=1_000_000)
    parser.add_argument('--fpr', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def configure_fari(pipe, args, device):
    checkpoint = Path(args.fari_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'FARI checkpoint not found: {checkpoint}')
    inject_fari_adapters(pipe.unet, args.fari_lora_rank)
    load_official_fari_state_dict(pipe.unet, torch.load(checkpoint, map_location='cpu'))
    pipe.unet.eval()
    null_prompt_embedding, _ = pipe.encode_prompt('', device, 1, False)
    return null_prompt_embedding.detach()


def load_osi_model(pipe, args, device, dtype):
    encoder_checkpoint = Path(args.osi_encoder_checkpoint)
    unet_checkpoint = Path(args.osi_unet_checkpoint)
    for checkpoint, name in ((encoder_checkpoint, 'encoder'), (unet_checkpoint, 'U-Net')):
        if not checkpoint.is_file():
            raise FileNotFoundError(f'OSI {name} checkpoint not found: {checkpoint}')

    model = OSIModel(
        unet=copy.deepcopy(pipe.unet),
        encoder=copy.deepcopy(pipe.vae.encoder),
        quant_conv=copy.deepcopy(pipe.vae.quant_conv),
        vae_scaling_factor=pipe.vae.config.scaling_factor).to(device=device, dtype=dtype)
    encoder_state = torch.load(encoder_checkpoint, map_location='cpu')
    unet_state = torch.load(unet_checkpoint, map_location='cpu')
    encoder_state = {name.replace('module.', ''): value for name, value in encoder_state.items()}
    unet_state = {name.replace('module.', ''): value for name, value in unet_state.items()}
    model.encoder.load_state_dict(encoder_state, strict=False)
    model.quant_conv.load_state_dict(encoder_state, strict=False)
    model.unet.load_state_dict(unet_state, strict=False)
    model.eval()
    null_prompt_embedding, _ = pipe.encode_prompt('', device, 1, False)
    return model, null_prompt_embedding.detach()


@torch.no_grad()
def osi_one_step_inversion(model, image, prompt_embeds, dtype):
    image = (image * 255).round().clamp(0, 255) / 255
    image = image * 2 - 1
    timestep = torch.full(
        (image.shape[0],), 999, device=image.device, dtype=torch.long)
    return model(image.to(dtype=dtype), timestep, prompt_embeds)


def aggregate_extraction_results(samples, extractor_name):
    metric_prefix = ('extractions', extractor_name)
    return {
        'sample_count': len(samples),
        'detection_rate': mean_metric(samples, metric_prefix + ('detected',)),
        'traceability_rate': mean_metric(samples, metric_prefix + ('traceable',)),
        'mean_bit_accuracy': mean_metric(samples, metric_prefix + ('bit_accuracy',)),
    }


def main():
    args = parse_args()
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

    device = torch.device(args.device)
    dtype = torch.float32
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.generation_cache_dir)
    prompts = [prompt.strip() for prompt in args.prompts]
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError('--prompts must contain at least one non-empty prompt.')
    prompt_records = [
        {'prompt_id': index, 'prompt': prompt}
        for index, prompt in enumerate(prompts)
    ]
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_model_id, torch_dtype=dtype)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    osi_model = osi_null_prompt_embedding = None
    if 'osi' in enabled_extractors:
        osi_model, osi_null_prompt_embedding = load_osi_model(pipe, args, device, dtype)
    fari_null_prompt_embedding = None
    if 'fari' in enabled_extractors:
        fari_null_prompt_embedding = configure_fari(pipe, args, device)
    model = opt = None
    if 'ours' in enabled_extractors:
        model, opt = build_model(args)
        if opt['sd_model_id'] != args.sd_model_id:
            raise ValueError(
                'The Ours option and --sd-model-id must use the same SD 2.1 backbone. '
                f'Got {opt["sd_model_id"]!r} and {args.sd_model_id!r}.')
        convert_reverse_model_to_float32(model)

    results = []
    progress_bar = tqdm(
        prompt_records, unit='sample', desc='Prompted SD 2.1 GS evaluation')
    for index, prompt_record in enumerate(progress_bar):
        prompt = prompt_record['prompt']
        prompt_embedding = (prompt_embeddings(pipe, [prompt], device, dtype)
                            if 'ours' in enabled_extractors else None)
        seed = args.seed + index
        cache_file = generation_cache_path(
            cache_dir, prompt_record, seed, args, args.sd_model_id)
        (watermark, watermarked_noise, _, generated_image,
         _, _, _, used_generation_cache) = load_or_generate_sample(
             cache_file, prompt, seed, args, pipe, device, dtype)
        image_latent = vae_roundtrip_image(pipe, generated_image)
        extractions = {}
        if 'fari' in enabled_extractors:
            predicted_noise = one_step_inversion(
                pipe, image_latent, fari_null_prompt_embedding)
            watermark_result = watermark.evaluate(predicted_noise)
            extractions['fari'] = {
                **watermark_result.to_dict(),
                'noise_metrics': latent_metrics(predicted_noise, watermarked_noise),
                'prediction_diagnostics': prediction_diagnostics(
                    predicted_noise, watermarked_noise),
            }
        if 'osi' in enabled_extractors:
            predicted_noise = osi_one_step_inversion(
                osi_model, generated_image, osi_null_prompt_embedding, dtype)
            watermark_result = watermark.evaluate(predicted_noise)
            extractions['osi'] = {
                **watermark_result.to_dict(),
                'noise_metrics': latent_metrics(predicted_noise, watermarked_noise),
                'prediction_diagnostics': prediction_diagnostics(
                    predicted_noise, watermarked_noise),
            }
        if 'ours' in enabled_extractors:
            ours_result = reverse_noise(model, image_latent, prompt_embedding)
            predicted_noise = ours_result['predicted_noise']
            watermark_result = watermark.evaluate(predicted_noise)
            extractions['ours'] = {
                **watermark_result.to_dict(),
                'noise_metrics': latent_metrics(predicted_noise, watermarked_noise),
                'prediction_diagnostics': prediction_diagnostics(
                    predicted_noise, watermarked_noise),
            }
        results.append({
            'index': index,
            'seed': seed,
            **prompt_record,
            'generation_cache_path': str(cache_file.resolve()),
            'used_generation_cache': used_generation_cache,
            'extractions': extractions,
        })
        progress_bar.set_postfix_str(f'cache={"hit" if used_generation_cache else "miss"}')

    aggregates = {
        name: aggregate_extraction_results(results, name)
        for name in enabled_extractors
    }
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
            'prompts': prompts,
            'sample_count': len(prompts),
            'steps': args.steps,
            'guidance_scale': args.guidance_scale,
            'generation_cache_dir': str(cache_dir.resolve()),
            'protocol_note': (
                'Clean generated images are evaluated without adversarial attacks. '
                'FARI and OSI use their native SD 2.1-base components. '
                'Generation cache entries are separated from the SD 1.5 evaluation cache.'),
        },
        'aggregate': aggregates,
        'samples': results,
    }
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report['aggregate'], indent=2))
    print(f'Saved SD 2.1 extraction results to {output_dir}')


if __name__ == '__main__':
    main()
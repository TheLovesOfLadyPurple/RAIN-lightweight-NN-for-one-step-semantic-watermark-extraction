"""Profile one SD v2.1 Gaussian-Shading extraction with calflops."""

import argparse
import json
from pathlib import Path

import torch
from calflops import calculate_flops
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from torch import nn

from coco2017_diffusers_unipc_gaussian_shading_adv_eval import encode_image, tensor_to_pil
from coco2017_diffusers_unipc_gaussian_shading_adv_eval_sdv21 import (
    DEFAULT_MODEL_ID,
    DEFAULT_OPTIONS,
    EXTRACTOR_NAMES,
    configure_fari,
    load_osi_model,
)
from diffusers_unipc_gaussian_shading_eval import generate_latent
from fari_inversion import FARIMode, fari_mode
from gaussian_shading_watermark import GaussianShadingWatermark
from reverse_distill_gaussian_shading_eval import (
    build_model,
    latent_metrics,
    prediction_diagnostics,
)
from reverse_distill_unipc_eval import (
    convert_reverse_model_to_float32,
    decode_latent,
)


PROJECT_ROOT = Path(__file__).resolve().parent


class FARIProfileAdapter(nn.Module):
    """Expose the existing FARI latent-to-noise call to calflops."""

    def __init__(self, pipe, prompt_embedding):
        super().__init__()
        self.unet = pipe.unet
        self.register_buffer('prompt_embedding', prompt_embedding)
        self.register_buffer(
            'alpha_terminal', pipe.scheduler.alphas_cumprod[-1].detach().float())
        self.last_output = None

    def forward(self, image_latent):
        with fari_mode(self.unet, FARIMode.INVERSION):
            noise_prediction = self.unet(
                image_latent,
                0,
                encoder_hidden_states=self.prompt_embedding,
                return_dict=False,
            )[0]
        alpha_terminal = self.alpha_terminal.to(dtype=image_latent.dtype)
        output = (
            alpha_terminal.sqrt() * image_latent
            + (1 - alpha_terminal).sqrt() * noise_prediction)
        self.last_output = output.detach()
        return output


class OSIProfileAdapter(nn.Module):
    """Expose only OSI's latent-to-noise U-Net backbone to calflops."""

    def __init__(self, model, prompt_embedding):
        super().__init__()
        self.unet = model.unet
        self.register_buffer('prompt_embedding', prompt_embedding)
        self.last_output = None

    def forward(self, image_latent):
        timestep = torch.full(
            (image_latent.shape[0],), 999,
            device=image_latent.device, dtype=torch.long)
        output = self.unet(
            image_latent,
            timestep,
            encoder_hidden_states=self.prompt_embedding,
            return_dict=False,
        )[0]
        self.last_output = output.detach()
        return output


class OursProfileAdapter(nn.Module):
    """Expose the existing reverse model's latent-to-noise call to calflops."""

    def __init__(self, reverse_model):
        super().__init__()
        self.net_xstart = reverse_model.net_xstart
        self.net_xt = reverse_model.net_xt
        if reverse_model.xstart_prompt_tokens is not None:
            self.xstart_prompt_tokens = reverse_model.xstart_prompt_tokens
        self.reverse_model = reverse_model
        self.last_output = None

    def forward(self, image_latent):
        self.reverse_model.lq = image_latent.float()
        self.reverse_model.prompt_emb = None
        self.reverse_model.use_prompt_emb = False
        output = self.reverse_model._predict(
            use_ema_model=False, clip_timestep=True)[-1]
        self.last_output = output.detach()
        return output


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate one SD v2.1 GS image and profile one FARI, OSI, and Ours '
            'watermark extraction with calflops.'))
    parser.add_argument('--opt', default=str(DEFAULT_OPTIONS))
    parser.add_argument('--sd-model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--fari-checkpoint', default=str(PROJECT_ROOT / 'fari_weights.pth'))
    parser.add_argument('--fari-lora-rank', type=int, default=8)
    parser.add_argument('--osi-unet-checkpoint', default=str(PROJECT_ROOT / 'osi_sd21_unet.pth'))
    parser.add_argument('--osi-encoder-checkpoint', default=str(PROJECT_ROOT / 'osi_sd21_encoder.pth'))
    parser.add_argument(
        '--skip-extractors', nargs='*', choices=EXTRACTOR_NAMES, default=(),
        help='Extractors to omit, for example: --skip-extractors fari osi.')
    parser.add_argument('--prompt', default='a photo of a cat sitting on a wooden chair')
    parser.add_argument('--outdir', default=str(PROJECT_ROOT / 'outputs' / 'extraction_cost_sdv21'))
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--guidance-scale', type=float, default=5.5)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--channel-copy', type=int, default=1)
    parser.add_argument('--hw-copy', type=int, default=8)
    parser.add_argument('--user-number', type=int, default=1_000_000)
    parser.add_argument('--fpr', type=float, default=1e-6)
    parser.add_argument(
        '--local-files-only', action='store_true',
        help='Load the Stable Diffusion pipeline only from the local cache.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def human_count(value, unit):
    for scale, prefix in ((1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K')):
        if value >= scale:
            return f'{value / scale:.3f} {prefix}{unit}'
    return f'{value} {unit}'


def profile_extractor(name, extractor, image_input, watermark, target_noise):
    extractor.eval()
    total_parameters = sum(parameter.numel() for parameter in extractor.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in extractor.parameters()
        if parameter.requires_grad)
    with torch.inference_mode():
        flops, macs, calflops_trainable_parameters = calculate_flops(
            model=extractor,
            args=[image_input],
            print_results=False,
            print_detailed=False,
            output_as_string=False,
        )
    if calflops_trainable_parameters != trainable_parameters:
        raise RuntimeError(
            f'calflops parameter count for {name} does not match the explicit '
            f'trainable count: {calflops_trainable_parameters} != '
            f'{trainable_parameters}.')
    if extractor.last_output is None:
        raise RuntimeError(f'calflops did not execute the {name} extractor.')
    predicted_noise = extractor.last_output
    watermark_result = watermark.evaluate(predicted_noise)
    return {
        'flops': flops,
        'macs': macs,
        'parameters': total_parameters,
        'trainable_parameters': trainable_parameters,
        'flops_human': human_count(flops, 'FLOPs'),
        'macs_human': human_count(macs, 'MACs'),
        'parameters_human': human_count(total_parameters, ' parameters'),
        'trainable_parameters_human': human_count(
            trainable_parameters, ' parameters'),
        'watermark': watermark_result.to_dict(),
        'noise_metrics': latent_metrics(predicted_noise, target_noise),
        'prediction_diagnostics': prediction_diagnostics(predicted_noise, target_noise),
    }


def main():
    args = parse_args()
    if args.device == 'cpu':
        raise ValueError('The SD v2.1 extraction-cost benchmark requires CUDA.')
    if not 12 <= args.steps <= 15:
        raise ValueError('--steps must be between 12 and 15.')
    if args.height != 512 or args.width != 512:
        raise ValueError('Gaussian Shading evaluation requires 512x512.')
    enabled_extractors = tuple(
        name for name in EXTRACTOR_NAMES if name not in args.skip_extractors)
    if not enabled_extractors:
        raise ValueError('At least one extractor must remain enabled.')

    device = torch.device(args.device)
    dtype = torch.float32
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_id,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    watermark = GaussianShadingWatermark(
        channel_copy=args.channel_copy,
        hw_copy=args.hw_copy,
        fpr=args.fpr,
        user_number=args.user_number,
    )
    watermarked_noise = watermark.create_latent(
        device=device, dtype=dtype, generator=generator)
    generated_latent = generate_latent(
        pipe, args.prompt, watermarked_noise, args.steps, args.guidance_scale,
        args.height, args.width)
    generated_image = decode_latent(pipe, generated_latent)
    generated_pil = tensor_to_pil(generated_image[0])
    generated_pil.save(output_dir / 'generated_watermarked.png')
    image_input = generated_image.mul(2).sub(1).to(device=device, dtype=dtype)
    encoded_latent = encode_image(pipe, generated_pil, dtype)

    extractors = {}

    if 'osi' in enabled_extractors:
        osi_model, osi_prompt_embedding = load_osi_model(pipe, args, device, dtype)
        with torch.inference_mode():
            osi_latent = osi_model.quant_conv(
                osi_model.encoder(image_input))[:, :4] * osi_model.vae_scaling_factor
        extractors['osi'] = (
            OSIProfileAdapter(osi_model, osi_prompt_embedding), osi_latent)

    ours_opt = None
    if 'ours' in enabled_extractors:
        reverse_model, ours_opt = build_model(args)
        if ours_opt['sd_model_id'] != args.sd_model_id:
            raise ValueError(
                'The Ours option and --sd-model-id must match. '
                f'Got {ours_opt["sd_model_id"]!r} and {args.sd_model_id!r}.')
        convert_reverse_model_to_float32(reverse_model)
        extractors['ours'] = (OursProfileAdapter(reverse_model), encoded_latent)

    if 'fari' in enabled_extractors:
        fari_prompt_embedding = configure_fari(pipe, args, device)
        extractors['fari'] = (FARIProfileAdapter(pipe, fari_prompt_embedding), encoded_latent)

    results = {}
    for name in enabled_extractors:
        extractor, extractor_input = extractors[name]
        results[name] = profile_extractor(
            name, extractor, extractor_input, watermark, watermarked_noise)
        print(
            f'{name}: {results[name]["flops_human"]}, '
            f'{results[name]["macs_human"]}, '
            f'{results[name]["parameters_human"]} total, '
            f'{results[name]["trainable_parameters_human"]} trainable')

    report = {
        'configuration': {
            'sd_model_id': args.sd_model_id,
            'prompt': args.prompt,
            'seed': args.seed,
            'steps': args.steps,
            'guidance_scale': args.guidance_scale,
            'image_size': [args.height, args.width],
            'enabled_extractors': enabled_extractors,
            'fari_checkpoint': (
                str(Path(args.fari_checkpoint).resolve())
                if 'fari' in enabled_extractors else None),
            'osi_encoder_checkpoint': (
                str(Path(args.osi_encoder_checkpoint).resolve())
                if 'osi' in enabled_extractors else None),
            'osi_unet_checkpoint': (
                str(Path(args.osi_unet_checkpoint).resolve())
                if 'osi' in enabled_extractors else None),
            'ours_option': str(Path(args.opt).resolve()) if ours_opt else None,
            'profile_scope': (
                'One batch-size-1 latent-to-noise backbone pass. FARI and Ours use '
                'the shared pre-encoded SD v2.1 latent. OSI uses the latent produced '
                'by its native encoder, but that encoder and quantizer are executed '
                'before profiling and excluded from FLOPs, MACs, and parameters. '
                'Image generation, latent encoding, and GS decoding are excluded. '
                'Parameters reports all backbone weights; trainable_parameters '
                'separately reports weights with requires_grad enabled.'),
        },
        'results': results,
    }
    with open(output_dir / 'inference_cost.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    print(f'Saved extraction cost report to {output_dir / "inference_cost.json"}')


if __name__ == '__main__':
    main()

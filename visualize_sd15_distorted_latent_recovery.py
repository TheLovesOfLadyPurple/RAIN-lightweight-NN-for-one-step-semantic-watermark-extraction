"""Visualize SD v1.5 trajectory, distortion, and recovered-noise latents."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from PIL import Image, ImageDraw

from coco2017_diffusers_unipc_gaussian_shading_adv_eval import (
    ATTACK_NAMES,
    apply_attack,
    encode_image,
)
from gaussian_shading_watermark import GaussianShadingWatermark
from reverse_distill_gaussian_shading_eval import (
    build_model,
    latent_metrics,
    prediction_diagnostics,
    reverse_noise,
)
from reverse_distill_unipc_eval import (
    convert_reverse_model_to_float32,
    decode_latent,
    prompt_embeddings,
    save_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = 'sd-legacy/stable-diffusion-v1-5'
DEFAULT_OPTIONS = (
    PROJECT_ROOT / 'options' / 'test' / 'cocoGSReverseDistillUNetAdv.yml')


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate one SD v1.5 image, distort it, and visualize the '
            'generation trajectory latent, distorted-image latent, and '
            'noise recovered by the project model.'))
    parser.add_argument('--prompt', default='a smiling woman')
    parser.add_argument('--opt', default=str(DEFAULT_OPTIONS))
    parser.add_argument('--sd-model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument(
        '--outdir',
        default=str(PROJECT_ROOT / 'outputs' / 'sd15_jpeg_latent_visualization'))
    parser.add_argument('--attack', choices=ATTACK_NAMES, default='jpeg')
    parser.add_argument('--trajectory-step', type=int, default=7)
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--guidance-scale', type=float, default=5.5)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--seed', type=int, default=25)
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
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def pil_from_tensor(image_tensor):
    pixels = (
        image_tensor.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        * 255).round().astype('uint8')
    return Image.fromarray(pixels, mode='RGB')


def save_four_channel_png(output_dir, name, latent):
    channels = latent[0].detach().float().cpu()
    if channels.shape[0] != 4:
        raise ValueError(
            f'Expected {name} to have four channels, got {channels.shape[0]}.')
    value_min = channels.min()
    value_range = (channels.max() - value_min).clamp_min(1e-6)
    pixels = (
        (channels - value_min) / value_range * 255
    ).round().to(torch.uint8).permute(1, 2, 0).numpy()
    path = output_dir / f'{name}_rgba.png'
    Image.fromarray(pixels, mode='RGBA').save(path)
    return path


def save_latent_artifacts(output_dir, name, latent, pipe):
    latent = latent.detach()
    torch.save(latent.cpu(), output_dir / f'{name}.pt')
    rgba_path = save_four_channel_png(output_dir, name, latent)
    decoded = decode_latent(pipe, latent.to(device=pipe.device, dtype=torch.float32))
    save_image(decoded[0], output_dir / f'{name}_vae_decoded.png')
    return rgba_path


def remove_legacy_channel_images(output_dir):
    for pattern in ('*_channel_?.png', '*_channels.png'):
        for path in output_dir.glob(pattern):
            path.unlink()
    legacy_overview = output_dir / 'four_channel_overview.png'
    if legacy_overview.is_file():
        legacy_overview.unlink()


def make_contact_sheet(output_dir, items, filename='overview.png'):
    thumb_size = (256, 256)
    label_height = 32
    columns = 3
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new(
        'RGB', (columns * thumb_size[0], rows * (thumb_size[1] + label_height)),
        'white')
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert('RGB')
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        column = index % columns
        row = index // columns
        left = column * thumb_size[0] + (thumb_size[0] - image.width) // 2
        top = row * (thumb_size[1] + label_height)
        sheet.paste(image, (left, top))
        draw.text(
            (column * thumb_size[0] + 6, top + thumb_size[1] + 8),
            label, fill='black')
    sheet.save(output_dir / filename)


@torch.inference_mode()
def main():
    args = parse_args()
    if args.device == 'cpu':
        raise ValueError('Gaussian Shading generation requires CUDA.')
    if args.height % 8 or args.width % 8:
        raise ValueError('--height and --width must be divisible by 8.')
    if not 0 <= args.trajectory_step < args.steps:
        raise ValueError('--trajectory-step must be in [0, steps).')

    device = torch.device(args.device)
    dtype = torch.float32
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_channel_images(output_dir)

    model_args = SimpleNamespace(
        opt=args.opt, device=args.device, sd_model_id=args.sd_model_id)
    model, option = build_model(model_args)
    convert_reverse_model_to_float32(model)
    if option['sd_model_id'] != args.sd_model_id:
        raise ValueError(
            f'Model uses {option["sd_model_id"]!r}, expected '
            f'{args.sd_model_id!r}.')

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_id,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to(device=device, dtype=dtype)
    pipe.set_progress_bar_config(disable=False)

    torch.manual_seed(args.seed)
    watermark = GaussianShadingWatermark(
        channel_copy=args.channel_copy,
        hw_copy=args.hw_copy,
        fpr=args.fpr,
        user_number=args.user_number,
    )
    terminal_noise = watermark.create_latent(device=device, dtype=dtype)
    trajectory = {}

    def capture_trajectory_step(_, step_index, timestep, callback_kwargs):
        if step_index == args.trajectory_step:
            trajectory['latent'] = callback_kwargs['latents'].detach().clone()
            trajectory['timestep'] = int(timestep.item())
        return callback_kwargs

    generated_latent = pipe(
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        latents=terminal_noise,
        output_type='latent',
        callback_on_step_end=capture_trajectory_step,
        callback_on_step_end_tensor_inputs=['latents'],
    ).images
    if 'latent' not in trajectory:
        raise RuntimeError('The requested trajectory latent was not captured.')

    generated_tensor = decode_latent(pipe, generated_latent)
    generated_image = pil_from_tensor(generated_tensor[0])
    generated_path = output_dir / 'generated_image.png'
    generated_image.save(generated_path)

    distorted_image = apply_attack(
        generated_image, args.attack, args, args.seed + 10_007)
    distorted_path = output_dir / f'distorted_image_{args.attack}.png'
    distorted_image.save(distorted_path)
    distorted_latent = encode_image(pipe, distorted_image, dtype)

    prompt_embedding = prompt_embeddings(pipe, [args.prompt], device, dtype)
    recovery = reverse_noise(
        model,
        distorted_latent,
        None,
        use_custom_model=True,
        pipe=pipe,
    )
    predicted_noise = recovery['predicted_noise']

    latents = {
        'terminal_watermarked_noise': terminal_noise,
        'trajectory_noisy_latent': trajectory['latent'],
        'generated_final_latent': generated_latent,
        'distorted_image_latent': distorted_latent,
        'predicted_terminal_noise': predicted_noise,
        'model_predicted_xstart': recovery['predicted_xstart'],
        'model_predicted_xt': recovery['predicted_xt'],
    }
    four_channel_paths = {
        name: save_latent_artifacts(output_dir, name, latent, pipe)
        for name, latent in latents.items()
    }

    evaluation = watermark.evaluate(predicted_noise)
    report = {
        'prompt': args.prompt,
        'seed': args.seed,
        'sd_model_id': args.sd_model_id,
        'model_option': str(Path(args.opt).resolve()),
        'attack': args.attack,
        'attack_seed': args.seed + 10_007,
        'inference_steps': args.steps,
        'trajectory_step_index': args.trajectory_step,
        'trajectory_scheduler_timestep': trajectory['timestep'],
        'guidance_scale': args.guidance_scale,
        'watermark_evaluation': evaluation.to_dict(),
        'recovered_noise_metrics': latent_metrics(predicted_noise, terminal_noise),
        'prediction_diagnostics': prediction_diagnostics(
            predicted_noise, terminal_noise),
        'artifacts': {
            'raw_latents': [f'{name}.pt' for name in latents],
            'four_channel_pngs': {
                name: path.name
                for name, path in four_channel_paths.items()
            },
            'vae_decoded_visualizations': [
                f'{name}_vae_decoded.png' for name in latents],
            'generated_image': generated_path.name,
            'distorted_image': distorted_path.name,
            'overview': 'overview.png',
        },
    }
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)

    make_contact_sheet(output_dir, [
        ('Generated image', generated_path),
        (f'Distorted: {args.attack}', distorted_path),
        ('Trajectory latent RGBA',
         four_channel_paths['trajectory_noisy_latent']),
        ('Trajectory latent decoded',
         output_dir / 'trajectory_noisy_latent_vae_decoded.png'),
        ('Distorted-image latent RGBA',
         four_channel_paths['distorted_image_latent']),
        ('Distorted-image latent decoded',
         output_dir / 'distorted_image_latent_vae_decoded.png'),
        ('Predicted noise RGBA',
         four_channel_paths['predicted_terminal_noise']),
        ('Predicted noise decoded',
         output_dir / 'predicted_terminal_noise_vae_decoded.png'),
        ('Target noise RGBA',
         four_channel_paths['terminal_watermarked_noise']),
    ])
    print(json.dumps(report, indent=2))
    print(f'Saved latent and image visualizations to {output_dir.resolve()}')


if __name__ == '__main__':
    main()

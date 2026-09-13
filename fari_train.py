"""Train FARI inversion-only LoRA adapters on distorted Stable Diffusion images."""

import argparse
import json
import random
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm

from coco2017_diffusers_unipc_gaussian_shading_adv_eval import apply_attack, pil_to_tensor
from coco2017_diffusers_unipc_gaussian_shading_eval import DEFAULT_CAPTIONS, load_coco_captions
from fari_inversion import (
    FARIMode,
    adapter_state_dict,
    fari_mode,
    inject_fari_adapters,
    one_step_inversion,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_CAPTIONS = PROJECT_ROOT / 'captions_train2014.json'
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description='Train FARI inversion-only LoRA adapters.')
    parser.add_argument('--sd-model-id', default='sd-legacy/stable-diffusion-v1-5')
    parser.add_argument('--prompts', default=str(DEFAULT_TRAIN_CAPTIONS))
    parser.add_argument('--outdir', default=str(PROJECT_ROOT / 'experiments' / 'FARI'))
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--guidance-scale', type=float, default=5.5)
    parser.add_argument('--inference-steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--use-adv-training',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Apply one FARI-style image distortion to every training sample.')
    parser.add_argument(
        '--prediction-type',
        choices=('fari', 'direct_xt'),
        default='fari',
        help='Use FARI DDIM reconstruction or train the inversion U-Net to directly predict x_T.')
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


def predict_terminal_noise(pipe, image_latent, null_prompt_embedding, prediction_type):
    if prediction_type == 'fari':
        return one_step_inversion(pipe, image_latent, null_prompt_embedding)
    if prediction_type == 'direct_xt':
        model_input = pipe.scheduler.scale_model_input(image_latent, 0)
        with fari_mode(pipe.unet, FARIMode.INVERSION):
            return pipe.unet(
                model_input,
                0,
                encoder_hidden_states=null_prompt_embedding,
                return_dict=False)[0]
    raise ValueError(f'Unsupported prediction type: {prediction_type}')


def main():
    args = parse_args()
    if args.device == 'cpu':
        raise ValueError('FARI training requires CUDA.')
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError('--steps and --batch-size must be positive.')

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pipe = StableDiffusionPipeline.from_pretrained(args.sd_model_id, torch_dtype=torch.float32)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    adapter_parameters = inject_fari_adapters(pipe.unet, args.lora_rank)
    optimizer = torch.optim.Adam(adapter_parameters, lr=args.lr)
    null_prompt_embedding, _ = pipe.encode_prompt('', device, 1, False)

    prompts = load_coco_captions(Path(args.prompts), args.steps, args.seed)
    losses = []
    progress_bar = tqdm(prompts, total=args.steps, unit='step', desc='FARI training')
    for step, prompt_record in enumerate(progress_bar):
        prompt = prompt_record['caption']
        targets = []
        predictions = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index in range(args.batch_size):
            generator = torch.Generator(device=device).manual_seed(
                args.seed + step * args.batch_size + batch_index)
            target_noise = torch.randn((1, 4, 64, 64), generator=generator, device=device)
            with torch.no_grad():
                image = pipe(
                    prompt, latents=target_noise, guidance_scale=args.guidance_scale,
                    num_inference_steps=args.inference_steps, height=512, width=512).images[0]
            if args.use_adv_training:
                attack_index = (step * args.batch_size + batch_index) % 10
                image_input = apply_attack(
                    image,
                    ('clean_roundtrip', 'jpeg', 'random_crop', 'random_drop', 'resize',
                     'gaussian_blur', 'median_blur', 'gaussian_noise', 'salt_pepper', 'brightness')[attack_index],
                    args, args.seed + step * args.batch_size + batch_index)
            else:
                image_input = image
            with torch.no_grad():
                image_latent = pipe.vae.encode(
                    (pil_to_tensor(image_input, device).unsqueeze(0) * 2 - 1)
                ).latent_dist.mode() * pipe.vae.config.scaling_factor
            predictions.append(predict_terminal_noise(
                pipe, image_latent, null_prompt_embedding.detach(), args.prediction_type))
            targets.append(target_noise)

        loss = torch.nn.functional.mse_loss(torch.cat(predictions), torch.cat(targets))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        progress_bar.set_postfix(loss=f'{loss.item():.5f}')

    checkpoint_path = output_dir / 'fari_adapters.pth'
    temporary_path = checkpoint_path.with_suffix('.tmp')
    torch.save(adapter_state_dict(pipe.unet), temporary_path)
    temporary_path.replace(checkpoint_path)
    metadata = {**vars(args), 'checkpoint': str(checkpoint_path.resolve()), 'losses': losses}
    with open(output_dir / 'training_settings.json', 'w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2)
    print(f'Saved FARI adapters to {checkpoint_path}')


if __name__ == '__main__':
    main()
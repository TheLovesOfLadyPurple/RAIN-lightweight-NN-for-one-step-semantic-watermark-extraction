"""Evaluate reverse-distilled noise predictions with Diffusers UniPC sampling."""

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image

from data.options import parse
from diffusers import StableDiffusionPipeline
from models import create_model
from sampler import UniPCSampler


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate predicted x_T values using the official Diffusers UniPC sampler.')
    parser.add_argument(
        '--opt', default='./options/test/reverseDistillUNetUniPCEval.yml',
        help='Complete reverse-distillation evaluation option YAML.')
    parser.add_argument(
        '--sd-model-id', default=None,
        help='Optional override for the Stable Diffusion model in the option YAML.')
    parser.add_argument('--outdir', default='./outputs/reverse_distill_unipc_eval')
    parser.add_argument('--prompt', default='a smiling woman')
    parser.add_argument('--guidance-scale', type=float, default=5.5)
    parser.add_argument('--unipc-steps', type=int, default=6)
    parser.add_argument('--start-free-u-step', type=int, default=4)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--no-prompt-emb', action='store_true',
        help='Do not provide CLIP prompt embeddings to the reverse-distillation model.')
    return parser.parse_args()


def build_model(args):
    opt = parse(args.opt, is_train=False)
    opt['dist'] = False
    opt['num_gpu'] = 0 if args.device == 'cpu' else 1
    if args.sd_model_id:
        opt['sd_model_id'] = args.sd_model_id
    return create_model(opt), opt


def convert_reverse_model_to_float32(model):
    for network_name in (
            'net_xstart', 'net_xt', 'net_xstart_ema', 'net_xt_ema',
            'xstart_prompt_tokens', 'xstart_prompt_tokens_ema', 'net_g',
            'net_g_ema'):
        network = getattr(model, network_name, None)
        if network is not None:
            network.float()


def prompt_embeddings(pipe, prompts, device, dtype):
    tokens = pipe.tokenizer(
        prompts,
        padding='max_length',
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors='pt')
    return pipe.text_encoder(tokens.input_ids.to(device))[0].to(dtype=dtype)


@torch.no_grad()
def create_network_input(sampler, terminal_noise, cond_embeds, uncond_embeds,
                         start_free_u_step):
    """Generate the final custom-UniPC latent used as the saved offline lq."""
    return generate_latent_with_custom_unipc(
        sampler, terminal_noise, cond_embeds, uncond_embeds, start_free_u_step)


@torch.no_grad()
def generate_latent_with_custom_unipc(sampler, noise, cond_embeds, uncond_embeds,
                                      start_free_u_step):
    """Generate one latent with the project custom UniPC sampler."""
    latent, _ = sampler.sample(
        conditioning=cond_embeds,
        unconditional_conditioning=uncond_embeds,
        batch_size=noise.shape[0],
        shape=noise.shape[1:],
        x_T=noise,
        start_free_u_step=start_free_u_step,
        use_corrector=True,
    )
    return latent


@torch.no_grad()
def decode_latent(pipe, latent):
    decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
    return (decoded / 2 + 0.5).clamp(0, 1)


@torch.no_grad()
def model_closure(pipe):
    def model_fn(latent, timestep, prompt_embeds):
        return pipe.unet(
            latent, timestep, encoder_hidden_states=prompt_embeds).sample
    return model_fn


def save_image(image_tensor, path):
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    Image.fromarray((image * 255).round().astype('uint8')).save(path)


def save_noise_image(noise_tensor, path):
    """Save all four latent-noise channels as a normalized 2x2 grayscale grid."""
    noise = noise_tensor[0].detach().float().cpu()
    value_range = (noise.max() - noise.min()).clamp_min(1e-6)
    pixels = ((noise - noise.min()) / value_range * 255).round().to(torch.uint8)
    noise_grid = torch.cat([
        torch.cat([pixels[0], pixels[1]], dim=1),
        torch.cat([pixels[2], pixels[3]], dim=1),
    ], dim=0)
    Image.fromarray(noise_grid.numpy(), mode='L').save(path)


def main():
    args = parse_args()
    if args.height % 8 or args.width % 8:
        raise ValueError('--height and --width must be divisible by 8.')
    opt_for_check = parse(args.opt, is_train=False)
    checkpoint_keys = ['pretrain_network_xstart', 'pretrain_network_xt']
    if opt_for_check.get('use_pred_xstart_prompt', False):
        checkpoint_keys.append('pretrain_xstart_prompt_tokens')
    for checkpoint_key in checkpoint_keys:
        checkpoint = opt_for_check['path'].get(checkpoint_key)
        if not checkpoint:
            raise ValueError(f'The option YAML must define path.{checkpoint_key}.')
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    device = torch.device(args.device)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, opt = build_model(args)
    dtype = torch.float32
    convert_reverse_model_to_float32(model)
    pipe = StableDiffusionPipeline.from_pretrained(
        opt['sd_model_id'], torch_dtype=dtype)
    pipe.to(device=device, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    sampler = UniPCSampler(
        pipe,
        model_closure=model_closure,
        steps=args.unipc_steps,
        guidance_scale=args.guidance_scale,
        is_high_resoulution=False,
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)
    cond_embeds = prompt_embeddings(pipe, [args.prompt], device, dtype)
    uncond_embeds = prompt_embeddings(pipe, [''], device, dtype)
    terminal_noise = torch.randn(
        (1, 4, args.height // 8, args.width // 8), generator=generator,
        device=device, dtype=dtype)
    network_input = create_network_input(
        sampler, terminal_noise, cond_embeds, uncond_embeds,
        args.start_free_u_step)
    model_data = {'final_latents': network_input.float()}
    if not args.no_prompt_emb:
        model_data['prompt'] = cond_embeds.float()
    model.feed_data(
        model_data, use_prompt_emb=not args.no_prompt_emb)
    model.test()
    predicted_noise = model.output.to(dtype=dtype)
    predicted_noise_float = predicted_noise.float().flatten(1)
    terminal_noise_float = terminal_noise.float().flatten(1)
    noise_difference = predicted_noise_float - terminal_noise_float
    mean_l1 = noise_difference.abs().mean().item()
    mean_l2 = noise_difference.square().mean().item()
    original_noise_min = terminal_noise_float.amin(dim=1, keepdim=True)
    original_noise_range = (
        terminal_noise_float.amax(dim=1, keepdim=True) - original_noise_min
    ).clamp_min(1e-12)
    normalized_difference = noise_difference / original_noise_range
    normalized_mean_l2 = normalized_difference.square().mean(dim=1)
    psnr = (-10 * torch.log10(normalized_mean_l2.clamp_min(1e-12))).mean().item()
    cosine_similarity = torch.nn.functional.cosine_similarity(
        predicted_noise_float, terminal_noise_float, dim=1).mean().item()
    decoded_noise = decode_latent(pipe, terminal_noise)
    input_image = decode_latent(pipe, network_input)
    regenerated_latent = generate_latent_with_custom_unipc(
        sampler, predicted_noise, cond_embeds, uncond_embeds,
        args.start_free_u_step)
    regenerated_image = decode_latent(pipe, regenerated_latent)
    save_noise_image(terminal_noise, output_dir / 'original_noise_0_255.png')
    save_noise_image(network_input, output_dir / 'input_latent_0_255.png')
    save_noise_image(model.pred_xstart, output_dir / 'predicted_xstart_0_255.png')
    save_noise_image(model.pred_xt, output_dir / 'predicted_xt_0_255.png')
    save_noise_image(predicted_noise, output_dir / 'predicted_noise_0_255.png')
    save_image(decoded_noise[0], output_dir / 'original_noise_vae_decoded.png')
    save_image(input_image[0], output_dir / 'input_noise_unipc.png')
    save_image(regenerated_image[0], output_dir / 'predicted_noise_unipc.png')
    torch.save(network_input.detach().cpu(), output_dir / 'input_latent.pt')
    torch.save(model.pred_xstart.detach().cpu(), output_dir / 'predicted_xstart.pt')
    torch.save(model.pred_xt.detach().cpu(), output_dir / 'predicted_xt.pt')
    torch.save(predicted_noise.detach().cpu(), output_dir / 'predicted_noise.pt')
    torch.save(terminal_noise.detach().cpu(), output_dir / 'original_noise.pt')

    metrics = {
        'prompt': args.prompt,
        'seed': args.seed,
        'height': args.height,
        'width': args.width,
        'mean_l1_predicted_vs_original_noise': mean_l1,
        'mean_l2_predicted_vs_original_noise': mean_l2,
        'psnr_predicted_vs_original_noise_db': psnr,
        'cosine_similarity_predicted_vs_original_noise': cosine_similarity,
        'custom_unipc_steps': args.unipc_steps,
        'start_free_u_step': args.start_free_u_step,
        'guidance_scale': args.guidance_scale,
        'xstart_checkpoint': opt['path']['pretrain_network_xstart'],
        'xt_checkpoint': opt['path']['pretrain_network_xt'],
        'xstart_prompt_checkpoint': opt['path'].get(
            'pretrain_xstart_prompt_tokens'),
        'sd_model_id': opt['sd_model_id'],
    }
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)
    print(f'Mean L1 distance between predicted and original noise: {mean_l1:.8f}')
    print(f'Mean L2 distance between predicted and original noise: {mean_l2:.8f}')
    print(f'PSNR between predicted and original noise: {psnr:.4f} dB')
    print(f'Cosine similarity between predicted and original noise: {cosine_similarity:.8f}')
    print(f'Saved the 0-255 original noise image to {output_dir / "original_noise_0_255.png"}')
    print(f'Saved the 0-255 input latent image to {output_dir / "input_latent_0_255.png"}')
    print(f'Saved the 0-255 predicted x-start image to {output_dir / "predicted_xstart_0_255.png"}')
    print(f'Saved the 0-255 predicted x-t image to {output_dir / "predicted_xt_0_255.png"}')
    print(f'Saved the 0-255 predicted noise image to {output_dir / "predicted_noise_0_255.png"}')
    print(f'Saved the direct VAE decode to {output_dir / "original_noise_vae_decoded.png"}')
    print(f'Saved the custom-UniPC input image to {output_dir / "input_noise_unipc.png"}')
    print(f'Saved the regenerated image to {output_dir / "predicted_noise_unipc.png"}')


if __name__ == '__main__':
    main()
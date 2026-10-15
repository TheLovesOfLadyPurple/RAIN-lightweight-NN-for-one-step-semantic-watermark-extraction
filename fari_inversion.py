"""FARI-compatible LoRA adapters and one-step terminal-noise inversion."""

from contextlib import contextmanager
from enum import Enum, auto

import torch
from torch import nn


class FARIMode(Enum):
    GENERATION = auto()
    INVERSION = auto()


class FARILinear(nn.Module):
    """Keep base generation unchanged and enable the LoRA residual for inversion."""

    def __init__(self, linear, rank):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.down = nn.Linear(linear.in_features, rank, bias=False)
        self.up = nn.Linear(rank, linear.out_features, bias=False)
        self.down.to(device=linear.weight.device, dtype=linear.weight.dtype)
        self.up.to(device=linear.weight.device, dtype=linear.weight.dtype)
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        self.mode = FARIMode.GENERATION

    def forward(self, inputs):
        output = self.linear(inputs)
        if self.mode is FARIMode.INVERSION:
            output = output + self.up(self.down(inputs))
        return output


def _replace_attention_linears(module, rank):
    replacements = []
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            replacements.append((name, FARILinear(child, rank)))
        else:
            _replace_attention_linears(child, rank)
    for name, replacement in replacements:
        setattr(module, name, replacement)


def inject_fari_adapters(unet, rank):
    """Inject LoRA adapters into Diffusers attention and feed-forward blocks.

    FARI uses the base UNet during image generation and activates these adapters
    only for the one-step inversion pass.
    """
    target_names = {'Attention', 'CrossAttention', 'GEGLU'}
    targets = [module for module in unet.modules()
               if module.__class__.__name__ in target_names]
    if not targets:
        raise RuntimeError('Could not find Diffusers attention modules for FARI adapters.')
    for module in targets:
        _replace_attention_linears(module, rank)

    for parameter in unet.parameters():
        parameter.requires_grad_(False)
    adapter_parameters = []
    for module in unet.modules():
        if isinstance(module, FARILinear):
            module.down.weight.requires_grad_(True)
            module.up.weight.requires_grad_(True)
            adapter_parameters.extend((module.down.weight, module.up.weight))
    return adapter_parameters


@contextmanager
def fari_mode(unet, mode):
    modules = [module for module in unet.modules() if isinstance(module, FARILinear)]
    previous_modes = [module.mode for module in modules]
    try:
        for module in modules:
            module.mode = mode
        yield
    finally:
        for module, previous_mode in zip(modules, previous_modes):
            module.mode = previous_mode


def adapter_state_dict(unet):
    return {name: value.detach().cpu() for name, value in unet.state_dict().items()
            if '.down.weight' in name or '.up.weight' in name}


def load_adapter_state_dict(unet, state_dict):
    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    missing = [name for name in missing if '.down.weight' not in name and '.up.weight' not in name]
    if unexpected:
        raise RuntimeError(f'Unexpected FARI checkpoint keys: {unexpected}')
    if missing:
        return missing
    return []


def load_official_fari_state_dict(unet, state_dict):
    """Load an official FARI ``fari_weights.pth`` checkpoint.

    Official FARI checkpoints use lora_diffusion names such as
    ``to_q.lora_layer.lora_down.weight``. The local adapter implementation
    stores the equivalent tensor as ``to_q.down.weight``.
    """
    converted = {}
    for name, tensor in state_dict.items():
        converted_name = name.replace(
            '.lora_layer.lora_down.weight', '.down.weight').replace(
                '.lora_layer.lora_up.weight', '.up.weight')
        if converted_name == name:
            raise RuntimeError(f'Unexpected official FARI key: {name}')
        converted[converted_name] = tensor

    adapter_keys = set(adapter_state_dict(unet))
    checkpoint_keys = set(converted)
    missing = sorted(adapter_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - adapter_keys)
    if missing or unexpected:
        raise RuntimeError(
            'Official FARI checkpoint does not match this U-Net adapter layout. '
            f'Missing {len(missing)} keys and found {len(unexpected)} unexpected keys.')
    for name, tensor in converted.items():
        expected_shape = adapter_state_dict(unet)[name].shape
        if tensor.shape != expected_shape:
            raise RuntimeError(
                f'FARI tensor shape mismatch for {name}: checkpoint {tuple(tensor.shape)}, '
                f'model {tuple(expected_shape)}.')
    unet.load_state_dict(converted, strict=False)


def one_step_inversion(pipe, image_latents, prompt_embeds):
    """Recover terminal noise from an image latent using FARI inversion mode."""
    model_input = pipe.scheduler.scale_model_input(image_latents, 0)
    with fari_mode(pipe.unet, FARIMode.INVERSION):
        noise_prediction = pipe.unet(
            model_input, 0, encoder_hidden_states=prompt_embeds,
            return_dict=False)[0]
    alpha_terminal = pipe.scheduler.alphas_cumprod[-1].to(image_latents.device)
    return (alpha_terminal.sqrt() * image_latents
            + (1 - alpha_terminal).sqrt() * noise_prediction)
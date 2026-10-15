"""Image-space perturbations used for reverse-distillation robustness training."""

from io import BytesIO
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


DEFAULT_DISTORTION_OPTIONS = {
    'jpeg_ratio': 25,
    'random_crop_ratio': 0.6,
    'random_drop_ratio': 0.8,
    'gaussian_blur_r': 4,
    'median_blur_k': 7,
    'gaussian_std': 0.05,
    'sp_prob': 0.05,
    'brightness_factor': 6,
    'resize_ratio': 0.25,
}


def tensor_to_pil(image):
    """Convert one Stable Diffusion image tensor in [-1, 1] to RGB PIL."""
    image = image.detach().float().cpu().clamp(-1, 1)
    pixels = ((image + 1) * 127.5).round().to(dtype=image.new_tensor(0).byte().dtype)
    return Image.fromarray(pixels.permute(1, 2, 0).numpy(), mode='RGB')


def pil_to_tensor(image):
    """Convert an RGB PIL image to a CHW float tensor in [-1, 1]."""
    pixels = torch_from_numpy(np.asarray(image.convert('RGB')).copy()).float()
    return pixels.permute(2, 0, 1).div(127.5).sub(1)


def torch_from_numpy(array):
    """Keep torch import lazy so image utilities remain lightweight."""
    import torch
    return torch.from_numpy(array)


def image_distortion(image, options=None, rng=None):
    """Apply one FARI-style image perturbation and return its name."""
    options = {**DEFAULT_DISTORTION_OPTIONS, **(options or {})}
    rng = rng or random
    choice = rng.randrange(10)
    name = (
        'none', 'jpeg', 'random.crop', 'random.drop', 'resize',
        'gaussian.blur', 'median.blur', 'gaussian.noise',
        'salt.and.pepper', 'brightness')[choice]
    image = image.convert('RGB')

    if choice == 1:
        jpeg_buffer = BytesIO()
        image.save(jpeg_buffer, format='JPEG', quality=options['jpeg_ratio'])
        jpeg_buffer.seek(0)
        return Image.open(jpeg_buffer).convert('RGB').copy(), name
    if choice == 2:
        array = np.asarray(image).copy()
        height, width = array.shape[:2]
        crop_width = max(1, int(width * options['random_crop_ratio']))
        crop_height = max(1, int(height * options['random_crop_ratio']))
        left = rng.randrange(width - crop_width + 1)
        top = rng.randrange(height - crop_height + 1)
        output = np.zeros_like(array)
        output[top:top + crop_height, left:left + crop_width] = (
            array[top:top + crop_height, left:left + crop_width])
        return Image.fromarray(output), name
    if choice == 3:
        array = np.asarray(image).copy()
        height, width = array.shape[:2]
        keep_width = max(1, int(width * options['random_drop_ratio']))
        keep_height = max(1, int(height * options['random_drop_ratio']))
        left = rng.randrange(width - keep_width + 1)
        top = rng.randrange(height - keep_height + 1)
        array[top:top + keep_height, left:left + keep_width] = 0
        return Image.fromarray(array), name
    if choice == 4:
        width, height = image.size
        resize_width = max(1, int(width * options['resize_ratio']))
        resize_height = max(1, int(height * options['resize_ratio']))
        return image.resize((resize_width, resize_height)).resize((width, height)), name
    if choice == 5:
        return image.filter(ImageFilter.GaussianBlur(options['gaussian_blur_r'])), name
    if choice == 6:
        return image.filter(ImageFilter.MedianFilter(options['median_blur_k'])), name
    if choice == 7:
        array = np.asarray(image).astype(np.float32)
        noise = np.random.default_rng(rng.randrange(2 ** 32)).normal(
            0, options['gaussian_std'] * 255, array.shape)
        return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8)), name
    if choice == 8:
        array = np.asarray(image).copy()
        random_values = np.random.default_rng(rng.randrange(2 ** 32)).random(
            array.shape[:2])
        array[random_values < options['sp_prob'] / 2] = 0
        array[random_values > 1 - options['sp_prob'] / 2] = 255
        return Image.fromarray(array), name
    if choice == 9:
        return ImageEnhance.Brightness(image).enhance(
            rng.uniform(1 / options['brightness_factor'],
                        options['brightness_factor'])), name
    return image, name

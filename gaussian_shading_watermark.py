"""Evaluator adapter for the official Gaussian Shading implementation."""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from watermark import Gaussian_Shading_chacha


@dataclass(frozen=True)
class WatermarkEvaluation:
    bit_accuracy: float
    detected: bool
    traceable: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            'bit_accuracy': self.bit_accuracy,
            'detected': self.detected,
            'traceable': self.traceable,
        }


class GaussianShadingWatermark:
    """Expose the official ChaCha20 Gaussian Shading API to evaluators."""

    def __init__(
            self,
            channel_copy: int = 1,
            hw_copy: int = 8,
            fpr: float = 1e-6,
            user_number: int = 1_000_000):
        self._official = Gaussian_Shading_chacha(
            channel_copy, hw_copy, fpr, user_number)
        self.detection_threshold = self._official.tau_onebit
        self.traceability_threshold = self._official.tau_bits

    def create_latent(
            self,
            device: torch.device,
            dtype: torch.dtype,
            generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if device.type != 'cuda':
            raise ValueError(
                'The official Gaussian Shading implementation requires CUDA.')
        return self._official.create_watermark_and_return_w().to(dtype=dtype)

    def evaluate(self, recovered_latent: torch.Tensor) -> WatermarkEvaluation:
        if tuple(recovered_latent.shape) != (1, 4, 64, 64):
            raise ValueError(
                'Expected recovered latent shape (1, 4, 64, 64), '
                f'got {tuple(recovered_latent.shape)}.')
        bit_accuracy = self._official.eval_watermark(recovered_latent)
        return WatermarkEvaluation(
            bit_accuracy=bit_accuracy,
            detected=bit_accuracy >= self.detection_threshold,
            traceable=bit_accuracy >= self.traceability_threshold)

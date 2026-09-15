# RAIN: a lightweight network for the watermark extraction

RAIN is a lightweight, one-step extractor for Gaussian-Shading
watermarks in Stable Diffusion images. It replaces iterative diffusion
inversion and diffusion-scale extraction networks with two compact,
prompt-free NAFNet branches.

Paper: [RAIN: Region-Aware Inversion Network for Semantic Watermark Extraction](https://arxiv.org/abs/2609.14856)

## Method

Given an image, the Stable Diffusion VAE first maps it to a latent. Two NAFNet
branches then process that latent independently:

1. The anchor branch estimates the clean latent representation.
2. The noise branch estimates an intermediate noisy latent and its timestep.
3. The two predictions are combined analytically with the diffusion schedule
   to recover the terminal noise used by Gaussian Shading.
4. The standard Gaussian-Shading decoder decrypts and votes on the recovered
   noise regions to detect and trace the watermark.

Because the branches are independent when the noise branch is a NAFNet, they
run concurrently on separate CUDA streams during inference. The extractor does
not require the generation prompt or repeated U-Net evaluations.

Training uses paired generated latents and terminal noise. Optional image-space
distortions expose the extractor to JPEG compression, cropping, masking,
resizing, blur, noise, and brightness changes before VAE re-encoding, improving
watermark recovery under common post-processing attacks.

The main training, clean evaluation, robustness evaluation, computational-cost,
and latency scripts are included in this repository together with their YAML
configurations under `options/`.

## Computational cost

Backbone cost of one latent-to-noise watermark extraction, measured with
`calflops`; lower is better. Image encoding and Gaussian-Shading decoding are
excluded for all methods.

| Method | GFLOPs ↓ | GMACs ↓ | Parameters (M) ↓ |
|:-------|---------:|--------:|-----------------:|
| FARI   | 682.675  | 341.079 | 868.469          |
| OSI    | 678.723  | 339.103 | 865.911          |
| **RAIN (ours)** | **14.394** | **7.148** | **15.468** |

## Environment setup

The evaluation scripts require an NVIDIA GPU. The following setup uses Python
3.11 and mutually compatible CUDA 12.8 builds of PyTorch and xFormers. A recent
NVIDIA driver that supports CUDA 12.8 is required; a separate CUDA Toolkit
installation is not needed for the PyTorch wheel.

```powershell
conda create -n simple-gs-inversion python=3.11 -y
conda activate simple-gs-inversion

python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 `
    torch==2.8.0 torchvision==0.23.0
python -m pip install --no-deps xformers==0.0.32.post2
python -m pip install -r requirements.txt
```

Verify that PyTorch and xFormers can load before running an evaluation:

```powershell
python -c "import torch, xformers.ops; print(torch.__version__); print(torch.cuda.is_available())"
```

The final line must print `True`. If PyTorch reports a version ending in
`+cpu`, reinstall PyTorch from the CUDA 12.8 index shown above.

## Check watermark extraction

Use the included `local_gaussian_shading_extraction_adv_eval_sdv21.py` script
to generate clean, watermarked Stable Diffusion 2.1 images and evaluate
watermark extraction.

To test all of the SOTA watermark extractor:

```powershell
python local_gaussian_shading_extraction_adv_eval_sdv21.py
```

To run the repository's extractor only:

```powershell
python local_gaussian_shading_extraction_adv_eval_sdv21.py `
    --skip-extractors fari osi `
    --prompts "a photo of a cat sitting on a wooden chair"
```

The extractor checkpoint paths are read from
`options/test/local_test_GS_reverse_distill_uet_adv_sdV21.yml`. Make sure those
paths point to the downloaded or trained `xstart` and `xt` checkpoints.

To compare Ours with FARI and OSI, run the script without
`--skip-extractors`:

```powershell
python local_gaussian_shading_extraction_adv_eval_sdv21.py `
    --prompts "a photo of a cat sitting on a wooden chair"
```

The comparison uses `fari_weights.pth` by default. Official OSI SD 2.1 weights
are downloaded automatically from `VIPL-GENUN/OSI` and then reused from the
Hugging Face cache. The first OSI run downloads approximately 3.6 GB.

Results are written to `outputs/prompted_gs_fari_sdv21/metrics.json`. Repeated
runs reuse generated images from `outputs/prompt_generation_cache_sdv21/`;
pass `--refresh-generation-cache` to regenerate them.

## Citation

If you find RAIN useful in your research, please cite our paper:

```bibtex
@article{li2026rain,
    title={RAIN: Region-Aware Inversion Network for Semantic Watermark Extraction},
    author={Li, Zilai},
    journal={arXiv preprint arXiv:2609.14856},
    year={2026}
}
```

## Acknowledgments

RAIN uses and adapts the [NAFNet](https://github.com/megvii-research/NAFNet)
architecture for watermark extraction. We thank the NAFNet authors and the
Megvii Research team for making their implementation publicly available.
NAFNet is released under the MIT License and builds on
[BasicSR](https://github.com/XPixelGroup/BasicSR), which is released under the
Apache License 2.0. The original copyright and attribution notices are retained
in the corresponding source files.

If you use the NAFNet-based components, please also cite:

```bibtex
@article{chen2022simple,
    title={Simple Baselines for Image Restoration},
    author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
    journal={arXiv preprint arXiv:2204.04676},
    year={2022}
}
```

## License

RAIN's original code is available under the
[Apache License 2.0](LICENSE), which permits use, copying, modification, and
redistribution subject to its terms. Third-party components remain under their
respective licenses and copyright notices; see [NOTICE](NOTICE) for details.
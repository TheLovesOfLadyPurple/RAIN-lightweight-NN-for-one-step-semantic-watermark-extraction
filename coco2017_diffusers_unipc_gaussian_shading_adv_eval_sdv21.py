"""Compare FARI, OSI, Ours, and DDIM inversion on SD 2.1 Gaussian-Shading image attacks."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from Crypto.Cipher import ChaCha20
from diffusers import DDIMInverseScheduler, StableDiffusionPipeline, UniPCMultistepScheduler
from tqdm import tqdm

from coco2017_diffusers_unipc_gaussian_shading_adv_eval import (
    ATTACK_NAMES,
    aggregate_attack_results,
    apply_attack,
    encode_image,
    pil_to_tensor,
    save_csv,
    tensor_to_pil,
)
from coco2017_diffusers_unipc_gaussian_shading_eval import (
    DEFAULT_CAPTIONS,
    generation_cache_path,
    load_coco_captions,
    load_or_generate_sample,
)
from data.options import parse
from fari_inversion import (
    inject_fari_adapters,
    load_official_fari_state_dict,
    one_step_inversion,
)
from gaussian_shading_watermark import GaussianShadingWatermark
from reverse_distill_gaussian_shading_eval import (
    build_model,
    latent_metrics,
    prediction_diagnostics,
    reverse_noise,
)
from reverse_distill_unipc_eval import convert_reverse_model_to_float32


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = 'sd2-community/stable-diffusion-2-1-base'
DEFAULT_OPTIONS = PROJECT_ROOT / 'options' / 'test' / 'cocoGSReverseDistillUNetAdvSDV21.yml'
EXTRACTOR_NAMES = ('fari', 'osi', 'ours', 'ddim')
DDIM_INVERSE_STEPS = 50
DDIM_INVERSE_GUIDANCE_SCALE = 1.0
DIVERSITY_FIELDS = ('key', 'nonce', 'message', 'encrypted_message')
FALSE_POSITIVE_FIELDS = (
    'trial_count', 'false_detection_count', 'false_detection_rate',
    'false_traceability_count', 'false_traceability_rate',
    'mean_bit_accuracy', 'std_bit_accuracy', 'max_bit_accuracy')


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
        description='Compare FARI, OSI, Ours, and DDIM inversion on SD 2.1 GS images under all attacks.')
    parser.add_argument(
        '--opt', default=str(DEFAULT_OPTIONS),
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
        '--skip-extractors', nargs='*', choices=EXTRACTOR_NAMES, default=(),
        help='Extractors to omit. For example: --skip-extractors fari ours evaluates OSI only.')
    parser.add_argument('--captions', default=str(DEFAULT_CAPTIONS))
    parser.add_argument(
        '--outdir', default=str(PROJECT_ROOT / 'outputs' / 'coco_gs_fari_sdv21'))
    parser.add_argument(
        '--clean-image-dir', default=None,
        help='Folder for the non-watermarked control images. Defaults to <outdir>/clean_images.')
    parser.add_argument(
        '--generation-cache-dir',
        default=str(PROJECT_ROOT / 'outputs' / 'coco_generation_cache_sdv21'),
        help='Separate SD 2.1 generation cache; never reuses the SD 1.5 cache.')
    parser.add_argument('--refresh-generation-cache', action='store_true')
    parser.add_argument('--num', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=15)
    parser.add_argument('--guidance-scale', type=float, default=5.5)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--seed', type=int, default=25)
    parser.add_argument('--caption-seed', type=int, default=8)
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


def check_ours_backbone(args):
    # build_model overwrites the option's sd_model_id with --sd-model-id, so read the file.
    option_model_id = parse(args.opt, is_train=False).get('sd_model_id')
    if option_model_id != args.sd_model_id:
        raise ValueError(
            'The Ours option and --sd-model-id must use the same SD 2.1 backbone. '
            f'Got {option_model_id!r} from {args.opt} and {args.sd_model_id!r}.')


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
def osi_one_step_inversion(model, attacked_image, prompt_embeds, device, dtype):
    image = pil_to_tensor(attacked_image, device).unsqueeze(0) * 2 - 1
    timestep = torch.full((image.shape[0],), 999, device=device, dtype=torch.long)
    return model(image.to(dtype=dtype), timestep, prompt_embeds)


@torch.no_grad()
def ddim_inverse_noise(pipe, latent, null_prompt_embedding, inverse_scheduler, guidance_scale):
    """Invert an image latent with DDIM at CFG 1.0, conditioned only on the null prompt."""
    if guidance_scale != 1.0:
        raise ValueError('DDIM inverse uses CFG 1.0 with the null prompt.')
    inverted_latent = latent.detach().to(dtype=torch.float32).clone()
    null_embeds = null_prompt_embedding.float()
    for timestep in inverse_scheduler.timesteps:
        model_input = inverse_scheduler.scale_model_input(inverted_latent, timestep)
        noise_prediction = pipe.unet(
            model_input, timestep, encoder_hidden_states=null_embeds, return_dict=False)[0]
        inverted_latent = inverse_scheduler.step(
            noise_prediction, timestep, inverted_latent).prev_sample
    return inverted_latent


def extraction_record(watermark, predicted_noise, target_noise):
    return {
        **watermark.evaluate(predicted_noise).to_dict(),
        'noise_metrics': latent_metrics(predicted_noise, target_noise),
        'prediction_diagnostics': prediction_diagnostics(predicted_noise, target_noise),
    }


def bytes_to_bits(data):
    return torch.from_numpy(np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(bool))


def watermark_bits(watermark):
    official = watermark._official
    spread_message = official.watermark.repeat(
        1, official.ch, official.hw, official.hw).flatten().bool().cpu()
    keystream = bytes_to_bits(ChaCha20.new(key=official.key, nonce=official.nonce).encrypt(
        bytes(official.latentlength // 8)))
    return {
        'key': bytes_to_bits(official.key),
        'nonce': bytes_to_bits(official.nonce),
        'message': official.watermark.flatten().bool().cpu(),
        'keystream': keystream,
        'encrypted_message': spread_message ^ keystream,
    }


def pairwise_hamming_summary(bits, device):
    bits = bits.to(device=device, dtype=torch.float32)
    count, bit_length = bits.shape
    summary = {
        'count': count,
        'bit_length': bit_length,
        'unique_count': torch.unique(bits, dim=0).shape[0],
    }
    if count < 2:
        return summary
    distances = (bits @ (1 - bits).T + (1 - bits) @ bits.T) / bit_length
    rows, cols = torch.triu_indices(count, count, 1, device=device)
    pairs = distances[rows, cols]
    return {
        **summary,
        'pair_count': pairs.numel(),
        'duplicate_pair_count': int((pairs == 0).sum()),
        'min_normalized_hamming': pairs.min().item(),
        'mean_normalized_hamming': pairs.mean().item(),
        'std_normalized_hamming': pairs.std(correction=0).item(),
        'max_normalized_hamming': pairs.max().item(),
    }


@torch.no_grad()
def cross_key_bit_accuracy(predicted_signs, keystreams, messages, reference_watermark, device):
    """Vectorized Gaussian_Shading_chacha.eval_watermark: rows are images, columns are keys."""
    official = reference_watermark._official
    ch, hw = official.ch, official.hw
    keystreams = keystreams.to(device)
    messages = messages.to(device)
    rows = []
    for signs in predicted_signs.to(device):
        decrypted = (signs[None] ^ keystreams).view(-1, ch, 4 // ch, hw, 64 // hw, hw, 64 // hw)
        recovered = decrypted.sum(dim=(1, 3, 5)).flatten(1) > official.threshold
        rows.append((recovered == messages).float().mean(dim=1))
    return torch.stack(rows).cpu()


def false_positive_summary(bit_accuracy, reference_watermark):
    bit_accuracy = bit_accuracy.flatten().double()
    trial_count = bit_accuracy.numel()
    false_detections = int((bit_accuracy >= reference_watermark.detection_threshold).sum())
    false_traces = int((bit_accuracy >= reference_watermark.traceability_threshold).sum())
    return {
        'trial_count': trial_count,
        'false_detection_count': false_detections,
        'false_detection_rate': false_detections / trial_count,
        'false_traceability_count': false_traces,
        'false_traceability_rate': false_traces / trial_count,
        'mean_bit_accuracy': bit_accuracy.mean().item(),
        'std_bit_accuracy': bit_accuracy.std(correction=0).item(),
        'max_bit_accuracy': bit_accuracy.max().item(),
    }


def save_false_positive_csv(false_positive, path):
    with open(path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['extractor', 'key_scope', *FALSE_POSITIVE_FIELDS])
        writer.writeheader()
        for extractor_name, scopes in false_positive.items():
            for key_scope, metrics in scopes.items():
                writer.writerow({'extractor': extractor_name, 'key_scope': key_scope, **metrics})


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
    if 'ours' in enabled_extractors:
        check_ours_backbone(args)

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

    # Generate every sample before loading any extractor: FARI freezes the pipe.unet
    # parameters, which changes PyTorch's matmul path and therefore the sampled images.
    cache_files = []
    used_generation_cache = []
    generation_bar = tqdm(captions, unit='sample', desc='SD 2.1 generation cache')
    for index, caption_record in enumerate(generation_bar):
        seed = args.seed + index
        cache_file = generation_cache_path(
            cache_dir, caption_record, seed, args, args.sd_model_id)
        *_, cache_hit = load_or_generate_sample(
            cache_file, caption_record['caption'], seed, args, pipe, device, dtype)
        cache_files.append(cache_file)
        used_generation_cache.append(cache_hit)
        generation_bar.set_postfix_str(f'cache={"hit" if cache_hit else "miss"}')
    args.refresh_generation_cache = False

    osi_model = osi_null_prompt_embedding = None
    if 'osi' in enabled_extractors:
        # OSI copies pipe.unet, so it must load before FARI injects adapters into it.
        osi_model, osi_null_prompt_embedding = load_osi_model(pipe, args, device, dtype)
    fari_null_prompt_embedding = None
    if 'fari' in enabled_extractors:
        fari_null_prompt_embedding = configure_fari(pipe, args, device)
    model = opt = None
    if 'ours' in enabled_extractors:
        model, opt = build_model(args)
        convert_reverse_model_to_float32(model)
    null_prompt_embedding = None
    if 'ours' in enabled_extractors:
        null_prompt_embedding, _ = pipe.encode_prompt('', device, 1, False)
        null_prompt_embedding = null_prompt_embedding.detach()
    ddim_inverse_scheduler = None
    extractors_before_ddim = tuple(name for name in enabled_extractors if name != 'ddim')

    def extract_noise(extractor_name, image, latent):
        if extractor_name == 'fari':
            return one_step_inversion(pipe, latent, fari_null_prompt_embedding)
        if extractor_name == 'osi':
            return osi_one_step_inversion(osi_model, image, osi_null_prompt_embedding, device, dtype)
        if extractor_name == 'ddim':
            return ddim_inverse_noise(
                pipe, latent, null_prompt_embedding,
                ddim_inverse_scheduler, DDIM_INVERSE_GUIDANCE_SCALE)
        if extractor_name == 'ours':
            return reverse_noise(model, latent, null_prompt_embedding)['predicted_noise']
        raise ValueError(f'Unknown extractor: {extractor_name}')

    clean_image_dir = (Path(args.clean_image_dir) if args.clean_image_dir
                       else output_dir / 'clean_images')
    clean_image_dir.mkdir(parents=True, exist_ok=True)
    reference_watermark = GaussianShadingWatermark(
        channel_copy=args.channel_copy, hw_copy=args.hw_copy,
        fpr=args.fpr, user_number=args.user_number)
    watermark_records = []
    watermark_bit_vectors = {name: [] for name in (*DIVERSITY_FIELDS, 'keystream')}
    clean_signs = {name: [] for name in enabled_extractors}
    results = []
    progress_bar = tqdm(captions, unit='sample', desc='COCO SD 2.1 GS evaluation')
    for index, caption_record in enumerate(progress_bar):
        prompt = caption_record['caption']
        seed = args.seed + index
        cache_file = cache_files[index]
        (watermark, watermarked_noise, _, generated_image,
         _, _, control_generated_image, _) = load_or_generate_sample(
             cache_file, prompt, seed, args, pipe, device, dtype, require_cache=True)
        bits = watermark_bits(watermark)
        for name, value in bits.items():
            watermark_bit_vectors[name].append(value)
        watermark_records.append({
            'index': index,
            'seed': seed,
            'image_id': caption_record['image_id'],
            'key_hex': watermark._official.key.hex(),
            'nonce_hex': watermark._official.nonce.hex(),
            'message_bits': ''.join('1' if bit else '0' for bit in bits['message'].tolist()),
        })

        base_image = tensor_to_pil(generated_image[0])
        attacks = {}
        for attack_index, attack_name in enumerate(ATTACK_NAMES):
            attacked_image = apply_attack(
                base_image, attack_name, args, seed + 10_007 * attack_index)
            attacked_latent = encode_image(pipe, attacked_image, dtype)
            attacks[attack_name] = {}
            for extractor_name in extractors_before_ddim:
                predicted_noise = extract_noise(
                    extractor_name, attacked_image, attacked_latent)
                attacks[attack_name][extractor_name] = extraction_record(
                    watermark, predicted_noise, watermarked_noise)

        clean_image = tensor_to_pil(control_generated_image[0])
        clean_image_path = (
            clean_image_dir / f'sample_{index:04d}_coco_{caption_record["image_id"]}.png')
        clean_image.save(clean_image_path)
        clean_latent = encode_image(pipe, clean_image, dtype)
        clean_results = {}
        for extractor_name in extractors_before_ddim:
            predicted_noise = extract_noise(
                extractor_name, clean_image, clean_latent)
            clean_results[extractor_name] = watermark.evaluate(predicted_noise).to_dict()
            clean_signs[extractor_name].append((predicted_noise > 0).flatten().cpu())

        results.append({
            'index': index,
            'seed': seed,
            **caption_record,
            'generation_cache_path': str(cache_file.resolve()),
            'used_generation_cache': used_generation_cache[index],
            'clean_image_path': str(clean_image_path.resolve()),
            'clean_image': clean_results,
            'attacks': attacks,
        })

    if 'ddim' in enabled_extractors:
        del pipe
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        pipe = StableDiffusionPipeline.from_pretrained(args.sd_model_id, torch_dtype=dtype)
        ddim_inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
        ddim_inverse_scheduler.set_timesteps(DDIM_INVERSE_STEPS, device=device)
        pipe.safety_checker = None
        pipe.to(device=device, dtype=dtype)
        pipe.set_progress_bar_config(disable=True)
        pipe.unet.eval()
        null_prompt_embedding, _ = pipe.encode_prompt('', device, 1, False)
        null_prompt_embedding = null_prompt_embedding.detach()
        ddim_bar = tqdm(captions, unit='sample', desc='SD 2.1 DDIM inverse')
        for index, caption_record in enumerate(ddim_bar):
            prompt = caption_record['caption']
            seed = args.seed + index
            cache_file = cache_files[index]
            (watermark, watermarked_noise, _, generated_image,
             _, _, control_generated_image, _) = load_or_generate_sample(
                 cache_file, prompt, seed, args, pipe, device, dtype, require_cache=True)
            base_image = tensor_to_pil(generated_image[0])
            for attack_index, attack_name in enumerate(ATTACK_NAMES):
                attacked_image = apply_attack(
                    base_image, attack_name, args, seed + 10_007 * attack_index)
                attacked_latent = encode_image(pipe, attacked_image, dtype)
                predicted_noise = extract_noise(
                    'ddim', attacked_image, attacked_latent)
                results[index]['attacks'][attack_name]['ddim'] = extraction_record(
                    watermark, predicted_noise, watermarked_noise)
            clean_image = tensor_to_pil(control_generated_image[0])
            clean_latent = encode_image(pipe, clean_image, dtype)
            predicted_noise = extract_noise('ddim', clean_image, clean_latent)
            results[index]['clean_image']['ddim'] = watermark.evaluate(predicted_noise).to_dict()
            clean_signs['ddim'].append((predicted_noise > 0).flatten().cpu())

    aggregates = {
        name: aggregate_attack_results(results, name) for name in enabled_extractors}
    stacked_bits = {name: torch.stack(values) for name, values in watermark_bit_vectors.items()}
    diversity = {
        name: pairwise_hamming_summary(stacked_bits[name], device) for name in DIVERSITY_FIELDS}
    false_positive = {}
    for extractor_name in enabled_extractors:
        cross_key = cross_key_bit_accuracy(
            torch.stack(clean_signs[extractor_name]), stacked_bits['keystream'],
            stacked_bits['message'], reference_watermark, device)
        paired = torch.tensor(
            [sample['clean_image'][extractor_name]['bit_accuracy'] for sample in results])
        if not torch.equal(cross_key.diagonal(), paired):
            raise RuntimeError(
                f'Vectorized {extractor_name} decoding disagrees with the official '
                'paired-key evaluation.')
        false_positive[extractor_name] = {
            'paired_key': false_positive_summary(paired, reference_watermark),
            'all_keys': false_positive_summary(cross_key, reference_watermark),
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
            'sample_count': args.num,
            'steps': args.steps,
            'guidance_scale': args.guidance_scale,
            'ddim_inverse_steps': DDIM_INVERSE_STEPS if 'ddim' in enabled_extractors else None,
            'ddim_guidance_scale': (
                DDIM_INVERSE_GUIDANCE_SCALE if 'ddim' in enabled_extractors else None),
            'generation_cache_dir': str(cache_dir.resolve()),
            'clean_image_dir': str(clean_image_dir.resolve()),
            'nominal_fpr': args.fpr,
            'user_number': args.user_number,
            'detection_threshold': reference_watermark.detection_threshold,
            'traceability_threshold': reference_watermark.traceability_threshold,
            'attack_order': ATTACK_NAMES,
            'protocol_note': (
                'FARI and OSI are evaluated with their native SD 2.1-base components. '
                'Generation cache entries are separated from the SD 1.5 evaluation cache. '
                'All samples are generated with the unmodified pipeline before any '
                'extractor is loaded; evaluation only reads the cache. '
                'DDIM inversion is a 50-step DDIM inverse at CFG 1.0. It runs after '
                'the other extractors, on a freshly loaded pipeline so it does not '
                'reuse the FARI LoRA UNet. Every extractor is conditioned on the null prompt: '
                'FARI, OSI, and DDIM receive the empty-string text embedding, and '
                'Ours does not receive a caption embedding.'),
            'false_positive_note': (
                'Clean images are the cached control generations: same COCO caption, '
                'standard Gaussian noise seeded with seed + 1_000_000, no watermark and '
                'no attack. paired_key tests each clean image against the key and message '
                'of the watermarked sample with the same caption; all_keys tests it '
                'against every sample key and message.'),
        },
        'aggregate': aggregates,
        'watermark_diversity': diversity,
        'false_positive': false_positive,
        'samples': results,
    }
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    with open(output_dir / 'watermark_diversity.json', 'w', encoding='utf-8') as file:
        json.dump({
            'message_shape': [4 // args.channel_copy, 64 // args.hw_copy, 64 // args.hw_copy],
            'summary': diversity,
            'samples': watermark_records,
        }, file, indent=2)
    save_csv(aggregates, output_dir / 'aggregate.csv')
    save_false_positive_csv(false_positive, output_dir / 'false_positive.csv')
    print(json.dumps(report['aggregate'], indent=2))
    print(json.dumps({'watermark_diversity': diversity, 'false_positive': false_positive}, indent=2))
    print(f'Saved SD 2.1 robustness results to {output_dir}')


if __name__ == '__main__':
    main()
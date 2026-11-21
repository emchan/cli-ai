#!/usr/bin/env python3
"""FramePack command-line interface.

Usage:
    python framepack-cli.py "a dancing man" --input image.png
    python framepack-cli.py --prompt "a dancing man" -i image.png

Generates a video with FramePack and saves it to outputs/<prompt>-<seed>.mp4.
"""
import argparse
import datetime
import os
import re
import sys
import traceback

import math
import shutil
import subprocess
import glob

# Script/output root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Self-bootstrapping virtual environment.
# FramePack needs a large set of Python packages (torch, diffusers, ...). To
# keep the script self-contained we make sure a dedicated venv exists and
# re-exec ourselves with it when the current interpreter lacks the deps.
# ---------------------------------------------------------------------------
FRAMEPACK_VENV_DIR = os.environ.get("FRAMEPACK_VENV", os.path.expanduser("~/cli-ai/framepack-cli"))


def _find_fp311_site_packages():
    candidates = [
        "/home/ubuntu/miniconda3/envs/fp311/lib/python3.11/site-packages",
    ]
    for pattern in (
        "/opt/conda/envs/fp311/lib/python*/site-packages",
        "/root/miniconda3/envs/fp311/lib/python*/site-packages",
        "/home/*/.conda/envs/fp311/lib/python*/site-packages",
        "/home/*/*conda*/envs/fp311/lib/python*/site-packages",
        "/usr/local/miniconda3/envs/fp311/lib/python*/site-packages",
    ):
        candidates.extend(glob.glob(pattern))
    for cand in candidates:
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "torch", "__init__.py")):
            return cand
    return None


def _base_python():
    for name in ("python3.11", "python3", "python"):
        path = shutil.which(name)
        if path:
            return path
    return sys.executable


def _venv_python():
    return os.path.join(FRAMEPACK_VENV_DIR, "bin", "python")


def _venv_site_packages():
    matches = glob.glob(os.path.join(FRAMEPACK_VENV_DIR, "lib", "python*", "site-packages"))
    if not matches:
        matches = glob.glob(os.path.join(FRAMEPACK_VENV_DIR, "Lib", "site-packages"))
    if not matches:
        raise RuntimeError("could not locate venv site-packages")
    return matches[0]


def _venv_ok(python):
    try:
        subprocess.run(
            [python, "-c", "import torch, diffusers, transformers"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=300,
        )
        return True
    except Exception:
        return False


def _create_venv():
    if os.path.exists(FRAMEPACK_VENV_DIR):
        shutil.rmtree(FRAMEPACK_VENV_DIR)
    parent = os.path.dirname(FRAMEPACK_VENV_DIR)
    if parent:
        os.makedirs(parent, exist_ok=True)
    subprocess.run([_base_python(), "-m", "venv", FRAMEPACK_VENV_DIR], check=True)


def _install_requirements(python):
    req_candidates = [
        os.path.join(SCRIPT_DIR, "requirements.txt"),
        "/tmp/FramePack/requirements.txt",
        os.path.join(os.path.expanduser("~"), ".cache", "framepack-src", "FramePack", "requirements.txt"),
    ]
    for req in req_candidates:
        if os.path.exists(req):
            subprocess.run(
                [python, "-m", "pip", "install", "-r", req,
                 "--extra-index-url", "https://download.pytorch.org/whl/cu126"],
                check=False,
            )
            return


def ensure_framepack_venv():
    python = _venv_python()
    if not (os.path.exists(python) and _venv_ok(python)):
        _create_venv()
        python = _venv_python()
        site = _venv_site_packages()
        fp_site = _find_fp311_site_packages()
        if fp_site is not None:
            with open(os.path.join(site, "fp311.pth"), "w") as f:
                f.write(fp_site + "\n")
        if not _venv_ok(python):
            _install_requirements(python)
    return python


# Bootstrap when the current interpreter does not already provide the deps.
if os.environ.get("FRAMEPACK_CLI_SKIP_VENV") != "1":
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        import transformers  # noqa: F401

        _framepack_deps_present = True
    except Exception:
        _framepack_deps_present = False
    if not _framepack_deps_present:
        _framepack_venv_python = ensure_framepack_venv()
        if os.path.abspath(sys.executable) != os.path.abspath(_framepack_venv_python):
            os.environ.setdefault("FRAMEPACK_CLI_BOOTSTRAPPED", "1")
            os.execv(_framepack_venv_python, [_framepack_venv_python] + sys.argv)

# Script/output root.
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# FramePack source (diffusers_helper, ...) is not vendored; fetch on demand.
SOURCE_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "framepack-src")
SOURCE_REPO = "https://github.com/lllyasviel/FramePack"


def ensure_framepack_source():
    candidates = [ROOT, "/tmp/FramePack", os.path.join(SOURCE_CACHE, "FramePack")]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "diffusers_helper")):
            return cand
    dest = os.path.join(SOURCE_CACHE, "FramePack")
    os.makedirs(SOURCE_CACHE, exist_ok=True)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    print("Downloading FramePack source dependencies ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", SOURCE_REPO, dest],
        check=True,
    )
    return dest


SOURCE_ROOT = ensure_framepack_source()
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

# Store Hugging Face downloads under ~/cli-ai/hf_downloads (home folder).
HF_DOWNLOADS = os.path.expanduser("~/cli-ai/hf_downloads")
os.makedirs(HF_DOWNLOADS, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_DOWNLOADS)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_DOWNLOADS, "hub"))

import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket


def sanitize_filename(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]", "", text)
    if not text:
        text = "prompt"
    return text[:80]


outputs_folder = os.path.join(ROOT, "outputs")
os.makedirs(outputs_folder, exist_ok=True)


def load_models():
    global free_mem_gb, high_vram
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae
    global feature_extractor, image_encoder, transformer

    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60

    print(f"Free VRAM {free_mem_gb} GB")
    print(f"High-VRAM Mode: {high_vram}")

    text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="text_encoder", torch_dtype=torch.float16).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="text_encoder_2", torch_dtype=torch.float16).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer_2")
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="vae", torch_dtype=torch.float16).cpu()

    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder="feature_extractor")
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16).cpu()

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained("lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16).cpu()

    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()

    if not high_vram:
        vae.enable_slicing()
        vae.enable_tiling()

    transformer.high_quality_fp32_output_for_inference = True
    print("transformer.high_quality_fp32_output_for_inference = True")

    transformer.to(dtype=torch.bfloat16)
    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)
    text_encoder.to(dtype=torch.float16)
    text_encoder_2.to(dtype=torch.float16)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    if not high_vram:
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
    else:
        text_encoder.to(gpu)
        text_encoder_2.to(gpu)
        image_encoder.to(gpu)
        vae.to(gpu)
        transformer.to(gpu)


def run_generation(
    input_image,
    prompt,
    output_path,
    n_prompt="",
    seed=31337,
    total_second_length=5,
    latent_window_size=9,
    steps=25,
    cfg=1.0,
    gs=10.0,
    rs=0.0,
    gpu_memory_preservation=6.0,
    use_teacache=True,
    mp4_crf=16,
):
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    try:
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

        # Text encoding
        print("Text encoding ...")
        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        print("Image processing ...")
        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        print("VAE encoding ...")
        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        print("CLIP Vision encoding ...")
        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        print("Start sampling ...")
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            print(f"latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}")

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                current_step = d["i"] + 1
                percentage = int(100.0 * current_step / steps)
                print(f"\rSampling {current_step}/{steps} ({percentage}%)", end="", flush=True)
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler="unipc",
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )
            print()

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            save_bcthw_as_mp4(history_pixels, output_path, fps=30, crf=mp4_crf)
            print(f"Saved {output_path} (latent frames: {total_generated_latent_frames})")

            if is_last_section:
                break
    except Exception:
        traceback.print_exc()
        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
    finally:
        if not high_vram:
            try:
                unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
            except Exception:
                pass

    return output_path


def main():
    parser = argparse.ArgumentParser(description="FramePack CLI")
    parser.add_argument("prompt", nargs="?", help="Text prompt to generate a video from.")
    parser.add_argument("--prompt", dest="prompt_opt", default=None, help="Text prompt (alternative to positional).")
    parser.add_argument("--input", "-i", required=True, help="Input image path.")
    parser.add_argument("--negative", default="", help="Negative prompt.")
    parser.add_argument("--output", default=None, help="Explicit output file path.")
    parser.add_argument("--seed", type=int, default=31337, help="Random seed.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Total video length in seconds.")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps.")
    parser.add_argument("--latent-window-size", type=int, default=9, help="Latent window size.")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale.")
    parser.add_argument("--gs", type=float, default=10.0, help="Distilled CFG scale.")
    parser.add_argument("--rs", type=float, default=0.0, help="CFG re-scale.")
    parser.add_argument("--gpu-memory-preservation", type=float, default=6.0, help="GPU memory preservation (GB).")
    parser.add_argument("--teacache", action="store_true", default=True, help="Use TeaCache.")
    parser.add_argument("--no-teacache", dest="teacache", action="store_false", help="Disable TeaCache.")
    parser.add_argument("--crf", type=int, default=16, help="MP4 CRF (lower = better quality).")
    args = parser.parse_args()

    prompt = args.prompt_opt if args.prompt_opt is not None else args.prompt
    if not prompt:
        parser.error("a prompt is required")

    if not os.path.exists(args.input):
        parser.error(f"input image not found: {args.input}")

    input_image = np.array(Image.open(args.input).convert("RGB"))

    load_models()

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(outputs_folder, f"{sanitize_filename(prompt)}-{args.seed}.mp4")

    # User requested outputs/<prompt>-<datestamp>.mp4 relative to ROOT; ensure outputs dir.
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Output: {output_path}")
    run_generation(
        input_image=input_image,
        prompt=prompt,
        output_path=output_path,
        n_prompt=args.negative,
        seed=args.seed,
        total_second_length=args.seconds,
        latent_window_size=args.latent_window_size,
        steps=args.steps,
        cfg=args.cfg,
        gs=args.gs,
        rs=args.rs,
        gpu_memory_preservation=args.gpu_memory_preservation,
        use_teacache=args.teacache,
        mp4_crf=args.crf,
    )
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()

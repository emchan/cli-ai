#!/usr/bin/env python3
"""FramePack command-line interface.

Usage:
    python framepack-cli.py "a dancing man" --input image.png
    python framepack-cli.py --prompt "a dancing man" -i image.png
    python framepack-cli.py "a dancing man"   # uses an image from the current folder
    python framepack-cli.py image.png "big waves crashing"   # image may come first

Relative image paths are resolved against the folder you run the command from,
not the folder containing this script. If --input is omitted, the most recently
modified image in that folder is used.

Generates a video with FramePack and saves it to the current folder as <prompt>-<seed>.mp4.
"""
import argparse
import datetime
import os
import re
import sys
import traceback
from contextlib import contextmanager

import math
import shutil
import subprocess
import glob

# Script/output root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder the CLI was launched from. The script chdir()s to ROOT below, so keep
# this to resolve user-supplied relative image paths against the current folder.
INVOCATION_DIR = os.getcwd()

# ROCm / WSL2 environment must be set before any torch import/subprocess.
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")


def is_tpu_backend():
    """Whether this interpreter sees PyTorch's native `tpu` accelerator.

    CLI_AI_BACKEND=tpu/native-tpu forces the TPU path; cuda/rocm/cpu forces the
    original CUDA path.
    """
    backend = os.environ.get("CLI_AI_BACKEND", "").lower()
    if backend in ("tpu", "native-tpu", "torch-tpu"):
        return True
    if backend in ("cuda", "rocm", "cpu", "nvidia"):
        return False
    try:
        import torch
        if hasattr(torch, "accelerator") and torch.accelerator.is_available():
            accelerator = torch.accelerator.current_accelerator()
            return accelerator is not None and accelerator.type == "tpu"
    except Exception:
        pass
    return False


TPU_AVAILABLE = is_tpu_backend()
if TPU_AVAILABLE:
    # Use the pre-installed torch_tpu stack instead of building a CUDA venv.
    os.environ["FRAMEPACK_CLI_SKIP_VENV"] = "1"

# Set to torch.bfloat16 on TPU once torch has been imported below.
INFERENCE_DTYPE = None

# ---------------------------------------------------------------------------
# Self-bootstrapping environment (conda/Miniforge preferred; venv fallback).
# ---------------------------------------------------------------------------
FRAMEPACK_VENV_DIR = os.environ.get("FRAMEPACK_VENV", os.path.expanduser("~/cli-ai/framepack-cli-uv"))


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


def _venv_python(prefix):
    return os.path.join(prefix, "bin", "python")


def _venv_site_packages(prefix):
    matches = glob.glob(os.path.join(prefix, "lib", "python*", "site-packages"))
    if not matches:
        matches = glob.glob(os.path.join(prefix, "Lib", "site-packages"))
    if not matches:
        raise RuntimeError("could not locate venv site-packages")
    return matches[0]


def _venv_ok(python):
    try:
        subprocess.run(
            [python, "-c", "import torch, diffusers, transformers; import sys; sys.exit(0 if getattr(torch.version, 'hip', None) else 1)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=300,
        )
        return True
    except Exception:
        return False


def _install_requirements(python):
    req_candidates = [
        os.path.join(SCRIPT_DIR, "requirements_rocm.txt"),
        os.path.join(SCRIPT_DIR, "requirements.txt"),
    ]
    for req in req_candidates:
        if os.path.exists(req):
            try:
                subprocess.run(
                    [python, "-m", "pip", "install", "-r", req,
                     "--extra-index-url", "https://download.pytorch.org/whl/rocm7.2"],
                    check=True,
                )
                # Ensure a ROCm build of torch/torchvision is present.
                try:
                    subprocess.run(
                        [python, "-c", "import torch,sys; sys.exit(0 if getattr(torch.version,'hip',None) else 1)"],
                        check=True,
                    )
                except Exception:
                    subprocess.run(
                        [python, "-m", "pip", "install", "--force-reinstall", "--no-deps",
                         "torch", "torchvision",
                         "--index-url", "https://download.pytorch.org/whl/rocm7.2"],
                        check=True,
                    )
            except Exception as e:
                print(f"pip install failed: {e}", flush=True)
            return


def _ensure_miniforge():
    conda_bin = shutil.which("conda")
    mamba_bin = shutil.which("mamba")
    micromamba_bin = shutil.which("micromamba")
    forge_bin = mamba_bin or conda_bin or micromamba_bin
    if forge_bin:
        return forge_bin, micromamba_bin
    miniforge_prefix = os.path.expanduser("~/miniforge3")
    miniforge_bin = os.path.join(miniforge_prefix, "bin", "conda")
    if os.path.exists(miniforge_bin):
        return miniforge_bin, None
    print("Miniforge/conda not found; downloading and installing Miniforge...", flush=True)
    try:
        installer = os.path.join("/tmp", "Miniforge3-Linux-x86_64.sh")
        if not os.path.exists(installer):
            subprocess.run(
                ["bash", "-lc",
                 "curl -L -o /tmp/Miniforge3-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"],
                check=True,
            )
        subprocess.run(["bash", installer, "-b", "-p", miniforge_prefix], check=True)
        if os.path.exists(miniforge_bin):
            return miniforge_bin, None
    except Exception as e:
        print(f"Miniforge install failed: {e}", flush=True)
    return None, None


def ensure_framepack_venv():
    forge_bin, micromamba_bin = _ensure_miniforge()
    if forge_bin:
        parent = os.path.dirname(FRAMEPACK_VENV_DIR)
        if parent:
            os.makedirs(parent, exist_ok=True)
        py = _venv_python(FRAMEPACK_VENV_DIR)
        if not (os.path.exists(py) and _venv_ok(py)):
            if os.path.exists(FRAMEPACK_VENV_DIR):
                shutil.rmtree(FRAMEPACK_VENV_DIR)
            print(f"Creating conda env for FramePack at {FRAMEPACK_VENV_DIR} ...", flush=True)
            try:
                if micromamba_bin and "micromamba" in (micromamba_bin or ""):
                    subprocess.run([micromamba_bin, "create", "-y", "-p", FRAMEPACK_VENV_DIR, "python=3.11", "pip"], check=True)
                else:
                    subprocess.run([forge_bin, "create", "-y", "-p", FRAMEPACK_VENV_DIR, "python=3.11", "pip"], check=True)
            except Exception as e:
                print(f"conda env creation failed: {e}", flush=True)
        py = _venv_python(FRAMEPACK_VENV_DIR)
        if os.path.exists(py):
            if not _venv_ok(py):
                _install_requirements(py)
            return py
    # Fallback to venv
    python = _venv_python(FRAMEPACK_VENV_DIR)
    if not (os.path.exists(python) and _venv_ok(python)):
        if os.path.exists(FRAMEPACK_VENV_DIR):
            shutil.rmtree(FRAMEPACK_VENV_DIR)
        parent = os.path.dirname(FRAMEPACK_VENV_DIR)
        if parent:
            os.makedirs(parent, exist_ok=True)
        subprocess.run([_base_python(), "-m", "venv", FRAMEPACK_VENV_DIR], check=True)
        python = _venv_python(FRAMEPACK_VENV_DIR)
        try:
            site = _venv_site_packages(FRAMEPACK_VENV_DIR)
            fp_site = _find_fp311_site_packages()
            if fp_site is not None:
                with open(os.path.join(site, "fp311.pth"), "w") as f:
                    f.write(fp_site + "\n")
        except Exception:
            pass
        if not _venv_ok(python):
            _install_requirements(python)
    return python


# Bootstrap when the current interpreter does not already provide the deps.
if os.environ.get("FRAMEPACK_CLI_SKIP_VENV") != "1":
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
        if not getattr(torch.version, "hip", None):
            raise RuntimeError("not a ROCm build")

        _framepack_deps_present = True
    except Exception:
        _framepack_deps_present = False
    if not _framepack_deps_present:
        _framepack_venv_python = ensure_framepack_venv()
        if os.path.abspath(sys.executable) != os.path.abspath(_framepack_venv_python):
            os.environ.setdefault("FRAMEPACK_CLI_BOOTSTRAPPED", "1")
            os.execv(_framepack_venv_python, [_framepack_venv_python] + sys.argv)


def ensure_tpu_requirements():
    """Install the TPU-safe FramePack extras into the torch_tpu interpreter."""
    import importlib

    modules = (
        "torch", "torchvision", "diffusers", "transformers", "accelerate",
        "safetensors", "einops", "sentencepiece", "scipy", "requests",
        "torchsde", "PIL", "cv2", "av",
    )
    missing = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    if not missing:
        return
    print(f"Installing TPU-safe FramePack dependencies (missing: {', '.join(missing)}) ...", flush=True)
    req = os.path.join(SCRIPT_DIR, "requirements_tpu.txt")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "-r", req],
            check=True,
        )
    except Exception as e:
        print(f"Warning: pip install failed: {e}", flush=True)


if TPU_AVAILABLE and os.environ.get("FRAMEPACK_TPU_SKIP_PIP_INSTALL") != "1":
    ensure_tpu_requirements()

# Script/output root.
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# FramePack source (diffusers_helper, ...) is not vendored; fetch on demand.
# The TPU patch gets a separate cache dir so a CUDA checkout is never patched.
SOURCE_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "framepack-src")
SOURCE_REPO = "https://github.com/lllyasviel/FramePack"
SOURCE_DIR = "FramePack-TPU" if TPU_AVAILABLE else "FramePack"
TPU_PATCH = os.path.join(SCRIPT_DIR, "framepack_tpu_patch.diff")


def ensure_framepack_source():
    candidates = [os.path.join(SOURCE_CACHE, SOURCE_DIR)]
    if not TPU_AVAILABLE:
        candidates = [ROOT, "/tmp/FramePack"] + candidates
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "diffusers_helper")):
            return cand
    dest = os.path.join(SOURCE_CACHE, SOURCE_DIR)
    os.makedirs(SOURCE_CACHE, exist_ok=True)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    print("Downloading FramePack source dependencies ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", SOURCE_REPO, dest],
        check=True,
    )
    return dest


def apply_tpu_patch(source_root):
    """Apply the local TPU backend patch to diffusers_helper (idempotent)."""
    if not os.path.isfile(TPU_PATCH):
        raise RuntimeError(f"FramePack TPU patch not found: {TPU_PATCH}")
    marker = os.path.join(source_root, "diffusers_helper", "memory.py")
    if os.path.isfile(marker):
        with open(marker, encoding="utf-8", errors="replace") as handle:
            if "tpu_available = _detect_tpu()" in handle.read():
                return  # already patched
    check = subprocess.run(
        ["git", "apply", "--check", TPU_PATCH], cwd=source_root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "FramePack TPU patch does not apply to the cached source.\n"
            + check.stderr.strip()
        )
    print("Applying FramePack TPU patch ...", flush=True)
    subprocess.run(["git", "apply", TPU_PATCH], cwd=source_root, check=True)


SOURCE_ROOT = ensure_framepack_source()
if TPU_AVAILABLE:
    apply_tpu_patch(SOURCE_ROOT)
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

# Store Hugging Face downloads under ~/cli-ai/hf_downloads (home folder).
HF_DOWNLOADS = os.path.expanduser("~/cli-ai/hf_downloads")
os.makedirs(HF_DOWNLOADS, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_DOWNLOADS)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_DOWNLOADS, "hub"))

import numpy as np
import torch
if TPU_AVAILABLE:
    INFERENCE_DTYPE = torch.bfloat16
else:
    INFERENCE_DTYPE = None  # keep each model's current dtype on CUDA
import torchvision.io

# Optional torchao quantization.
try:
    from torchao.quantization import quantize_ as _torchao_quantize
    from torchao.quantization import Float8WeightOnlyConfig as _Float8Cfg
    from torchao.quantization import Int8WeightOnlyConfig as _Int8Cfg
    from torchao.quantization import IntxWeightOnlyConfig as _IntxCfg
    from torchao.quantization.quant_api import _is_linear as _torchao_is_linear
    from torchao.quantization.transform_module import _QUANTIZE_CONFIG_HANDLER as _TORCHAO_HANDLER
    TORCHAO_AVAILABLE = True
except Exception:
    TORCHAO_AVAILABLE = False
    _torchao_is_linear = None
    _TORCHAO_HANDLER = None

# FramePack's dit_common imports `LayerNorm` from diffusers.models.normalization,
# which was removed in diffusers >= 0.35. The upstream class is the same as
# torch.nn.LayerNorm, so restore the alias for newer TPU images.
try:
    import diffusers.models.normalization as _diffusers_norm
    if not hasattr(_diffusers_norm, "LayerNorm"):
        _diffusers_norm.LayerNorm = torch.nn.LayerNorm
except Exception:
    pass
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


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def _save_bcthw_as_mp4(x, output_filename, fps=30, crf=16):
    """Save a BCTHW video tensor, falling back to ffmpeg on newer torchvision.

    torchvision >= 0.29 no longer exposes ``io.write_video``, which upstream
    FramePack still calls. Keep using the upstream helper when it is available
    and otherwise write raw RGB frames straight to ffmpeg via a pipe.
    """
    if hasattr(torchvision.io, "write_video"):
        return save_bcthw_as_mp4(x, output_filename, fps=fps, crf=crf)

    os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
    x = torch.clamp(x.float(), -1.0, 1.0) * 127.5 + 127.5
    x = x.detach().cpu().to(torch.uint8)

    b, _c, t, h, w = x.shape
    per_row = b
    for p in (6, 5, 4, 3, 2):
        if b % p == 0:
            per_row = p
            break
    rows = b // per_row

    # (rows, per_row, c, t, h, w) -> (t, rows*h, per_row*w, c)
    frames = x.reshape(rows, per_row, x.shape[1], t, h, w)
    frames = frames.permute(3, 0, 4, 1, 5, 2).reshape(t, rows * h, per_row * w, x.shape[1])
    frames = frames.contiguous().numpy()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to save videos with this torchvision version")

    out_h, out_w = frames.shape[1], frames.shape[2]
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{out_w}x{out_h}", "-r", str(fps), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-crf", str(int(crf)),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        output_filename,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        proc.stdin.write(frames.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}")
    except Exception:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
        proc.wait()
        raise
    return output_filename


def resolve_input_path(path):
    """Resolve an input image path against the folder the CLI was run from.

    Absolute paths are used as-is. Relative paths are checked in the invocation
    directory first, then in the script directory for backwards compatibility
    with images placed next to framepack-cli.py.
    """
    path = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(path):
        return os.path.abspath(path)

    in_current = os.path.abspath(os.path.join(INVOCATION_DIR, path))
    if os.path.exists(in_current):
        return in_current

    in_root = os.path.abspath(os.path.join(ROOT, path))
    if os.path.exists(in_root):
        return in_root

    return in_current


def looks_like_image_argument(value):
    """Whether a positional token refers to an input image rather than prompt text."""
    if not value:
        return False
    expanded = os.path.expandvars(os.path.expanduser(value))
    if os.path.splitext(expanded)[1].lower() in IMAGE_EXTENSIONS:
        return True
    return os.path.isfile(resolve_input_path(value))


def find_image_in_current_folder():
    """Return the most recently modified image in the invocation folder, if any."""
    try:
        names = os.listdir(INVOCATION_DIR)
    except OSError:
        return None
    images = [
        os.path.join(INVOCATION_DIR, name)
        for name in names
        if name.lower().endswith(IMAGE_EXTENSIONS)
        and os.path.isfile(os.path.join(INVOCATION_DIR, name))
    ]
    if not images:
        return None
    return max(images, key=os.path.getmtime)


# Output to the folder the CLI was launched from, not the script folder.
outputs_folder = INVOCATION_DIR
os.makedirs(outputs_folder, exist_ok=True)


def _keep_output_capture_on_swapped_model(model):
    """Keep hidden-state capture working after DynamicSwapInstaller.

    DynamicSwapInstaller replaces a module's class with a generated subclass.
    transformers >= 5 looks output hooks up by the exact class string, so the
    generated name makes `output_hidden_states=True` silently return None.
    Give the generated class the original identity; the override stays valid
    on older transformers versions too.
    """
    swapped_cls = type(model)
    if not swapped_cls.__name__.startswith("DynamicSwap_"):
        return
    original_cls = swapped_cls.__bases__[0]
    swapped_cls.__name__ = original_cls.__name__
    swapped_cls.__qualname__ = original_cls.__qualname__
    swapped_cls.__module__ = original_cls.__module__


@contextmanager
def _memory_efficient_attention_on_older_gpus():
    """Avoid the O(n^2) SDPA fallback on GPUs without bf16 flash support.

    FramePack's attention kernels call `F.scaled_dot_product_attention` with
    bfloat16 tensors. On pre-Ampere GPUs (e.g. T4) PyTorch has no efficient
    bf16 kernel and falls back to materializing the full attention matrix.
    Running that call in fp16 through the memory-efficient kernel keeps VRAM
    usage bounded; newer GPUs keep their native flash/cudnn kernels.
    """
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0)):
        yield
        return

    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    original_sdpa = F.scaled_dot_product_attention

    def sdpa(query, key, value, *args, **kwargs):
        if query.is_cuda and query.dtype == torch.bfloat16:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                result = original_sdpa(query.half(), key.half(), value.half(), *args, **kwargs)
            return result.to(torch.bfloat16)
        return original_sdpa(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa


def _quantize_transformer_memory_efficient(transformer, cfg):
    if not TORCHAO_AVAILABLE:
        return
    handler = _TORCHAO_HANDLER[type(cfg)]
    target_names = [name for name, mod in transformer.named_modules() if name and _torchao_is_linear(mod)]
    total = len(target_names)
    for idx, fqn in enumerate(target_names, 1):
        parts = fqn.split(".")
        parent = transformer
        for p in parts[:-1]:
            parent = getattr(parent, p)
        attr = parts[-1]
        mod = getattr(parent, attr)
        new_mod = handler(mod, cfg)
        setattr(parent, attr, new_mod)
        del mod, new_mod
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if idx % 50 == 0 or idx == total:
            print(f"  quantized {idx}/{total} linear modules", flush=True)


def load_models():
    global free_mem_gb, high_vram
    global text_encoder, text_encoder_2, tokenizer, tokenizer_2, vae
    global feature_extractor, image_encoder, transformer

    free_mem_gb = get_cuda_free_memory_gb(gpu)
    high_vram = free_mem_gb > 60

    print(f"Free VRAM {free_mem_gb} GB")
    print(f"High-VRAM Mode: {high_vram}")

    dtype = INFERENCE_DTYPE or torch.float16
    transformer_dtype = INFERENCE_DTYPE or torch.bfloat16
    text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True).cpu()
    text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="text_encoder_2", torch_dtype=dtype, low_cpu_mem_usage=True).cpu()
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer_2")
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True).cpu()

    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder="feature_extractor")
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=dtype, low_cpu_mem_usage=True).cpu()

    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained("lllyasviel/FramePackI2V_HY", torch_dtype=transformer_dtype, low_cpu_mem_usage=True).cpu()

    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    image_encoder.eval()
    transformer.eval()

    if not high_vram:
        vae.enable_slicing()
        vae.enable_tiling()

    transformer.high_quality_fp32_output_for_inference = False
    print("transformer.high_quality_fp32_output_for_inference = False")

    transformer.to(dtype=transformer_dtype)
    vae.to(dtype=dtype)
    image_encoder.to(dtype=dtype)
    text_encoder.to(dtype=dtype)
    text_encoder_2.to(dtype=dtype)

    # Quantize transformer to fp8 (weight-only) to reduce memory on iGPU.
    if TORCHAO_AVAILABLE:
        print("Quantizing transformer to int4 (weight-only) ...", flush=True)
        _quantize_transformer_memory_efficient(transformer, _IntxCfg(weight_dtype=torch.int4))
        print("Transformer quantization done.", flush=True)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    transformer.requires_grad_(False)

    if not high_vram:
        # Keep the largest models on CPU and swap individual layers onto the
        # accelerator. The TPU only has ~16 GiB HBM, so we cannot keep the
        # transformer, text encoders, VAE and image encoder resident together.
        DynamicSwapInstaller.install_model(transformer, device=gpu)
        DynamicSwapInstaller.install_model(text_encoder, device=gpu)
        _keep_output_capture_on_swapped_model(text_encoder)
        if not TPU_AVAILABLE:
            text_encoder_2.to(gpu)
            image_encoder.to(gpu)
            vae.to(gpu)
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

    dynamic_swap = not high_vram
    try:
        if dynamic_swap:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

        # Text encoding
        print("Text encoding ...")
        if dynamic_swap:
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
        if dynamic_swap:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        print("CLIP Vision encoding ...")
        if dynamic_swap:
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

            if dynamic_swap:
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
                dtype=INFERENCE_DTYPE or torch.bfloat16,
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

            if dynamic_swap:
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

            if dynamic_swap:
                unload_complete_models()

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            _save_bcthw_as_mp4(history_pixels, output_path, fps=30, crf=mp4_crf)
            print(f"Saved {output_path} (latent frames: {total_generated_latent_frames})")

            if is_last_section:
                break
    except Exception:
        traceback.print_exc()
        if dynamic_swap:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
        raise
    finally:
        if dynamic_swap:
            try:
                unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
            except Exception:
                pass

    return output_path


def main():
    parser = argparse.ArgumentParser(description="FramePack CLI")
    parser.add_argument(
        "positional", nargs="*", metavar="ARG",
        help="Prompt text and/or input image path. The image may be given here "
             "in either order instead of with --input/-i.",
    )
    parser.add_argument("--prompt", dest="prompt_opt", default=None, help="Text prompt (alternative to positional).")
    parser.add_argument(
        "--input", "-i", default=None,
        help="Input image path (absolute or relative to the current folder). "
             "If omitted, the most recent image in the current folder is used.",
    )
    parser.add_argument("--negative", default="", help="Negative prompt.")
    parser.add_argument("--output", default=None, help="Explicit output file path.")
    parser.add_argument("--seed", type=int, default=31337, help="Random seed.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Total video length in seconds.")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps.")
    parser.add_argument("--latent-window-size", type=int, default=9, help="Latent window size.")
    parser.add_argument("--cfg", type=float, default=1.0, help="CFG scale.")
    parser.add_argument("--gs", type=float, default=10.0, help="Distilled CFG scale.")
    parser.add_argument("--rs", type=float, default=0.0, help="CFG re-scale.")
    parser.add_argument("--gpu-memory-preservation", type=float, default=12.0, help="GPU memory preservation (GB).")
    parser.add_argument("--teacache", action="store_true", default=True, help="Use TeaCache.")
    parser.add_argument("--no-teacache", dest="teacache", action="store_false", help="Disable TeaCache.")
    parser.add_argument("--crf", type=int, default=16, help="MP4 CRF (lower = better quality).")
    args = parser.parse_args()

    # Accept the image either as a positional argument (in either order relative
    # to the prompt) or via --input/-i, as in:
    #   framepack-cli.py image.png "big waves crashing"
    #   framepack-cli.py "big waves crashing" image.png
    image_args = [a for a in args.positional if looks_like_image_argument(a)]
    prompt_args = [a for a in args.positional if not looks_like_image_argument(a)]

    input_arg = args.input
    if input_arg is not None and image_args:
        parser.error(
            f"input image given twice: {image_args[0]!r} as a positional argument "
            "and --input/-i; use one or the other"
        )
    if len(image_args) > 1:
        parser.error("more than one input image given: " + ", ".join(image_args))
    if image_args:
        input_arg = image_args[0]

    prompt_from_positionals = " ".join(prompt_args).strip()
    if args.prompt_opt is not None:
        if prompt_from_positionals:
            parser.error("prompt given twice: pass it either positionally or with --prompt, not both")
        prompt = args.prompt_opt
    else:
        prompt = prompt_from_positionals

    if not prompt:
        parser.error("a prompt is required")

    if input_arg:
        input_path = resolve_input_path(input_arg)
    else:
        input_path = find_image_in_current_folder()
        if input_path:
            print(f"No --input given; using image from current folder: {input_path}")

    if not input_path or not os.path.isfile(input_path):
        if input_arg:
            parser.error(f"input image not found: {input_arg} (resolved to {input_path})")
        parser.error(
            "no input image found: pass --input/-i or place an image in the "
            f"current folder ({INVOCATION_DIR})"
        )

    input_image = np.array(Image.open(input_path).convert("RGB"))

    load_models()

    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.abspath(os.path.join(INVOCATION_DIR, output_path))
    else:
        output_path = os.path.join(outputs_folder, f"{sanitize_filename(prompt)}-{args.seed}.mp4")

    # Ensure the destination directory exists.
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Output: {output_path}")
    with _memory_efficient_attention_on_older_gpus():
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

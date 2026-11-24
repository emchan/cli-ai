#!/usr/bin/env python3
"""Fooocus command-line interface.

Usage:
    python fooocus-cli.py "a cute cat" [--negative "..."] [--seed N]

Generates an image with Fooocus and saves it to outputs/<prompt>-<seed>.png.
"""
import argparse
import datetime
import os
import re
import shutil
import subprocess
import sys

# Script/output root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder the CLI was launched from. The script chdir()s to the upstream Fooocus
# source below, so keep this to resolve --output against the caller's folder.
INVOCATION_DIR = os.getcwd()


def is_tpu_backend():
    """Whether this interpreter sees PyTorch's native `tpu` accelerator.

    Set CLI_AI_BACKEND=tpu/native-tpu to force the TPU path, or
    CLI_AI_BACKEND=cuda/rocm/cpu to force the original path.
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
    # The TPU VM already ships a working torch/torch_tpu stack. Do not create
    # a CUDA/ROCm venv or install the CUDA-era Fooocus requirements over it.
    os.environ["FOOOCUS_NO_VENV"] = "1"
    os.environ["FOOOCUS_SKIP_PIP_INSTALL"] = "1"

# Ensure a dedicated env exists for this CLI (prefer Miniforge/conda/mamba).
VENV_DIR = os.path.expanduser("~/cli-ai/fooocus-cli")
if os.environ.get("FOOOCUS_NO_VENV") == "1":
    pass
else:
    conda_bin = shutil.which("conda")
    mamba_bin = shutil.which("mamba")
    micromamba_bin = shutil.which("micromamba")
    forge_bin = mamba_bin or conda_bin or micromamba_bin

    def _env_python(prefix):
        return os.path.join(prefix, "bin", "python")

    # Auto-install Miniforge if no conda/mamba tool is available.
    if not forge_bin:
        miniforge_prefix = os.path.expanduser("~/miniforge3")
        miniforge_bin = os.path.join(miniforge_prefix, "bin", "conda")
        if os.path.exists(miniforge_bin):
            forge_bin = miniforge_bin
        else:
            print("Miniforge/conda not found; downloading and installing Miniforge...", flush=True)
            try:
                installer = os.path.join("/tmp", "Miniforge3-Linux-x86_64.sh")
                if not os.path.exists(installer):
                    subprocess.run(
                        ["bash", "-lc",
                         "curl -L -o /tmp/Miniforge3-Linux-x86_64.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"],
                        check=True,
                    )
                subprocess.run(
                    ["bash", installer, "-b", "-p", miniforge_prefix],
                    check=True,
                )
                if os.path.exists(miniforge_bin):
                    forge_bin = miniforge_bin
                else:
                    print("Miniforge install finished but conda not found at expected path.", flush=True)
            except Exception as e:
                print(f"Miniforge install failed: {e}", flush=True)

    if forge_bin and not os.path.abspath(sys.prefix).startswith(os.path.abspath(VENV_DIR)):
        os.makedirs(os.path.dirname(VENV_DIR), exist_ok=True)
        env_py = _env_python(VENV_DIR)
        if not os.path.exists(env_py):
            print(f"Creating conda env (miniforge/conda) at {VENV_DIR} ...", flush=True)
            try:
                if micromamba_bin and "micromamba" in (micromamba_bin or ""):
                    subprocess.run(
                        [micromamba_bin, "create", "-y", "-p", VENV_DIR, "python=3.10", "pip"],
                        check=True,
                    )
                else:
                    subprocess.run(
                        [forge_bin, "create", "-y", "-p", VENV_DIR, "python=3.10", "pip"],
                        check=True,
                    )
            except Exception as e:
                print(f"conda/mamba env creation failed: {e}", flush=True)
                os.environ["FOOOCUS_NO_VENV"] = "1"
        if os.environ.get("FOOOCUS_NO_VENV") != "1" and os.path.exists(env_py):
            req = os.path.join(SCRIPT_DIR, "requirements_rocm.txt")
            if os.path.exists(req) and os.environ.get("FOOOCUS_SKIP_PIP_INSTALL") != "1":
                print("Installing requirements into conda env ...", flush=True)
                try:
                    # Install ROCm-compatible dependencies.
                    subprocess.run(
                        [env_py, "-m", "pip", "install", "-r", req,
                         "--extra-index-url", "https://download.pytorch.org/whl/rocm7.2"],
                        check=True,
                    )
                    # Ensure a ROCm build of torch/torchvision is present.
                    try:
                        subprocess.run(
                            [env_py, "-c", "import torch,sys; sys.exit(0 if getattr(torch.version,'hip',None) else 1)"],
                            check=True,
                        )
                    except Exception:
                        subprocess.run(
                            [env_py, "-m", "pip", "install", "--force-reinstall", "--no-deps",
                             "torch", "torchvision",
                             "--index-url", "https://download.pytorch.org/whl/rocm7.2"],
                            check=True,
                        )
                except Exception as e:
                    print(f"pip install in conda env failed: {e}", flush=True)
            os.execv(env_py, [env_py] + sys.argv)

    if os.environ.get("FOOOCUS_NO_VENV") != "1":
        # Fallback to venv if no conda/mamba found.
        if not os.path.abspath(sys.prefix).startswith(os.path.abspath(VENV_DIR)):
            os.makedirs(os.path.dirname(VENV_DIR), exist_ok=True)
            venv_python = os.path.join(VENV_DIR, "bin", "python")
            if not os.path.exists(venv_python):
                print(f"Creating venv at {VENV_DIR} ...", flush=True)
                venv_ok = False
                try:
                    subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
                    venv_ok = True
                except Exception as e:
                    print(f"venv creation failed: {e}", flush=True)
                if not venv_ok:
                    try:
                        subprocess.run(["virtualenv", "-p", sys.executable, VENV_DIR], check=True)
                        venv_ok = True
                    except Exception as e:
                        print(f"virtualenv fallback failed: {e}", flush=True)
                if not venv_ok:
                    print("Warning: could not create venv. Continuing in current Python.", flush=True)
                    os.environ["FOOOCUS_NO_VENV"] = "1"
            if os.environ.get("FOOOCUS_NO_VENV") != "1" and os.path.exists(venv_python):
                pip = os.path.join(VENV_DIR, "bin", "pip")
                req = os.path.join(SCRIPT_DIR, "requirements_rocm.txt")
                if os.path.exists(req):
                    print("Installing requirements into venv ...", flush=True)
                    subprocess.run(
                        [pip, "install", "-r", req,
                         "--extra-index-url", "https://download.pytorch.org/whl/rocm7.2"],
                        check=True,
                    )
                # Ensure a ROCm build of torch/torchvision is present.
                try:
                    subprocess.run(
                        [venv_python, "-c", "import torch,sys; sys.exit(0 if getattr(torch.version,'hip',None) else 1)"],
                        check=True,
                    )
                except Exception:
                    subprocess.run(
                        [pip, "install", "--force-reinstall", "--no-deps",
                         "torch", "torchvision",
                         "--index-url", "https://download.pytorch.org/whl/rocm7.2"],
                        check=True,
                    )
                os.execv(venv_python, [venv_python] + sys.argv)

# If venv was skipped/unavailable, optionally install requirements into current env.
if os.environ.get("FOOOCUS_NO_VENV") == "1":
    req = os.path.join(SCRIPT_DIR, "requirements_rocm.txt")
    if os.path.exists(req) and os.environ.get("FOOOCUS_SKIP_PIP_INSTALL") != "1":
        try:
            import importlib.util
            missing = False
            for mod in ("supervision", "gradio", "torch"):
                if importlib.util.find_spec(mod) is None:
                    missing = True
                    break
            if missing:
                print("Installing requirements into current Python (FOOOCUS_NO_VENV=1) ...", flush=True)
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", req,
                     "--extra-index-url", "https://download.pytorch.org/whl/rocm7.2"],
                    check=True,
                )
            # Ensure a ROCm build of torch/torchvision is present.
            try:
                subprocess.run(
                    [sys.executable, "-c", "import torch,sys; sys.exit(0 if getattr(torch.version,'hip',None) else 1)"],
                    check=True,
                )
            except Exception:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps",
                     "torch", "torchvision",
                     "--index-url", "https://download.pytorch.org/whl/rocm7.2"],
                    check=True,
                )
        except Exception as e:
            print(f"Warning: could not auto-install requirements in current env: {e}", flush=True)


def ensure_tpu_requirements():
    """Install the TPU-safe dependency set, mirroring fooocus-tpu.sh.

    Fooocus's pinned requirements are CUDA-era and would replace NumPy/torch
    in the TPU image. Only install the small set the real pipeline needs and
    skip it entirely when everything is already present. See
    requirements_tpu.txt and the Google_TPU write-up in portpilot.
    """
    import importlib.metadata as md

    pinned = {
        "torchsde": "0.2.6",
        "gradio": "3.41.2",
        "gradio-client": "0.5.0",
        "fastapi": "0.104.1",
        "starlette": "0.27.0",
        "pydantic": "2.9.2",
        "websockets": "11.0.3",
    }
    required = [
        "trampoline", "timm", "supervision", "rembg", "onnxruntime",
        "opencv-python-headless", "ffmpy", "pydub", "semantic-version",
        "orjson", "importlib-resources",
    ]

    def missing():
        for name, version in pinned.items():
            try:
                if md.version(name) != version:
                    return True
            except md.PackageNotFoundError:
                return True
        for name in required:
            try:
                md.version(name)
            except md.PackageNotFoundError:
                return True
        return False

    if not missing():
        return

    print("Installing TPU-safe Fooocus dependencies ...", flush=True)
    pip = [sys.executable, "-m", "pip", "install", "--quiet"]
    commands = [
        pip + ["torchsde==0.2.6", "trampoline"],
        pip + ["--no-deps", "opencv-python-headless"],
        pip + ["supervision", "timm", "pycocotools", "addict",
               "segment-anything", "groundingdino-py", "onnxruntime", "rembg"],
        pip + ["--no-deps", "gradio==3.41.2", "gradio-client==0.5.0"],
        pip + ["fastapi==0.104.1", "starlette==0.27.0", "pydantic==2.9.2",
               "websockets==11.0.3"],
        pip + ["ffmpy", "pydub", "semantic-version", "orjson", "importlib-resources"],
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Warning: pip install failed ({e}); continuing.", flush=True)


if TPU_AVAILABLE and os.environ.get("FOOOCUS_TPU_SKIP_PIP_INSTALL") != "1":
    ensure_tpu_requirements()

# Parse our own CLI arguments *before* importing Fooocus, because Fooocus parses
# sys.argv during import.
_parser = argparse.ArgumentParser(description="Fooocus CLI")
_parser.add_argument("prompt", nargs="?", help="Text prompt to generate an image from.")
_parser.add_argument("--prompt", dest="prompt_opt", default=None, help="Text prompt (alternative to positional).")
_parser.add_argument("--negative", default=None, help="Negative prompt.")
_parser.add_argument("--seed", type=int, default=None, help="Seed for generation.")
_parser.add_argument(
    "--output", default=None,
    help="Explicit output file path (relative paths are resolved against the folder you run this from).",
)
CLI_ARGS = _parser.parse_args()

# Prevent Fooocus's argument parser from choking on our positional prompt.
if not TPU_AVAILABLE:
    # ROCm / WSL2: make the Radeon 780M usable.
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    os.environ.setdefault("ROCPROFILER_REGISTER_ENABLED", "0")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

# Disable xformers (never available on TPU) and keep Fooocus from parsing our
# positional prompt. TPU wants SDPA and full-size VRAM accounting, not the
# low-VRAM/FP16 policy that ROCm uses.
fooocus_flags = ["--disable-xformers"]
if TPU_AVAILABLE:
    fooocus_flags += ["--attention-pytorch", "--always-normal-vram"]
sys.argv = [sys.argv[0]] + fooocus_flags

# Fooocus source is not vendored; fetch on demand. The TPU port is verified
# against the v2.5.5 tag and gets its own cache dir so CUDA/ROCm checkouts are
# never patched by accident.
SOURCE_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "fooocus-src")
SOURCE_REPO = "https://github.com/lllyasviel/Fooocus.git"
SOURCE_DIR = "Fooocus-TPU" if TPU_AVAILABLE else "Fooocus"
TPU_TAG = "v2.5.5"
TPU_PATCH = os.path.join(SCRIPT_DIR, "fooocus_tpu_patch.diff")


def ensure_fooocus_source():
    candidates = [os.path.join(SOURCE_CACHE, SOURCE_DIR)]
    if not TPU_AVAILABLE:
        candidates.insert(0, "/tmp/fooocus-src")
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "modules")):
            return cand
    dest = os.path.join(SOURCE_CACHE, SOURCE_DIR)
    os.makedirs(SOURCE_CACHE, exist_ok=True)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    print("Downloading Fooocus source dependencies ...")
    clone = ["git", "clone", "--depth", "1"]
    if TPU_AVAILABLE:
        clone += ["--branch", TPU_TAG]
    subprocess.run(clone + [SOURCE_REPO, dest], check=True)
    return dest


def apply_tpu_patch(source_root):
    """Apply the portpilot Google_TPU patch (idempotent) to Fooocus v2.5.5."""
    if not os.path.isfile(TPU_PATCH):
        raise RuntimeError(f"Fooocus TPU patch not found: {TPU_PATCH}")

    marker = os.path.join(source_root, "ldm_patched", "modules", "model_management.py")
    if os.path.isfile(marker):
        with open(marker, encoding="utf-8", errors="replace") as handle:
            if "def is_tpu()" in handle.read():
                return  # already patched

    check = subprocess.run(
        ["git", "apply", "--check", TPU_PATCH], cwd=source_root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "Fooocus TPU patch does not apply to the cached source. "
            f"Expected upstream tag {TPU_TAG}.\n{check.stderr.strip()}"
        )
    print(f"Applying Fooocus TPU patch ({TPU_TAG}) ...", flush=True)
    subprocess.run(["git", "apply", TPU_PATCH], cwd=source_root, check=True)


SOURCE_ROOT = ensure_fooocus_source()
if TPU_AVAILABLE:
    apply_tpu_patch(SOURCE_ROOT)
os.chdir(SOURCE_ROOT)
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Keep generated images in <script_dir>/outputs.
os.environ.setdefault("path_outputs", os.path.join(SCRIPT_DIR, "outputs"))

# Store Hugging Face downloads under ~/cli-ai/hf_downloads.
HF_DOWNLOADS = os.path.expanduser("~/cli-ai/hf_downloads")
os.makedirs(HF_DOWNLOADS, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_DOWNLOADS)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_DOWNLOADS, "hub"))

def configure_numpy_compat():
    """Restore NumPy 1.x aliases used by Gradio 3.41 (NumPy 2 removed them)."""
    import numpy as np

    for name, value in (("bool8", np.bool_), ("float_", np.float64)):
        if not hasattr(np, name):
            setattr(np, name, value)
    if not hasattr(np, "obj2sctype"):
        np.obj2sctype = lambda obj: np.dtype(obj).type


if TPU_AVAILABLE:
    configure_numpy_compat()

# Prevent Fooocus from launching the Gradio web server on import.
import gradio as gr

gr.Blocks.launch = lambda self, *a, **k: None

# Configure Fooocus before it reads its args. TPU uses SDPA + bf16 and must not
# be pushed into the ROCm low-VRAM/FP16 policy.
try:
    import ldm_patched.modules.args_parser as _fooocus_args  # noqa: E402
    _fooocus_args.args.disable_xformers = True
    _fooocus_args.args.attention_pytorch = True
    if TPU_AVAILABLE:
        _fooocus_args.args.always_normal_vram = True
        _fooocus_args.args.always_low_vram = False
    else:
        _fooocus_args.args.vae_in_fp16 = True
        _fooocus_args.args.always_low_vram = True
except Exception:
    pass

# Import launch first: upstream download_models() runs at import time there.
# webui imports async_worker, which starts the model-loading thread immediately;
# importing webui before launch makes a fresh install fail with a missing
# checkpoint because the worker starts before the download finishes.
import launch  # noqa: F401,E402
import webui  # noqa: E402  (model loading / worker are now ready)
import modules.config  # noqa: E402
import modules.constants as _constants  # noqa: E402


def sanitize_filename(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9._-]", "", text)
    if not text:
        text = "prompt"
    return text[:80]


def resolve_output_path(path: str) -> str:
    """Resolve --output against the folder the CLI was run from.

    The upstream source is imported from its own directory (see the chdir()
    above), so a bare filename like ``beach.png`` would otherwise be written
    into that source cache instead of the caller's current folder.
    """
    path = os.path.expandvars(os.path.expanduser(path))
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(INVOCATION_DIR, path))


def main():
    args = CLI_ARGS
    prompt = args.prompt_opt if args.prompt_opt is not None else args.prompt
    if not prompt:
        _parser.error("a prompt is required")

    values = [component.value for component in webui.ctrls]

    try:
        prompt_index = webui.ctrls.index(webui.prompt)
        values[prompt_index] = prompt
    except ValueError:
        pass

    if args.negative is not None:
        try:
            neg_index = webui.ctrls.index(webui.negative_prompt)
            values[neg_index] = args.negative
        except (ValueError, AttributeError):
            pass

    import random

    if args.seed is not None:
        seed_value = args.seed
    else:
        seed_value = random.randint(_constants.MIN_SEED, _constants.MAX_SEED)

    for attr, val in (("seed_random", False), ("image_seed", seed_value)):
        comp = getattr(webui, attr, None)
        if comp is not None and comp in webui.ctrls:
            values[webui.ctrls.index(comp)] = val

    try:
        grid_index = webui.ctrls.index(webui.generate_image_grid)
        values[grid_index] = False
    except (ValueError, AttributeError):
        pass

    try:
        imgnum_index = webui.ctrls.index(webui.image_number)
        values[imgnum_index] = 1
    except (ValueError, AttributeError):
        pass

    task = webui.get_task(*values)

    for _ in webui.generate_clicked(task):
        pass

    results = getattr(task, "results", []) or []
    image_paths = [p for p in results if isinstance(p, str) and os.path.exists(p)]
    if not image_paths:
        print("Generation finished but no output image was produced.", file=sys.stderr)
        sys.exit(1)

    src = image_paths[0]

    seed_for_name = getattr(task, "seed", seed_value)

    if args.output:
        dst = resolve_output_path(args.output)
    else:
        os.makedirs(os.path.join(SCRIPT_DIR, "outputs"), exist_ok=True)
        dst = os.path.join(SCRIPT_DIR, "outputs",
                           f"{sanitize_filename(prompt)}-{seed_for_name}.png")

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    shutil.copyfile(src, dst)
    print(dst)
    os._exit(0)


if __name__ == "__main__":
    main()

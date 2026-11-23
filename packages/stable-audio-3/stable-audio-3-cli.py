#!/usr/bin/env python3
"""Stable Audio 3 command-line interface.

Usage:
    python stable-audio-3-cli.py "lo-fi hip hop beat, 90 BPM" [--model small-music] [--duration 30] [-o out.wav]

Generates an audio file from a text prompt using Stable Audio 3 and saves it to the
path given by --output (default: output.wav).
"""
import os
import sys
import shutil
import subprocess
import glob

# Script/output root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Self-bootstrapping environment (conda/Miniforge preferred; venv fallback).
# ---------------------------------------------------------------------------
VENV_DIR = os.environ.get("STABLE_AUDIO_VENV", os.path.expanduser("~/cli-ai/stable-audio-cli"))


def _base_python():
    for name in ("python3.11", "python3.10", "python3", "python"):
        path = shutil.which(name)
        if path:
            return path
    return sys.executable


def _venv_python(prefix):
    return os.path.join(prefix, "bin", "python")


def _venv_ok(python):
    env = dict(os.environ)
    env["PYTHONPATH"] = SCRIPT_DIR + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    try:
        subprocess.run(
            [python, "-c", "import torch, torchaudio, transformers, stable_audio_3"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=300,
            env=env,
        )
        return True
    except Exception:
        return False


def _torch_available(python):
    try:
        subprocess.run(
            [python, "-c", "import torch, torchaudio"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=120,
        )
        return True
    except Exception:
        return False


def _install_requirements(python):
    req = os.path.join(SCRIPT_DIR, "requirements.txt")
    if not os.path.exists(req):
        return
    try:
        if _torch_available(python):
            filtered = os.path.join(os.path.dirname(python), "..", "requirements.notorch.txt")
            filtered = os.path.normpath(filtered)
            with open(req) as f:
                lines = [ln for ln in f if not ln.strip().lower().startswith(("torch==", "torchaudio==", "torch>=", "torchaudio>="))]
            with open(filtered, "w") as f:
                f.writelines(lines)
            req_to_use = filtered
        else:
            req_to_use = req
        subprocess.run(
            [python, "-m", "pip", "install", "-r", req_to_use,
             "--extra-index-url", "https://download.pytorch.org/whl/cu126"],
            check=True,
        )
    except Exception as e:
        print(f"pip install failed: {e}", flush=True)


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


def ensure_venv():
    forge_bin, micromamba_bin = _ensure_miniforge()
    if forge_bin:
        parent = os.path.dirname(VENV_DIR)
        if parent:
            os.makedirs(parent, exist_ok=True)
        py = _venv_python(VENV_DIR)
        if not (os.path.exists(py) and _venv_ok(py)):
            if os.path.exists(VENV_DIR):
                shutil.rmtree(VENV_DIR)
            print(f"Creating conda env for stable-audio at {VENV_DIR} ...", flush=True)
            try:
                if micromamba_bin and "micromamba" in (micromamba_bin or ""):
                    subprocess.run([micromamba_bin, "create", "-y", "-p", VENV_DIR, "python=3.11", "pip"], check=True)
                else:
                    subprocess.run([forge_bin, "create", "-y", "-p", VENV_DIR, "python=3.11", "pip"], check=True)
            except Exception as e:
                print(f"conda env creation failed: {e}", flush=True)
        py = _venv_python(VENV_DIR)
        if os.path.exists(py):
            if not _venv_ok(py):
                _install_requirements(py)
            return py
    # Fallback to venv
    python = _venv_python(VENV_DIR)
    if not (os.path.exists(python) and _venv_ok(python)):
        if os.path.exists(VENV_DIR):
            shutil.rmtree(VENV_DIR)
        parent = os.path.dirname(VENV_DIR)
        if parent:
            os.makedirs(parent, exist_ok=True)
        subprocess.run([_base_python(), "-m", "venv", "--system-site-packages", VENV_DIR], check=True)
        python = _venv_python(VENV_DIR)
        if not _venv_ok(python):
            _install_requirements(python)
    return python


# Bootstrap when the current interpreter does not already provide the deps.
if os.environ.get("STABLE_AUDIO_CLI_SKIP_VENV") != "1":
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        import transformers  # noqa: F401
        import stable_audio_3  # noqa: F401

        _deps_present = True
    except Exception:
        _deps_present = False
    if not _deps_present:
        os.environ["PYTHONPATH"] = SCRIPT_DIR + (os.pathsep + os.environ.get("PYTHONPATH", "") if os.environ.get("PYTHONPATH") else "")
        _venv_python_path = ensure_venv()
        if os.path.abspath(sys.executable) != os.path.abspath(_venv_python_path):
            os.environ.setdefault("STABLE_AUDIO_CLI_BOOTSTRAPPED", "1")
            os.execv(_venv_python_path, [_venv_python_path] + sys.argv)

# Upstream stable-audio-3 package is vendored alongside this script.
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Store Hugging Face downloads under ~/cli-ai/hf_downloads.
HF_DOWNLOADS = os.path.expanduser("~/cli-ai/hf_downloads")
os.makedirs(HF_DOWNLOADS, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_DOWNLOADS)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_DOWNLOADS, "hub"))

# Register public (no-token) cocktailpeanut mirrors before loading.
import stable_audio_3.model as model_module  # noqa: E402
import stable_audio_3.model_configs as model_configs  # noqa: E402
from stable_audio_3.model_configs import ModelConfig  # noqa: E402

_MIRRORS = {
    "small-music": "cocktailpeanut/stable-audio-3-small-music",
    "small-sfx": "cocktailpeanut/stable-audio-3-small-sfx",
    "medium": "cocktailpeanut/stable-audio-3-medium",
}
for _name, _repo in _MIRRORS.items():
    _cfg = ModelConfig(_repo, "model_config.json", "model.safetensors")
    model_configs.models[_name] = _cfg
    model_configs.all_models[_name] = _cfg
model_module.all_models = model_configs.all_models

# Delegate to upstream CLI.
import stable_audio_3.cli as _upstream_cli  # noqa: E402

if __name__ == "__main__":
    _upstream_cli.main()

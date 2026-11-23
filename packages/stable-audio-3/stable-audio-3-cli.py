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
# Self-bootstrapping virtual environment.
# ---------------------------------------------------------------------------
VENV_DIR = os.environ.get("STABLE_AUDIO_VENV", os.path.expanduser("~/cli-ai/stable-audio-cli"))


def _base_python():
    for name in ("python3.11", "python3.10", "python3", "python"):
        path = shutil.which(name)
        if path:
            return path
    return sys.executable


def _venv_python():
    return os.path.join(VENV_DIR, "bin", "python")


def _venv_site_packages():
    matches = glob.glob(os.path.join(VENV_DIR, "lib", "python*", "site-packages"))
    if not matches:
        matches = glob.glob(os.path.join(VENV_DIR, "Lib", "site-packages"))
    if not matches:
        raise RuntimeError("could not locate venv site-packages")
    return matches[0]


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


def _create_venv():
    if os.path.exists(VENV_DIR):
        shutil.rmtree(VENV_DIR)
    parent = os.path.dirname(VENV_DIR)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Use system site-packages so a pre-installed torch (if any) is reused,
    # avoiding huge downloads on first run when possible.
    subprocess.run([_base_python(), "-m", "venv", "--system-site-packages", VENV_DIR], check=True)


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
    if _torch_available(python):
        # Reuse the (system) torch/torchaudio already available; install the rest.
        filtered = os.path.join(VENV_DIR, "requirements.notorch.txt")
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


def ensure_venv():
    python = _venv_python()
    if not (os.path.exists(python) and _venv_ok(python)):
        _create_venv()
        python = _venv_python()
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

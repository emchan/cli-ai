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
            req = os.path.join(SCRIPT_DIR, "requirements_original.txt")
            if os.path.exists(req) and os.environ.get("FOOOCUS_SKIP_PIP_INSTALL") != "1":
                print("Installing requirements into conda env ...", flush=True)
                try:
                    subprocess.run(
                        [env_py, "-m", "pip", "install", "-r", req,
                         "--extra-index-url", "https://download.pytorch.org/whl/cu121"],
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
                req = os.path.join(SCRIPT_DIR, "requirements_original.txt")
                if os.path.exists(req):
                    print("Installing requirements into venv ...", flush=True)
                    subprocess.run(
                        [pip, "install", "-r", req,
                         "--extra-index-url", "https://download.pytorch.org/whl/cu121"],
                        check=True,
                    )
                os.execv(venv_python, [venv_python] + sys.argv)

# If venv was skipped/unavailable, optionally install requirements into current env.
if os.environ.get("FOOOCUS_NO_VENV") == "1":
    req = os.path.join(SCRIPT_DIR, "requirements_original.txt")
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
                     "--extra-index-url", "https://download.pytorch.org/whl/cu121"],
                    check=True,
                )
        except Exception as e:
            print(f"Warning: could not auto-install requirements in current env: {e}", flush=True)

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
sys.argv = [sys.argv[0]]

# Fooocus source is not vendored; fetch on demand.
SOURCE_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "fooocus-src")
SOURCE_REPO = "https://github.com/lllyasviel/Fooocus.git"


def ensure_fooocus_source():
    candidates = ["/tmp/fooocus-src", os.path.join(SOURCE_CACHE, "Fooocus")]
    for cand in candidates:
        if os.path.isdir(os.path.join(cand, "modules")):
            return cand
    dest = os.path.join(SOURCE_CACHE, "Fooocus")
    os.makedirs(SOURCE_CACHE, exist_ok=True)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    print("Downloading Fooocus source dependencies ...")
    subprocess.run(["git", "clone", "--depth", "1", SOURCE_REPO, dest], check=True)
    return dest


SOURCE_ROOT = ensure_fooocus_source()
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

# Prevent Fooocus from launching the Gradio web server on import.
import gradio as gr

gr.Blocks.launch = lambda self, *a, **k: None

import webui  # noqa: E402  (triggers model loading / downloads)
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

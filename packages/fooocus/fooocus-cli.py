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

# Parse our own CLI arguments *before* importing Fooocus, because Fooocus parses
# sys.argv during import.
_parser = argparse.ArgumentParser(description="Fooocus CLI")
_parser.add_argument("prompt", nargs="?", help="Text prompt to generate an image from.")
_parser.add_argument("--prompt", dest="prompt_opt", default=None, help="Text prompt (alternative to positional).")
_parser.add_argument("--negative", default=None, help="Negative prompt.")
_parser.add_argument("--seed", type=int, default=None, help="Seed for generation.")
_parser.add_argument("--output", default=None, help="Explicit output file path.")
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
        dst = args.output
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

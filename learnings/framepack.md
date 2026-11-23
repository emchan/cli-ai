# Porting FramePack to the Command Line

This document details the steps used to port [FramePack](https://github.com/lllyasviel/FramePack)
to a self-contained command-line utility.

## 1. Study upstream

- Read the upstream `README.md` and `requirements.txt` to understand dependencies
  (`torch`, `torchvision`, `torchaudio` with CUDA 12.6, `diffusers==0.33.1`,
  `transformers==4.46.2`, `accelerate`, `gradio`, `safetensors`, `einops`, `av`, etc.)
  and the GUI entry point `python demo_gradio.py`.
- Read `demo_gradio.py` to learn the generation pipeline: model loading (HunyuanVideo
  text encoders/VAE, Siglip image encoder, `HunyuanVideoTransformer3DModelPacked`),
  text/image encoding, latent section sampling loop, VAE decode, and MP4 saving via
  `save_bcthw_as_mp4`.
- Note the GUI uses an `AsyncStream` and Gradio progress updates; these need to be
  replaced for a non-interactive CLI.

## 2. Create the package folder

Created `packages/framepack/` with:
- `framepack-cli.py` (the CLI entry point)
- `requirements.txt` (upstream dependencies)
- `README.md`
- On-demand source download instead of vendoring `diffusers_helper` (kept lightweight).

## 3. Self-bootstrapping virtual environment

Followed the pattern from `fooocus-cli.py`:
- If `torch`/`diffusers`/`transformers` cannot be imported, create a venv at
  `~/cli-ai/framepack-cli` (overridable via `FRAMEPACK_VENV`).
- Reuse a pre-installed `fp311` environment by dropping a `.pth` file into the venv
  site-packages when available.
- Install `requirements.txt` with `--extra-index-url https://download.pytorch.org/whl/cu126`.
- Re-exec the script with the venv interpreter.

## 4. Resolve upstream source

Instead of vendoring `diffusers_helper`, locate it in `/tmp/FramePack` or
`~/.cache/framepack-src/FramePack`, cloning from GitHub on demand.

## 5. Hugging Face cache

Set `HF_HOME` and `HUGGINGFACE_HUB_CACHE` to `~/cli-ai/hf_downloads` so models are
stored in the shared location used by the other CLI ports.

## 6. Convert the Gradio worker to a CLI

- Reused `demo_gradio.py`'s `worker()` pipeline, removing Gradio/`AsyncStream` and
  progress-bar HTML.
- Replaced progress updates with simple console prints (`Text encoding ...`,
  `Sampling x/y`).
- Kept model loading, memory management, TeaCache, and section sampling unchanged.

## 7. Default output format

Output filename uses a sanitized prompt plus the seed:

```
outputs/<prompt>-<seed>.mp4
```

MP4 is written with `save_bcthw_as_mp4` at 30 fps using the configured CRF.

## 8. Documentation

Updated `README.md` with CLI usage, options, and the prompts used to steer this port.
Created this `learnings/framepack.md`.
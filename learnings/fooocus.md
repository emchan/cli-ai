# Porting Fooocus to the Command Line

This document details the steps used to port [Fooocus](https://github.com/lllyasviel/Fooocus)
to a self-contained command-line utility.

## 1. Study upstream

- Read the upstream `README.md` and `requirements_versions.txt` to understand dependencies
  (`torch==2.1.0`, `torchvision==0.16.0`, `gradio==3.41.2`, `transformers==4.42.4`,
  `safetensors`, `accelerate`, `opencv-contrib-python-headless`, etc.) and the GUI
  entry point `python launch.py` (which launches a Gradio server).
- Read `webui.py`, `modules/async_worker.py`, and `modules/config.py` to learn the
  generation pipeline: model loading, prompt processing, SDXL sampling, VAE decode,
  and image saving via `modules.private_logger.log`.
- Note the GUI uses Gradio components and an async worker thread; these need to be
  replaced/driven for a non-interactive CLI.

## 2. Create the package folder

Created `packages/fooocus/` with:
- `fooocus-cli.py` (the CLI entry point)
- `requirements_original.txt` (upstream dependencies, plus torch/torchvision)
- `requirements_cli_nvidia.txt` (fork for NVIDIA)
- `README.md`
- Fooocus source downloaded on demand (not vendored) to keep the package lightweight.

## 3. Self-bootstrapping virtual environment

Followed the pattern from the other CLI ports:
- If the dedicated venv `~/cli-ai/fooocus-cli` is not active, create it with
  `python -m venv`.
- Install `requirements_original.txt` with `--extra-index-url https://download.pytorch.org/whl/cu121`.
- Re-exec the script with the venv interpreter.

## 4. Resolve upstream source

Instead of vendoring Fooocus, locate it in `/tmp/fooocus-src` or
`~/.cache/fooocus-src/Fooocus`, cloning from GitHub on demand.

## 5. Hugging Face cache

Set `HF_HOME` and `HUGGINGFACE_HUB_CACHE` to `~/cli-ai/hf_downloads` so models are
stored in the shared location used by the other CLI ports.

## 6. Convert the Gradio worker to a CLI

- Patch `gradio.Blocks.launch` to a no-op so importing `webui` does not start a server.
- Build the `AsyncTask` argument list from the default values of the Gradio components
  (`webui.ctrls`), overriding prompt, negative prompt, seed, image number, and
  disabling the image grid.
- Drive generation to completion by iterating `webui.generate_clicked(task)`.
- The worker thread (started on import of `modules.async_worker`) processes the task.

## 7. Default output format

Output filename uses a sanitized prompt plus the seed:

```
outputs/<prompt>-<seed>.png
```

Images are saved by Fooocus's `private_logger.log` and then copied to the final
path.

## 8. Documentation

Added/updated `README.md` in the package noting the link to the original project,
that this is a CLI port, and that the original license (GPL-3.0) is retained.
Created this `learnings/fooocus.md`.
PY
echo done
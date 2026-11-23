# Porting Stable Audio 3 to the Command Line

This document details the steps used to port [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3)
(using the public mirrors from [cocktailpeanut/stable-audio-3-small.pinokio](https://github.com/cocktailpeanut/stable-audio-3-small.pinokio))
to a self-contained command-line utility.

## 1. Study upstream

- Read the upstream `README.md` and `pyproject.toml` to understand dependencies
  (`torch==2.7.1`, `torchaudio==2.7.1`, `transformers>=5.8.0`, `huggingface-hub`, `soundfile`,
  `safetensors`, `einops`, `accelerate`, etc.) and the console entry point
  `stable-audio = "stable_audio_3.cli:main"`.
- Read `stable_audio_3/cli.py` to learn the CLI surface (`--model`, `-p/--prompt`,
  `--duration`, `--steps`, `-o/--output`, audio-to-audio, inpainting, LoRA, etc.).
- Read `stable_audio_3/model.py` and `model_configs.py` to understand model loading and
  how model names map to Hugging Face repos.
- Note the Pinokio launcher uses public mirrors (`cocktailpeanut/stable-audio-3-small-music`,
  `...-small-sfx`, `...-medium`) instead of the gated `stabilityai/...` repos, and registers
  them via `ModelConfig` before `StableAudioModel.from_pretrained(...)`.

## 2. Create the package folder

Created `packages/stable-audio-3/` with:
- `stable-audio-3-cli.py` (the CLI entry point)
- `requirements.txt` (upstream dependencies)
- `README.md`
- Vendored copy of the upstream `stable_audio_3/` package (all `.py` modules) so no
  source download is needed at runtime.

## 3. Self-bootstrapping virtual environment

Followed the pattern from `fooocus-cli.py` / `framepack-cli.py`:
- If `torch`/`torchaudio`/`transformers`/`stable_audio_3` cannot be imported, create a
  venv at `~/cli-ai/stable-audio-cli` (overridable via `STABLE_AUDIO_VENV`).
- Use `--system-site-packages` so a pre-installed torch is reused when available.
- If torch is already importable, filter `torch`/`torchaudio` lines out of
  `requirements.txt` before installing the rest.
- Re-exec the script with the venv interpreter.

## 4. Resolve upstream source

Instead of cloning at runtime (slow/unreliable), vendor the entire `stable_audio_3`
package into the script directory and put `SCRIPT_DIR` on `sys.path`.

## 5. Hugging Face cache

Set `HF_HOME` and `HUGGINGFACE_HUB_CACHE` to `~/cli-ai/hf_downloads` so models are
stored in the shared location used by the other CLI ports.

## 6. Register public mirrors

Before importing the CLI, mutate `stable_audio_3.model_configs` to point `small-music`,
`small-sfx`, and `medium` at the `cocktailpeanut/...` mirrors:

```python
config = ModelConfig("cocktailpeanut/stable-audio-3-small-music", "model_config.json", "model.safetensors")
model_configs.models["small-music"] = config
model_configs.all_models["small-music"] = config
model_module.all_models = model_configs.all_models
```

## 7. Delegate to upstream CLI

After setup, `import stable_audio_3.cli` and call its `main()`, which parses `sys.argv`
and runs generation.

## 8. Default output format

Changed the default output from `output.wav` to `output.mp3`. Since `torchaudio.save`
requires `torchcodec` in this environment, `_save_output` was rewritten to:
- save a temporary WAV via `soundfile`, then convert to MP3 with `ffmpeg` when the
  target extension is `.mp3`;
- otherwise write directly with `soundfile`.

## 9. Convenience aliases

Created `packages/stable-audio/` and `packages/stable-audio-3-small/` with thin wrapper
scripts that `exec` the canonical `packages/stable-audio-3/stable-audio-3-cli.py`, so the
port is reachable under several likely directory names.

## 10. Documentation

Added/updated `README.md` files in each package noting the link to the original project,
that this is a CLI port, and that the original license is retained. Created this
`learnings/stableaudio.md`.
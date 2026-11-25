# Porting FramePack CLI to ROCm

This document details the steps used to port `packages/framepack/framepack-cli.py`
from its original NVIDIA/CUDA setup to AMD/ROCm.

## Environment

- GPU: AMD Radeon 780M (`gfx1103`, RDNA3) via WSL2/DXG
- Stack: ROCm 7.2.4 + PyTorch 2.14.1 (ROCm) + diffusers 0.33.1
- Venv: `~/cli-ai/framepack-cli-uv` (uv, Python 3.12), ROCm wheels from
  `https://download.pytorch.org/whl/rocm7.2`

## Key changes

- Set `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `ROCPROFILER_REGISTER_ENABLED=0`,
  `HIP_VISIBLE_DEVICES=0`, `MIOPEN_FIND_MODE=FAST` before any torch import.
- Bootstrap now requires a HIP build of torch (`torch.version.hip`) and installs
  from `requirements_rocm.txt` with the ROCm index.
- Reuse a ROCm venv (uv) instead of the CUDA conda env.
- Quantize the transformer to int4 (weight-only, torchao) to reduce memory on the
  iGPU; default `--gpu-memory-preservation` raised to 12 GB.
- `transformer.high_quality_fp32_output_for_inference = False` (ROCm).

## Notes

- Model loading uses `low_cpu_mem_usage=True`.
- The iGPU reports ~21 GB VRAM but real budget is ~8 GB RAM; expect swap pressure.
- Generation is slow; quantization itself takes ~30 minutes on this hardware.

## Findings from `framepack-cli-quant.py` test run

- Quantization is in-memory only in the original script. The quantized variant writes a
  reusable cache under `/tmp/opencode/framepack-quant-cache/` (override with
  `FRAMEPACK_QUANT_CACHE`).
- `torch.save(transformer, ...)` (whole model) failed with
  `'functools.partial' object has no attribute '__name__'`, so the cache stores
  `transformer.state_dict()` instead. On a cache hit it loads with
  `torch.load(..., mmap=True)` and applies the tensors with
  `load_state_dict(..., assign=True)`.
- `weights_only=False` is required when loading the cached quantized state dict
  because it contains torchao quantized tensor subclasses.
- Quantization is checkpointed every 50 of 549 linear modules to
  `transformer-int4-*.partial.pt`; a crashed/interrupted run resumes from the
  last completed module instead of starting over.
- Generation is checkpointed after each completed latent section to
  `/tmp/opencode/framepack-gen-checkpoints/generation-*.ckpt.pt`; an OOM or crash
  resumes from the last completed section. Main loop retries up to 3 times on
  CUDA/ROCm OOM after clearing cache.
- On the Radeon 780M test, quantization completed in roughly 15-20 minutes and
  produced a 13 GB cache file. The first 8-step sampling section took about
  36 minutes. VAE decode after sampling is also very slow and may appear stalled.
- Peak RSS reached ~14.8 GB while writing the 13 GB cache; expect heavy swap
  pressure on this iGPU.
- torchao import emits warnings about missing `_C_cutlass`/`_C_mxfp8` and
  "Unable to import torchao Tensor objects", but int4 weight-only quantization
  still works.
- Process state can sit in uninterruptible disk wait (`D`) while paging the
  13 GB cache; avoid starting a second instance during this time.
- The script now has a stderr status bar showing CPU/RAM and GPU VRAM usage plus
  the current stage (`set_status_stage`). GPU utilization is not available without
  `amdsmi`; VRAM allocated/reserved is shown instead.

## mmap usage

- `torch.load(..., mmap=True)` is used for cached quantized transformer state dicts,
  quantization partial checkpoints, and generation checkpoints.
- This maps tensor storage from disk and pages it lazily instead of loading the
  whole checkpoint into RAM up front.
- `weights_only=False` is required because cached quantized state dicts contain
  torchao quantized tensor subclasses.
- Tradeoff: lower initial memory pressure, but disk paging can cause stalls and
  uninterruptible disk wait (`D`) while accessing large caches.

## Generation flow (2-second, 8-step test)

1. **Model loading / quantization** (`load_models`)
   - Load text encoders, VAE, image encoder, transformer.
   - Quantize 549 transformer linear modules (default NF4; falls back to int3 on OOM).
   - Checkpoint quantization every 50 modules.
   - Install dynamic layer swapping on low-VRAM systems.
   - Observed: roughly 15-20 minutes on cache miss; cache load is much faster.
   - Memory: about 15 GB RAM, 7.5 GB GPU VRAM allocated / 11.7 GB reserved.
2. **Text encoding**
   - Encode prompt and optional negative prompt.
   - Crop/pad to 512 tokens.
3. **Image processing**
   - Find nearest 640px bucket and resize/crop input image.
4. **VAE encoding**
   - Encode the input image to the starting latent.
5. **CLIP Vision encoding**
   - Encode the input image with SigLIP.
6. **Sampling loop**
   - Number of latent sections = `max(round(seconds * 30 / (latent_window_size * 4)), 1)`.
   - For 2 seconds with `latent_window_size=9`, this is 2 sections.
   - Each section runs `steps` denoising steps (8 in the test, 25 by default).
   - Observed: about 5 minutes per step, ~30 minutes per 8-step section.
   - TeaCache may be enabled; dynamic swap moves the transformer to GPU per section.
7. **VAE decoding**
   - Decode generated latents for the current section.
   - This is also expensive and can take several minutes; status bar may remain in
     `VAE decoding section N/2`.
8. **Pixel append and save**
   - Soft-append decoded pixels to history and write the MP4 after each section.
   - Save a generation checkpoint after each completed section.
9. **Next section / finalization**
   - Repeat sampling and VAE decode for the final section.
   - Remove the generation checkpoint on successful completion.

Observed resource usage during the 2-second test:
- RAM: about 15.1 / 15.6 GB.
- GPU VRAM: about 7.5-12.4 GB allocated, 11.7-12.8 GB reserved, 21.8 GB total.
- First 8-step sampling section: about 30 minutes.
- VAE decode after section 1: still running/long; treat it as a major cost.
- VAE encode/decode now try GPU first and fall back to CPU on OOM, then stay on CPU
  for the rest of the run.

## Why sampling is slow

- Target hardware is a Radeon 780M iGPU over WSL2/DXG, not an optimized CUDA/ROCm dGPU path.
- No fast attention backend is available (no XFormers, Flash Attention, or Sage Attention), so attention falls back to slower PyTorch SDPA.
- Dynamic layer swapping moves transformer layers between CPU and GPU because practical VRAM budget is much lower than the reported ~21 GB.
- Quantized int4/NF4 weights reduce memory but add dequantization overhead to matmuls.
- RAM is nearly full (~15/15.6 GB), causing swap/paging and frequent GC/cache cleanup.
- Default sampling is 25 steps per latent section; even 8 steps took about 30 minutes in testing.
PY
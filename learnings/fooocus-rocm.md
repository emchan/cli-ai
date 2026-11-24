# Porting Fooocus to ROCm

This document details the steps used to port `packages/fooocus/fooocus-cli.py`
from its original NVIDIA/CUDA setup to AMD/ROCm, and how the final approach
differed from the first attempt.

## 1. Study upstream (ROCm specifics)

- Fooocus targets `torch==2.1.0` with CUDA 12.1 wheels. The ROCm port uses
  `torch==2.14.1+rocm7.2` / `torchvision==0.29.1+rocm7.2` from
  `https://download.pytorch.org/whl/rocm7.2`.
- Fooocus's `ldm_patched` model management calls `torch.cuda.*` APIs. On ROCm these
  resolve to the HIP backend, so no source-level device changes are required, but
  `torch.version.cuda` is `None` on ROCm. Fooocus treats this as "not NVIDIA" and
  falls back to safer defaults (FP32 VAE, sub-quadratic attention) unless
  overridden.
- `xformers` is not available for ROCm in this environment, so disable it via
  Fooocus's args before import.
- Set `ROCPROFILER_REGISTER_ENABLED=0` (required for WSL2/DXG; otherwise
  rocprofiler aborts during `torch.cuda.device_count()`). This was the key missing
  piece in the first attempt: ROCm 7.2 wheels imported but reported zero GPUs
  until this variable was set.
- Set `HSA_OVERRIDE_GFX_VERSION=11.0.0` for the Radeon 780M (`gfx1103`), whose
  rocBLAS Tensile kernels are absent from ROCm 7.2 (it reuses `gfx1100` kernels).
- Pin the Gradio 3.41.2 web stack (pydantic 1.10.13, fastapi 0.99.1,
  starlette 0.27.0, uvicorn 0.23.2, python-multipart 0.0.6) because Gradio 3.41.2
  is a 2023 release incompatible with 2026 FastAPI/Pydantic.
- Pass `--attention-pytorch --vae-in-fp16 --always-low-vram --disable-xformers`
  to Fooocus for the iGPU (ROCm is not auto-detected as NVIDIA).

## 2. Package changes

- `packages/fooocus/requirements_rocm.txt` mirrors `requirements_original.txt` but
  pins the ROCm builds of `torch`/`torchvision` and the Gradio web-stack pins,
  and drops the CUDA-only torch pin.
- `packages/fooocus/fooocus-cli.py` now bootstraps from `requirements_rocm.txt`
  with `--extra-index-url https://download.pytorch.org/whl/rocm7.2`.
- The bootstrap installs/verifies a ROCm build of `torch`/`torchvision`
  (`getattr(torch.version, "hip", None)`) instead of relying on the CUDA wheels.

## 3. Differences from the first (rocm6.3) attempt

| Aspect | First attempt (rocm6.3) | Final approach (rocm7.2 / portpilot) |
| --- | --- | --- |
| PyTorch | `2.9.1+rocm6.3` | `2.14.1+rocm7.2` |
| GPU detection | Failed (`No HIP GPUs are available`) | Works after `ROCPROFILER_REGISTER_ENABLED=0` |
| rocprofiler | Not disabled (crashed) | Disabled (required on WSL2/DXG) |
| Fooocus flags | Only `--disable-xformers` | `--attention-pytorch --vae-in-fp16 --always-low-vram --disable-xformers` |
| Web stack | Modern (broke Gradio 3.41.2) | Pinned to 2023-era versions |
| VAE dtype | FP32 (Fooocus default on non-NVIDIA) | FP16 via `--vae-in-fp16` |
| Attention | Sub-quadratic fallback | PyTorch SDPA |

## 4. Self-bootstrapping virtual environment

- Reused the existing `~/cli-ai/fooocus-cli` environment path, converted to the
  ROCm 7.2 wheels.
- Conda/venv creation still targets Python 3.10; ROCm 7.2 wheels for `cp310` are
  available.
- Pip installs use `--extra-index-url https://download.pytorch.org/whl/rocm7.2`.

## 5. Runtime adjustments

- Set `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `ROCPROFILER_REGISTER_ENABLED=0`,
  `HIP_VISIBLE_DEVICES=0`, and `MIOPEN_FIND_MODE=FAST`.
- Configure Fooocus args before import: `disable_xformers`, `attention_pytorch`,
  `vae_in_fp16`, `always_low_vram`.
- `gr.Blocks.launch` is still patched to a no-op, and generation is driven the same
  way as the CUDA port.

## 6. Output

Output handling is unchanged: images are written to
`outputs/<prompt>-<seed>.png` (or `--output`).

## 7. Why this works

This approach works because it aligns Fooocus with the WSL2/ROCm runtime that
actually exposes the GPU instead of fighting it:

1. **Right PyTorch build + right index** — Uses `torch==2.14.1+rocm7.2` from
   `download.pytorch.org/whl/rocm7.2`, whose HIP backend talks to the Windows
   driver through `/dev/dxg`. Older (rocm6.3) wheels imported but couldn't see
   the device.

2. **Disable the profiler registration** — `ROCPROFILER_REGISTER_ENABLED=0` is
   the critical fix. Without it, `rocprofiler-register` tries to read
   `/sys/class/kfd/...`, which doesn't exist in WSL2, and aborts during
   `torch.cuda.device_count()`. With it, PyTorch enumerates the 780M normally
   (`is_available() == True`).

3. **GFX version shim** — `HSA_OVERRIDE_GFX_VERSION=11.0.0` makes the 780M
   (`gfx1103`) reuse `gfx1100` rocBLAS Tensile kernels, since ROCm 7.2 ships
   none for `gfx1103`.

4. **Pin the 2023 Gradio web stack** — Gradio 3.41.2 breaks on 2026
   FastAPI/Pydantic, so pinning pydantic 1.10.13, fastapi 0.99.1, starlette
   0.27.0, uvicorn 0.23.2, and python-multipart 0.0.6 keeps the UI starting.

5. **Tell Fooocus it's on an iGPU** — Fooocus only auto-enables PyTorch SDPA/FP16
   VAE for NVIDIA/Intel. Passing `--attention-pytorch --vae-in-fp16
   --always-low-vram --disable-xformers` (and `--disable-xformers` for the missing
   xformers) makes it use the GPU efficiently instead of falling back to slow
   CPU/sub-quadratic paths and misjudging the 25 GB "VRAM" as real.

Together these turn a setup where PyTorch reported zero GPUs into one where
`Device: cuda:0 AMD Radeon 780M Graphics` and generation proceeds.

## 8. Notes

- Verified end-to-end on the Radeon 780M: `torch.cuda.is_available() == True`,
  device `AMD Radeon 780M Graphics`, and the script reaches model download /
  generation.
- `rocm-smi` does not work in WSL (no `amdgpu` module); use `rocminfo` or PyTorch.
- Fooocus still reports `is_nvidia() == False` on ROCm, so the flags above are
  required to get SDPA and FP16 VAE.
PY
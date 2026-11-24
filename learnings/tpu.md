# Google TPU port learnings

Both commands in `demo.sh` now run on a Google TPU VM. The TPU device is
auto-detected with `torch.accelerator`; set `CLI_AI_BACKEND=tpu` to force it, or
`CLI_AI_BACKEND=cuda|rocm|cpu` to force the original path. No changes to
`demo.sh` are needed: relative input/output paths still resolve against the
folder the CLI was launched from.

## Environment

- Native PyTorch TPU backend: `torch 2.14.1+cpu`, `torch_tpu 0.2.0`, Python 3.13,
  one TPU device with ~16 GiB HBM.
- Detection command:
  `python -c "import torch; print(torch.accelerator.is_available(), torch.accelerator.current_accelerator())"`
  should print `True tpu`.
- No `torch_xla`, no replacement sampler, no model surgery. Both projects run
  their real pipelines on `torch.device("tpu")`.
- Core policy: bf16 compute everywhere, PyTorch SDPA attention, no
  xformers/flash-attn/sage-attn/split kernels, and guard every `torch.cuda.*`
  call with a TPU-aware fallback.

## Fooocus

Basis: [portpilot `fooocus/Google_TPU`](https://github.com/emchan/portpilot/tree/main/fooocus/Google_TPU).

- Pin upstream to tag `v2.5.5`; clone into `~/.cache/fooocus-src/Fooocus-TPU`
  so the patch never touches a CUDA/ROCm checkout.
- Vendor the reference patch as `packages/fooocus/fooocus_tpu_patch.diff` and
  apply it idempotently (`git apply --check`, then a marker check in
  `ldm_patched/modules/model_management.py`).
- The patch makes `model_management`, `model_sampling`, `attention`,
  `sub_quadratic_attention`, `patch`, `patch_clip`, `core`, `anisotropic` and
  `launch` TPU-aware: bf16 UNet/VAE/text encoders, SDPA, no xformers, guarded
  CUDA helpers, NumPy↔tensor and bf16→fp32 conversions, CLIP checkpoint fixes.
- Do **not** install upstream `requirements_versions.txt`: its CUDA-era NumPy
  and torch pins break `torch_tpu`. The patched `launch.py` skips that install;
  `fooocus-cli.py` installs only the small set in
  `packages/fooocus/requirements_tpu.txt` when something is missing, and adds
  the NumPy 1.x aliases (`np.bool8`, `np.float_`, `np.obj2sctype`) Gradio 3.41
  expects but NumPy 2 removed.
- Pass `--disable-xformers --attention-pytorch --always-normal-vram`; keep the
  ROCm-only `--vae-in-fp16` / low-VRAM policy out of the TPU path.
- Import `launch` **before** `webui`: upstream `download_models()` runs at
  import time in `launch.py`, while `webui` imports `async_worker`, which starts
  the model-loading thread immediately. Importing `webui` first makes a fresh
  install fail with a missing checkpoint because the worker starts before the
  download finishes.
- Validated: real 30-step SDXL generation on TPU, `Device: tpu (PyTorch TPU
  backend)`, `VAE dtype: torch.bfloat16`, `Using PyTorch SDPA cross attention
  for TPU`, output written to `--output` path.

## FramePack

portpilot has no FramePack TPU patch, so an equivalent patch was written using
the same principles: `packages/framepack/framepack_tpu_patch.diff`, applied
idempotently to `~/.cache/framepack-src/FramePack-TPU`.

- `diffusers_helper/memory.py`: select `torch.device("tpu")`, dispatch
  `empty_cache()` to `torch.accelerator.memory.empty_cache()`, and compute free
  HBM via `torch.accelerator.memory.get_memory_info()` instead of
  `torch.cuda.mem_get_info()`.
- `diffusers_helper/models/hunyuan_video_packed.py`: skip the CUDA SDPA
  backend query and xformers/flash/sage imports on TPU, create `cu_seqlens` on
  the text-mask device, and add a TPU path to `attn_varlen_func` that splits the
  flattened `(B*L, H, C)` sequence at `cu_seqlens` boundaries and runs
  `F.scaled_dot_product_attention` per segment. (Batch size 1 uses the existing
  direct-SDPA path, so this mainly matters for batched inference.)
- `diffusers_helper/utils.py`: make `print_free_mem()` TPU-aware.
- CLI: skip the Python 3.11 conda/venv bootstrap and CUDA 12.6 extra index;
  install missing extras from `packages/framepack/requirements_tpu.txt` (no
  CUDA wheels). Load models in bf16 instead of fp16.
- Newer `diffusers` removed `diffusers.models.normalization.LayerNorm`, which
  `diffusers_helper/dit_common.py` imports. `framepack-cli.py` restores the
  alias to `torch.nn.LayerNorm` before importing `diffusers_helper`.
- 16 GiB HBM cannot hold the transformer, both text encoders, VAE and image
  encoder simultaneously. Keep FramePack's dynamic weight swapping enabled:
  the transformer and LLaMA text encoder are swap-installed, and the smaller
  models move to the TPU one at a time with `load_model_as_complete()`. Without
  this, `load_models()` fails with `RuntimeBufferAllocationFailure` when it
  tries to move every model to the TPU at once.
- Swapping is **slow on TPU**, in addition to the slow first-run checkpoint
  load. `DynamicSwapInstaller.hacked_get_attr()` moves each parameter tensor to
  the TPU individually, so `move_model_to_device_with_memory_preservation()` at
  each generation stage performs thousands of small host→device copies. Expect
  multi-minute pauses (with rising host RSS and ~25–30% CPU) before `Text
  encoding ...` / `Start sampling ...` appear; this is progress, not a hang.
- Two avoidable copies were removed from the swap path. `hasattr(m, "weight")`
  in `move_model_to_device_with_memory_preservation()` and
  `offload_model_from_device_for_memory_preservation()` invoked
  `hacked_get_attr()` and copied every weight to the TPU merely to test for the
  attribute; those checks now read `m.__dict__["_parameters"]` directly.
  `hacked_get_attr()` also returns the stored parameter unchanged when it is
  already on the requested device, avoiding a redundant Parameter rewrap/copy
  on every forward access. Both fixes are in the regenerated
  `packages/framepack/framepack_tpu_patch.diff`. Because patch application is
  idempotent via a marker in `memory.py`, an existing patched cache keeps its
  old code until the cache is reset and repatched (or edited in place).
- Redirected stdout is block-buffered, so stage messages (`Text encoding ...`,
  `Start sampling ...`) can appear much later than the stage itself. To confirm
  a stall is the staged swap and not deadlock, use `py-spy dump --pid <pid>`.
  A healthy run shows the main thread in
  `DynamicSwapInstaller._install_module.<locals>.hacked_get_attr` called from
  `move_model_to_device_with_memory_preservation()` in `diffusers_helper/memory.py`;
  `run_generation()` at the transformer move means text/VAE/CLIP stages already
  finished and the first sampling section is loading the transformer.
- After `Moving DynamicSwap_HunyuanVideoTransformer3DModelPacked to tpu ...`
  finishes, the first denoising step can be slow again while the TPU compiles
  the attention path. `py-spy` then shows `hacked_get_attr` reached from
  `Linear.forward` inside `HunyuanAttnProcessorFlashAttnDouble.__call__`, not
  from `move_model_to_device_with_memory_preservation()`. That means the
  weights are already staged and sampling is actually running; the later steps
  reuse the compiled kernels.
- `torch.accelerator.memory.empty_cache()` does not reclaim allocator caching
  the way CUDA's `empty_cache()` does, so low-VRAM thresholds are approximate.

## General takeaways

- Prefer a separate source cache per backend over patching a shared checkout.
- Pin the upstream tag the patch was written against, and check `git apply`
  before mutating the cache.
- CUDA-era requirement pins are the main portability hazard; install only what
  the runtime actually needs and let the pre-installed torch stack win.
- Startup ordering matters: async model-loading threads can race ahead of
  model downloads on a fresh install.

# cli-ai

This repository ports popular AI projects to a **command-line only interface**, removing the need for Gradio/web UIs. Each project retains its original license; these are wrappers around the upstream cores.

The initial versions target the original CUDA setup. Ports for ROCm and Google TPU are also included; see [learnings/tpu.md](learnings/tpu.md) and [learnings/fooocus-rocm.md](learnings/fooocus-rocm.md) for details. Ports are based on findings from [portpilot](https://github.com/emchan/portpilot); as projects are ported, updates will be propagated here in CLI form.

TPU runs are auto-detected (`torch.accelerator` reporting a `tpu` device) and can be forced with `CLI_AI_BACKEND=tpu`. They keep the pre-installed `torch_tpu` stack instead of creating a CUDA/ROCm environment.

## Packages

- `packages/fooocus/` — Fooocus CLI (`fooocus-cli.py`), with ROCm and TPU ports
- `packages/framepack/` — FramePack CLI (`framepack-cli.py`), with CUDA and TPU ports

## Examples

### Fooocus

Generate an idyllic beachscape with roaring waves and a lighthouse, saving to `beach.png`:

```bash
python packages/fooocus/fooocus-cli.py "idyllic beach scape with roaring waves and a lighthouse" --output beach.png
```

### FramePack + demo

Generate a video from an input image:

```bash
python packages/framepack/framepack-cli.py beach.png "big waves crashing before a storm"
```

On a TPU VM both commands are auto-detected and `demo.sh` runs unchanged:

```bash
bash demo.sh
```

TPU note: the first FramePack run is slow — it must download and read roughly
12 GB of checkpoints. The TPU has only ~16 GiB HBM, which is not enough to hold
FramePack's transformer, text encoders, VAE and image encoder at once, so the
CLI keeps the large models on CPU and dynamically loads/swaps weights onto the
TPU one stage at a time. Later runs are faster once the weights are cached.

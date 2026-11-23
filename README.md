# cli-ai

This repository ports popular AI projects to a **command-line only interface**, removing the need for Gradio/web UIs. Each project retains its original license; these are wrappers around the upstream cores.

The initial versions target the original CUDA setup. Efforts are underway to port to other platforms via [portpilot](https://github.com/emchan/portpilot); as projects are ported, updates will be propagated here in CLI form.

## Packages

- `packages/fooocus/` — Fooocus CLI (`fooocus-cli.py`)
- `packages/framepack/` — FramePack CLI (`framepack-cli.py`)

## Examples

### Fooocus

Generate an idyllic beachscape with roaring waves and a lighthouse, saving to `beach.png`:

```bash
python packages/fooocus/fooocus-cli.py "idyllic beach scape with roaring waves and a lighthouse" --output beach.png
```
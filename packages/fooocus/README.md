# fooocus-cli

Command-line interface for [Fooocus](https://github.com/lllyasviel/Fooocus).

## Usage

```bash
/tmp/fooocus-cli/bin/python fooocus-cli.py "a cute cat" [--negative "..."] [--seed N] [--output path]
```

Output images are written to `outputs/<prompt>-<seed>.png`.

## Environment

- Venv: `/tmp/fooocus-cli`
- Fooocus source is downloaded on demand into `~/.cache/fooocus-src/Fooocus` if not already present (or `/tmp/fooocus-src`).
- Models (including JuggernautXL) are downloaded by Fooocus on first run.

## Steering prompts used for this port

The port was guided by these user requests:

1. "adapt https://github.com/lllyasviel/Fooocus to a cli interface accepting a prompt and producing an output image in outputs/<prompt>-<datestampe>.png, save the script as fooocus-cli.py in @packages/fooocus/. make a new venv in /tmp/fooocus-cli."
2. "is the full model used?" → switched from a converted SDXL-base checkpoint to Fooocus's default JuggernautXL model.
3. "remove the full source from @packages/fooocus/ and download the source on the fly upon first run while setting up the environment, also use the full juggernaut model"
4. "add the seed to the output file name"
5. "remove timestamp from the output file name"
6. "remove the source dependencies in @packages/fooocus/ and download the dependencies on demand"
7. "add a README.md to @packages/fooocus/ and explain the prompts used to steer this port"

## Notes

- Fooocus's Gradio UI is not started; `gr.Blocks.launch` is patched to a no-op.
- Generation reuses Fooocus's `AsyncTask` + worker thread, driven to completion by iterating `generate_clicked()`.
- The process exits with `os._exit(0)` after writing the output to avoid interpreter shutdown crashes from Fooocus's worker threads.
PY
echo done
# stable-audio-3-cli

> **CLI port:** This is a command-line port of the original [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3) project (launcher: [cocktailpeanut/stable-audio-3-small.pinokio](https://github.com/cocktailpeanut/stable-audio-3-small.pinokio)).

Command-line interface for [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3), using the public
[cocktailpeanut](https://github.com/cocktailpeanut/stable-audio-3-small.pinokio) mirrors (no HF_TOKEN required).

## Usage

```bash
python stable-audio-3-cli.py "lo-fi hip hop beat, 90 BPM" --model small-music --duration 30 -o beat.wav
```

## Notes

- On first run a virtual environment is created at `~/cli-ai/stable-audio-cli` and dependencies are installed.
- The upstream `stable_audio_3` package is vendored alongside this script.
- Models are downloaded on first use into `~/cli-ai/hf_downloads`.
- Supports text-to-audio, audio-to-audio (--init-audio), inpainting/continuation, LoRA, chunked decode, etc. (upstream CLI).

## License
This is a command-line port of [Stable Audio 3](https://github.com/Stability-AI/stable-audio-3). Stable Audio 3 retains its original license; see the upstream project for full license terms.
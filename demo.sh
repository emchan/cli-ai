[ -f beach.png ] || python packages/fooocus/fooocus-cli.py "idyllic beach scape with roaring waves and a lighthouse" --output beach.png
python packages/framepack/framepack-cli.py beach.png "big waves crashing before a storm"

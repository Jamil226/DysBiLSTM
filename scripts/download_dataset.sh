#!/usr/bin/env bash
# Download and extract the Kaggle "Dysarthria and Non-Dysarthria Speech Dataset".
# Requires the Kaggle CLI (`pip install kaggle`) and an API token in ~/.kaggle/kaggle.json.
set -euo pipefail

DEST="${1:-data}"
mkdir -p "$DEST"
kaggle datasets download -d poojag718/dysarthria-and-nondysarthria-speech-dataset -p "$DEST" --unzip
echo "Dataset extracted to: $DEST"
echo "Next: python -m dysbilstm.train --data-root \"$DEST\" --models all"

"""Score WAV files with a trained deep model.

Example
-------
    python -m dysbilstm.predict --model results/bilstm/model.keras path/to/a.wav path/to/folder/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import config
from .features import load_audio, mfcc_matrix


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path to a saved .keras model (bilstm, lstm or cnn).")
    p.add_argument("inputs", nargs="+", help="WAV files and/or folders (searched recursively).")
    args = p.parse_args(argv)

    from tensorflow import keras

    model = keras.models.load_model(args.model)
    is_image = len(model.input_shape) == 4

    wavs = []
    for item in map(Path, args.inputs):
        wavs.extend(sorted(item.rglob("*.wav")) if item.is_dir() else [item])
    if not wavs:
        raise SystemExit("No WAV files found.")

    X = np.stack([mfcc_matrix(load_audio(w)) for w in wavs])
    X = X[..., np.newaxis] if is_image else np.transpose(X, (0, 2, 1))
    proba = model.predict(X, verbose=0)[:, 1]

    width = max(len(str(w)) for w in wavs)
    for w, s in zip(wavs, proba):
        print(f"{str(w):<{width}}  {config.CLASS_NAMES[int(s >= 0.5)]:<15}  p(dysarthric)={s:.3f}")


if __name__ == "__main__":
    main()

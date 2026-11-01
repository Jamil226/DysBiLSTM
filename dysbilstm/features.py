"""Acoustic feature extraction.

Two representations are used (manuscript Section III-C):

* ``mfcc``  – a per-utterance-normalised 40 x 94 MFCC matrix
  (n_fft=1024, hop=512). Used by the BiLSTM, LSTM, CNN and, flattened
  to 3,760-D, by DT, RF, SVM and NB-flat.
* ``stats`` – mean and standard deviation of 40 MFCC + delta + delta-delta
  coefficients over time (240-D, librosa default STFT). Used by LR and
  NB-stat.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from . import config
from .data import Record

log = logging.getLogger(__name__)

FEATURE_TYPES = ("mfcc", "stats")


def load_audio(path: str | Path) -> np.ndarray:
    """Load as 16 kHz mono and pad / truncate to exactly 3 s."""
    y, _ = librosa.load(str(path), sr=config.SAMPLE_RATE, mono=True)
    if len(y) < config.N_SAMPLES:
        y = np.pad(y, (0, config.N_SAMPLES - len(y)))
    return y[: config.N_SAMPLES]


def mfcc_matrix(y: np.ndarray) -> np.ndarray:
    """Normalised MFCC matrix of shape (N_MFCC, N_FRAMES)."""
    mfcc = librosa.feature.mfcc(y=y, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC,
                                n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    return (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)


def mfcc_statistics(y: np.ndarray) -> np.ndarray:
    """240-D vector of per-coefficient mean and std of MFCC, delta and delta-delta."""
    mfcc = librosa.feature.mfcc(y=y, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC)
    feat = np.vstack([mfcc,
                      librosa.feature.delta(mfcc),
                      librosa.feature.delta(mfcc, order=2)])
    return np.hstack([feat.mean(axis=1), feat.std(axis=1)])


_EXTRACTORS = {"mfcc": mfcc_matrix, "stats": mfcc_statistics}


def _cache_key(records: list[Record], kind: str) -> str:
    h = hashlib.sha1(kind.encode())
    for r in records:
        h.update(str(r.path).encode())
    return h.hexdigest()[:12]


def extract(records: list[Record], kind: str, cache_dir: str | Path | None = None):
    """Extract features for every decodable record.

    Returns ``(X, kept)`` where ``kept`` are the indices into ``records``
    that decoded successfully (two TORGO F01 files in the Kaggle release
    are corrupt and are skipped, as in the manuscript).
    """
    if kind not in _EXTRACTORS:
        raise ValueError(f"Unknown feature type {kind!r}; choose from {FEATURE_TYPES}")

    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / f"features_{kind}_{_cache_key(records, kind)}.npz"
        if cache.exists():
            log.info("Loading cached %s features from %s", kind, cache)
            data = np.load(cache)
            return data["X"], data["kept"]

    feats, kept = [], []
    for i, rec in enumerate(tqdm(records, desc=f"Extracting {kind} features")):
        try:
            y = load_audio(rec.path)
        except Exception as exc:  # corrupt / undecodable WAV
            log.warning("Skipping %s: %s", rec.path, exc)
            continue
        feats.append(_EXTRACTORS[kind](y))
        kept.append(i)

    X, kept = np.asarray(feats, dtype=np.float32), np.asarray(kept)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, kept=kept)
    return X, kept

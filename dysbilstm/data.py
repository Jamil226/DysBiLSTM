"""Dataset discovery and stratified partitioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from . import config


@dataclass
class Record:
    path: Path
    label: int    # 0 = control, 1 = dysarthric
    speaker: str  # TORGO speaker ID, e.g. "F01", "MC02"


def resolve_dataset_root(path: str | Path) -> Path:
    """Accept the extraction folder or any parent of ``.../Dataset``."""
    path = Path(path).expanduser().resolve()
    candidates = [
        path,
        path / "Dataset",
        path / "Dysarthria and Non Dysarthria" / "Dataset",
    ]
    for candidate in candidates:
        if all((candidate / d).is_dir() for d in config.DYSARTHRIC_DIRS + config.NON_DYSARTHRIC_DIRS):
            return candidate
    raise FileNotFoundError(
        f"Could not find the four class folders under {path}.\n"
        f"Expected {config.DYSARTHRIC_DIRS + config.NON_DYSARTHRIC_DIRS} inside a 'Dataset' "
        "directory. See README.md -> Dataset for the expected layout."
    )


def _speaker_id(wav: Path, class_dir: Path) -> str:
    # Layout: <class_dir>/<speaker>/<session>/wav_*/<file>.wav
    return wav.relative_to(class_dir).parts[0]


def collect_records(root: str | Path) -> list[Record]:
    """Walk the four class folders and return one record per WAV file.

    Files are sorted so that the order (and hence the random split) is
    identical on every operating system and file system.
    """
    root = resolve_dataset_root(root)
    records = []
    for dirs, label in ((config.DYSARTHRIC_DIRS, 1), (config.NON_DYSARTHRIC_DIRS, 0)):
        for d in dirs:
            class_dir = root / d
            for wav in sorted(class_dir.rglob("*.wav")):
                records.append(Record(wav, label, _speaker_id(wav, class_dir)))
    return records


def stratified_split(labels: np.ndarray, seed: int = config.SEED):
    """Return train/val/test index arrays for a stratified 80/10/10 split.

    The test set is held out first, then 11.11 % of the remainder becomes
    the validation set, exactly as in the original experiments.
    """
    idx = np.arange(len(labels))
    idx_trval, idx_test = train_test_split(
        idx, test_size=config.TEST_SIZE, stratify=labels, random_state=seed)
    idx_train, idx_val = train_test_split(
        idx_trval, test_size=config.VAL_SIZE, stratify=labels[idx_trval], random_state=seed)
    return idx_train, idx_val, idx_test

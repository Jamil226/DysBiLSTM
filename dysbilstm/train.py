"""Train and evaluate one or more models on identical stratified partitions.

Example
-------
    python -m dysbilstm.train --data-root data/ --models bilstm cnn lr
    python -m dysbilstm.train --data-root data/ --models all
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from . import config, plots
from .data import collect_records, stratified_split
from .features import extract
from .metrics import evaluate
from .models import ALL_MODELS, CLASSICAL_MODELS, DEEP_MODEL_NAMES

log = logging.getLogger("dysbilstm")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True,
                   help="Folder containing the extracted Kaggle dataset (or its 'Dataset' subfolder).")
    p.add_argument("--models", nargs="+", default=["bilstm"],
                   help=f"Models to run, or 'all'. Choices: {', '.join(ALL_MODELS)}")
    p.add_argument("--output-dir", default="results", help="Where metrics, figures and models are written.")
    p.add_argument("--cache-dir", default=".cache", help="Feature cache directory ('' disables caching).")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args(argv)


def _feature_kind(model: str) -> str:
    return "mfcc" if model in DEEP_MODEL_NAMES else CLASSICAL_MODELS[model][1]


def run_deep(name, X, y, splits, seed, out_dir):
    from tensorflow import keras
    from tensorflow.keras import callbacks

    from .models.deep import DEEP_MODELS

    build, cfg, layout = DEEP_MODELS[name]
    # (N, 40, 94) -> (N, 94, 40) for recurrent models, (N, 40, 94, 1) for the CNN
    X = np.transpose(X, (0, 2, 1)) if layout == "sequence" else X[..., np.newaxis]
    tr, va, te = splits

    keras.utils.set_random_seed(seed)
    model = build(input_shape=X.shape[1:])
    model.compile(optimizer=keras.optimizers.Adam(cfg.learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    log.info("%s: %s parameters", name, f"{model.count_params():,}")

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y[tr])
    history = model.fit(
        X[tr], y[tr],
        validation_data=(X[va], y[va]),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        class_weight={0: weights[0], 1: weights[1]},
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=cfg.early_stop_patience,
                                    restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=cfg.lr_factor,
                                        patience=cfg.lr_patience, min_lr=cfg.min_lr, verbose=1),
        ],
        verbose=2,
    ).history

    model.save(out_dir / "model.keras")
    hist = {k: [float(v) for v in vals] for k, vals in history.items()}
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    plots.plot_history(hist, out_dir / "training_curves.png", f"{name.upper()} training history")

    val_acc = float(np.mean(model.predict(X[va], verbose=0).argmax(1) == y[va]))
    proba = model.predict(X[te], verbose=0)[:, 1]
    extra = {"parameters": int(model.count_params()), "epochs_run": len(hist["loss"])}
    return proba, val_acc, extra


def run_classical(name, X, y, splits, seed, out_dir):
    import joblib

    factory, _ = CLASSICAL_MODELS[name]
    tr, va, te = splits
    X = X.reshape(len(X), -1)
    model = factory()
    model.fit(X[tr], y[tr])
    joblib.dump(model, out_dir / "model.joblib")
    val_acc = float(model.score(X[va], y[va]))
    proba = model.predict_proba(X[te])[:, 1]
    extra = {"input_dim": int(X.shape[1])}
    if "pca" in model.named_steps:
        extra["pca_components"] = int(model.named_steps["pca"].n_components_)
    return proba, val_acc, extra


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")

    models = list(ALL_MODELS) if args.models == ["all"] else args.models
    unknown = set(models) - set(ALL_MODELS)
    if unknown:
        raise SystemExit(f"Unknown model(s): {', '.join(sorted(unknown))}. Choices: {', '.join(ALL_MODELS)}")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or None

    records = collect_records(args.data_root)
    log.info("Found %d WAV files (%d dysarthric, %d control)", len(records),
             sum(r.label for r in records), sum(1 - r.label for r in records))

    # Extract every representation that is needed. Decoding failures are
    # identical for both representations, so all models share one index set.
    features, kept = {}, None
    for kind in sorted({_feature_kind(m) for m in models}):
        X, k = extract(records, kind, cache_dir)
        if kept is not None and not np.array_equal(kept, k):
            raise RuntimeError("Feature extractors kept different files; partitions would not match.")
        features[kind], kept = X, k

    kept_records = [records[i] for i in kept]
    y = np.array([r.label for r in kept_records])
    splits = stratified_split(y, args.seed)
    log.info("Usable utterances: %d | train %d / val %d / test %d",
             len(y), *(len(s) for s in splits))
    (out_root / "splits.json").write_text(json.dumps(
        {name: [str(kept_records[i].path) for i in idx] for name, idx in zip(("train", "val", "test"), splits)},
        indent=1))

    summary = []
    for name in models:
        log.info("──── %s ────", name)
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        X = features[_feature_kind(name)]
        runner = run_deep if name in DEEP_MODEL_NAMES else run_classical

        t0 = time.time()
        proba, val_acc, extra = runner(name, X, y, splits, args.seed, out_dir)
        te = splits[2]
        pred = (proba >= 0.5).astype(int)
        metrics = {"model": name, **evaluate(y[te], pred, proba), "val_accuracy": val_acc,
                   "train_seconds": round(time.time() - t0, 1), **extra}
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        # Per-utterance predictions enable paired tests (e.g. McNemar) later.
        with open(out_dir / "predictions.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "speaker", "y_true", "y_pred", "p_dysarthric"])
            for i, p, s in zip(te, pred, proba):
                w.writerow([kept_records[i].path, kept_records[i].speaker, y[i], p, f"{s:.6f}"])

        plots.plot_confusion_matrix(y[te], pred, out_dir / "confusion_matrix.png", f"{name.upper()} – test set")
        plots.plot_roc(y[te], proba, metrics["auc"], out_dir / "roc_curve.png", f"{name.upper()} – ROC")
        log.info("%s  acc=%.2f%%  sens=%.2f%%  spec=%.2f%%  AUC=%.3f", name, 100 * metrics["accuracy"],
                 100 * metrics["sensitivity"], 100 * metrics["specificity"], metrics["auc"])
        summary.append(metrics)

    cols = ["model", "accuracy", "precision", "sensitivity", "specificity", "macro_f1", "auc", "val_accuracy"]
    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["ci95_low", "ci95_high"])
        for m in summary:
            w.writerow([m["model"]] + [f"{m[c]:.4f}" for c in cols[1:]] + [f"{v:.4f}" for v in m["accuracy_ci95"]])

    print("\n| Model | Acc. (%) | 95% CI | Sens. | Spec. | Macro F1 | AUC |")
    print("|---|---|---|---|---|---|---|")
    for m in summary:
        lo, hi = m["accuracy_ci95"]
        print(f"| {m['model']} | {100 * m['accuracy']:.2f} | [{100 * lo:.2f}, {100 * hi:.2f}] | "
              f"{m['sensitivity']:.3f} | {m['specificity']:.3f} | {m['macro_f1']:.3f} | {m['auc']:.3f} |")
    print(f"\nResults written to {out_root.resolve()}")


if __name__ == "__main__":
    main()

"""Model zoo. Deep models are imported lazily so that the classical
baselines can run without TensorFlow installed."""

from .classical import CLASSICAL_MODELS

DEEP_MODEL_NAMES = ("bilstm", "lstm", "cnn")
ALL_MODELS = DEEP_MODEL_NAMES + tuple(CLASSICAL_MODELS)

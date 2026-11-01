"""Keras deep-learning models and their training configurations (Table VI)."""

from __future__ import annotations

from dataclasses import dataclass

from tensorflow import keras
from tensorflow.keras import layers


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float
    epochs: int
    early_stop_patience: int
    lr_factor: float = 0.1
    lr_patience: int = 3
    min_lr: float = 0.0
    batch_size: int = 32


def build_bilstm(input_shape=(94, 40), num_classes: int = 2) -> keras.Model:
    """Proposed two-layer BiLSTM (347,330 parameters)."""
    return keras.Sequential([
        keras.Input(shape=input_shape, name="mfcc_sequence"),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name="DysBiLSTM")


def build_lstm(input_shape=(94, 40), num_classes: int = 2) -> keras.Model:
    """Unidirectional counterpart of the BiLSTM (ablation, 140,994 parameters)."""
    return keras.Sequential([
        keras.Input(shape=input_shape, name="mfcc_sequence"),
        layers.LSTM(128, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name="UniLSTM")


def build_cnn(input_shape=(40, 94, 1), num_classes: int = 2) -> keras.Model:
    """Three-block 2-D CNN baseline (110,338 parameters)."""
    inp = keras.Input(shape=input_shape, name="mfcc_image")
    x = inp
    for filters, drop in ((32, 0.25), (64, 0.25), (128, 0.30)):
        x = layers.Conv2D(filters, (3, 3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(drop)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.40)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="MFCC_CNN")


# name -> (builder, training config, input layout)
# "sequence" = (frames, coefficients); "image" = (coefficients, frames, 1)
DEEP_MODELS = {
    "bilstm": (build_bilstm, TrainConfig(learning_rate=3e-4, epochs=25, early_stop_patience=6), "sequence"),
    "lstm":   (build_lstm,   TrainConfig(learning_rate=5e-4, epochs=20, early_stop_patience=5), "sequence"),
    "cnn":    (build_cnn,    TrainConfig(learning_rate=1e-3, epochs=20, early_stop_patience=5,
                                         lr_factor=0.5, min_lr=1e-6), "image"),
}

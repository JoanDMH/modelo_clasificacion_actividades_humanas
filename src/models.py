"""Arquitecturas recurrentes (tarea b). Tamaños pequeños por restricción de CPU."""
from tensorflow import keras
from tensorflow.keras import layers

INPUT_SHAPE = (128, 9)
N_CLASSES = 6


def build_lstm(dropout=0.2, input_shape=INPUT_SHAPE, n_classes=N_CLASSES):
    """(A) LSTM apilada 64 -> 32."""
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(32),
        layers.Dropout(dropout),
        layers.Dense(n_classes, activation="softmax"),
    ], name="lstm")


def build_conv_lstm(dropout=0.2, input_shape=INPUT_SHAPE, n_classes=N_CLASSES):
    """(B) 2x Conv1D(64, k=5) + MaxPool(2) -> LSTM(64)."""
    return keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Conv1D(64, 5, padding="same", activation="relu"),
        layers.Conv1D(64, 5, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Dropout(dropout),
        layers.LSTM(64),
        layers.Dropout(dropout),
        layers.Dense(n_classes, activation="softmax"),
    ], name="conv_lstm")


ARCHITECTURES = {"lstm": build_lstm, "conv_lstm": build_conv_lstm}


def build_model(arch: str, lr: float = 1e-3, dropout: float = 0.2):
    model = ARCHITECTURES[arch](dropout=dropout)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

"""Carga de las señales crudas (Inertial Signals) del UCI HAR Dataset.

Uso compartido por las fases 2 (EDA), 3 (preparación) y 6 (interfaz).
No normaliza ni particiona: eso pertenece a la Fase 3.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Raíz del proyecto = carpeta padre de src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "human+activity+recognition+using+smartphones" / "UCI HAR Dataset"

# Orden fijo de los 9 canales (se reutiliza en todo el proyecto)
CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]
FS_HZ = 50          # frecuencia de muestreo [DATO: README.txt]
WINDOW_LEN = 128    # muestras por ventana (2.56 s) [DATO: README.txt]


def load_activity_labels(data_dir: Path = DATA_DIR) -> dict:
    """Devuelve {id: nombre} desde activity_labels.txt."""
    df = pd.read_csv(data_dir / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "name"])
    return dict(zip(df["id"], df["name"]))


def load_split(split: str, data_dir: Path = DATA_DIR):
    """Carga un split ('train' o 'test').

    Returns
    -------
    X : np.ndarray float32, forma (n_ventanas, 128, 9) en el orden CHANNELS
    y : np.ndarray int, etiquetas 1..6
    subjects : np.ndarray int, id de sujeto 1..30
    """
    assert split in ("train", "test"), split
    sig_dir = data_dir / split / "Inertial Signals"
    X = np.stack(
        [np.loadtxt(sig_dir / f"{ch}_{split}.txt", dtype=np.float32) for ch in CHANNELS],
        axis=-1,
    )
    y = np.loadtxt(data_dir / split / f"y_{split}.txt", dtype=int)
    subjects = np.loadtxt(data_dir / split / f"subject_{split}.txt", dtype=int)
    assert X.shape[0] == y.shape[0] == subjects.shape[0], "n de filas inconsistente"
    return X, y, subjects


def load_or_cache(split: str, cache_dir: Path = PROJECT_ROOT / "data_cache"):
    """Igual que load_split pero guarda un .npz para no re-parsear los .txt (lento)."""
    cache_dir.mkdir(exist_ok=True)
    f = cache_dir / f"{split}_raw.npz"
    if f.exists():
        d = np.load(f)
        return d["X"], d["y"], d["subjects"]
    X, y, s = load_split(split)
    np.savez_compressed(f, X=X, y=y, subjects=s)
    return X, y, s

"""Configuración centralizada del proyecto (semillas, rutas, W&B, partición)."""
import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# Particiones
HOLDOUT_N_SPLITS = 5      # 1 de 5 grupos de sujetos -> validación hold-out (tarea c)
CV_N_SPLITS = 2           # CV exigida por el enunciado (tarea b)
SPLITS_FILE = PROJECT_ROOT / "fase_3_preparacion_datos" / "particiones.json"

# Weights & Biases
WANDB_ENTITY = "joan-martinez-universidad-de-los-llanos"
WANDB_PROJECT = "clasificacion-actividades-humanas"

# Artefactos
MODELS_DIR = PROJECT_ROOT / "models"


def set_global_seed(seed: int = SEED) -> None:
    """Fija semillas de random, numpy y (si está instalado) TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # resultados deterministas en CPU
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    except ImportError:
        pass

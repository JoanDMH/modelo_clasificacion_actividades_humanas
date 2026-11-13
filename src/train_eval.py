"""Entrenamiento, evaluación y registro (W&B + registro_experimentos.csv)."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from config import PROJECT_ROOT

LOG_FILE = PROJECT_ROOT / "registro_experimentos.csv"
LOG_COLS = ["ID", "fecha", "fase", "arquitectura", "hiperparametros", "semilla", "particion_fold",
            "exactitud", "macro_f1", "run_id_wandb", "url_wandb", "version_codigo", "observaciones"]


def code_version() -> str:
    """Hash corto del contenido de src/*.py (sustituto de un commit de git)."""
    h = hashlib.sha1()
    for f in sorted((PROJECT_ROOT / "src").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


def compute_metrics(y_true, y_pred, class_names):
    """Exactitud, P/R/F1 por clase y macro, matriz de confusión. Etiquetas 0..5."""
    labels = list(range(len(class_names)))
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro",
                                                    zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(pm), "macro_recall": float(rm), "macro_f1": float(fm),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "support": int(sup[i])} for i, c in enumerate(class_names)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


class WandbEpochLogger:
    """Callback Keras mínimo: registra todas las métricas de cada época en W&B."""

    def __new__(cls, run):
        from tensorflow import keras

        class _CB(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                run.log({"epoch": epoch + 1, **{k: float(v) for k, v in (logs or {}).items()}})
        return _CB()


def append_log(row: dict):
    new = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_COLS})


def next_id(prefix="EXP") -> str:
    n = 0
    if LOG_FILE.exists():
        with open(LOG_FILE, encoding="utf-8") as fh:
            n = sum(1 for r in csv.DictReader(fh) if r.get("ID", "").startswith(prefix))
    return f"{prefix}-{n + 1:03d}"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def read_json(path: Path):
    """Lee JSON en UTF-8; si falla (archivos antiguos escritos en cp1252 por Windows), reintenta."""
    raw = Path(path).read_bytes()
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        return json.loads(raw.decode("cp1252"))


def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

"""Fase 3: partición por sujeto sin fuga y normalización por canal.

Esquema (ver fase_3_preparacion_datos/01_preparacion_datos.md):
  train oficial (21 sujetos)
    ├── HOLD-OUT de validación (~1/5 de sujetos)  -> tarea (c)
    └── DEV (resto)                                -> CV de 2 particiones, tarea (b);
                                                      entrenamiento del modelo final
  test oficial (9 sujetos)                         -> tareas (d) y (f), solo al final
"""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from config import CV_N_SPLITS, HOLDOUT_N_SPLITS, SEED, SPLITS_FILE


# ---------------------------------------------------------------- particiones
def assert_disjoint_subjects(subj_a, subj_b, name_a="A", name_b="B"):
    """Falla si algún sujeto aparece en ambos conjuntos (fuga entre sujetos)."""
    inter = set(np.unique(subj_a)) & set(np.unique(subj_b))
    assert not inter, f"FUGA: sujetos {sorted(inter)} presentes en {name_a} y {name_b}"


def make_holdout(y, subjects, n_splits=HOLDOUT_N_SPLITS, seed=SEED):
    """Separa un grupo de sujetos como validación hold-out (estratificado por clase).

    Returns (idx_dev, idx_holdout) sobre los índices del train oficial.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    idx_dev, idx_hold = next(sgkf.split(np.zeros(len(y)), y, groups=subjects))
    assert_disjoint_subjects(subjects[idx_dev], subjects[idx_hold], "dev", "holdout")
    return idx_dev, idx_hold


def make_cv_folds(y, subjects, n_splits=CV_N_SPLITS, seed=SEED):
    """Folds de CV estratificados por sujeto. Devuelve lista de (idx_tr, idx_va) locales."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for k, (tr, va) in enumerate(sgkf.split(np.zeros(len(y)), y, groups=subjects)):
        assert_disjoint_subjects(subjects[tr], subjects[va], f"fold{k}-train", f"fold{k}-val")
        folds.append((tr, va))
    return folds


def build_and_save_splits(y_train, s_train, s_test, path: Path = SPLITS_FILE, seed=SEED):
    """Crea hold-out + folds de CV y los guarda como listas de SUJETOS (trazable)."""
    assert_disjoint_subjects(s_train, s_test, "train oficial", "test oficial")
    idx_dev, idx_hold = make_holdout(y_train, s_train, seed=seed)
    folds = make_cv_folds(y_train[idx_dev], s_train[idx_dev], seed=seed)
    out = {
        "seed": seed,
        "holdout_subjects": sorted(map(int, np.unique(s_train[idx_hold]))),
        "dev_subjects": sorted(map(int, np.unique(s_train[idx_dev]))),
        "test_subjects": sorted(map(int, np.unique(s_test))),
        "cv_folds": [
            {"fold": k,
             "train_subjects": sorted(map(int, np.unique(s_train[idx_dev][tr]))),
             "val_subjects": sorted(map(int, np.unique(s_train[idx_dev][va])))}
            for k, (tr, va) in enumerate(folds)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out


def mask_by_subjects(subjects, subject_list):
    """Índices booleanos de las ventanas cuyos sujetos están en subject_list."""
    return np.isin(subjects, subject_list)


# ------------------------------------------------------------- normalización
class ChannelStandardizer:
    """z-score por canal. fit() SOLO con datos de entrenamiento (del fold)."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        # X: (n, 128, 9) -> estadísticos sobre ventanas y pasos de tiempo
        self.mean_ = X.mean(axis=(0, 1)).astype(np.float32)
        self.std_ = X.std(axis=(0, 1)).astype(np.float32)
        assert np.all(self.std_ > 0), "canal con desviación 0"
        return self

    def transform(self, X):
        assert self.mean_ is not None, "llamar fit() primero"
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mean": self.mean_.tolist(), "std": self.std_.tolist()}, indent=2))

    @classmethod
    def load(cls, path: Path):
        d = json.loads(Path(path).read_text())
        obj = cls()
        obj.mean_ = np.array(d["mean"], dtype=np.float32)
        obj.std_ = np.array(d["std"], dtype=np.float32)
        return obj

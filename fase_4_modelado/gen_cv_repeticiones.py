"""Genera fase_4_modelado/cv_repeticiones.json: folds de la VC (2 particiones) sobre DEV para las semillas 42, 43 y 44.

El archivo incluido en el repositorio se generó con scikit-learn 1.6.1 y está CONGELADO: run_cv.py lo lee
en lugar de recalcularlo, porque el orden aleatorio de StratifiedGroupKFold puede cambiar entre versiones.
Este script documenta cómo se obtuvo; si se ejecuta con otra versión de scikit-learn, compare el resultado
con el archivo congelado antes de reemplazarlo.

Uso (raíz del proyecto):  python fase_4_modelado/gen_cv_repeticiones.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_io import load_or_cache  # noqa: E402
from prep import make_cv_folds, mask_by_subjects  # noqa: E402

sp = json.loads((ROOT / "fase_3_preparacion_datos" / "particiones.json").read_text(encoding="utf-8"))
X, y, s = load_or_cache("train")
m = mask_by_subjects(s, sp["dev_subjects"]); yd, sd = y[m], s[m]
out = {"dev_subjects": sp["dev_subjects"], "holdout_subjects": sp["holdout_subjects"], "repeticiones": []}
for seed in [42, 43, 44]:
    folds = make_cv_folds(yd, sd, seed=seed)
    out["repeticiones"].append({"seed": seed, "folds": [
        {"fold": k, "train_subjects": sorted(map(int, np.unique(sd[a]))), "val_subjects": sorted(map(int, np.unique(sd[b])))}
        for k, (a, b) in enumerate(folds)]})
assert out["repeticiones"][0]["folds"] == sp["cv_folds"], "la semilla 42 debe coincidir con la Fase 3"
dest = ROOT / "fase_4_modelado" / ("cv_repeticiones.json" if "--escribir" in sys.argv else "cv_repeticiones_regenerado.json")
dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("Escrito:", dest)

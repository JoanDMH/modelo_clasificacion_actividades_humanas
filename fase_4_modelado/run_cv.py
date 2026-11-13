"""Fase 4 – Tarea (b): CV estratificada por sujeto (2 particiones) x 3 semillas x 2 arquitecturas.

Uso (desde la raíz del proyecto, con .venv activo):
    python fase_4_modelado/run_cv.py                 # corrida completa (12 runs)
    python fase_4_modelado/run_cv.py --smoke         # prueba rápida: 1 época, 1 semilla, sin W&B online

Particiones: se LEEN de fase_4_modelado/cv_repeticiones.json (no se recalculan).
Cada (arquitectura, semilla, fold) = 1 run de W&B, group = arquitectura.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import wandb  # noqa: E402
from tensorflow import keras  # noqa: E402

from config import WANDB_ENTITY, WANDB_PROJECT, set_global_seed  # noqa: E402
from data_io import load_activity_labels, load_or_cache  # noqa: E402
from models import build_model  # noqa: E402
from prep import ChannelStandardizer, assert_disjoint_subjects, mask_by_subjects  # noqa: E402
from train_eval import (WandbEpochLogger, append_log, code_version, compute_metrics,  # noqa: E402
                        dump_json, next_id, now)

OUT = ROOT / "fase_4_modelado"
HP = {"epochs": 50, "batch_size": 64, "lr": 1e-3, "dropout": 0.2,
      "early_stopping_patience": 8, "monitor": "val_loss"}


def run_one(arch, seed, fold, X, y, s, tr_subj, va_subj, names, hp, mode):
    set_global_seed(seed)
    mtr, mva = mask_by_subjects(s, tr_subj), mask_by_subjects(s, va_subj)
    assert_disjoint_subjects(s[mtr], s[mva], "train", "val")
    scaler = ChannelStandardizer().fit(X[mtr])                       # solo train del fold
    Xtr, Xva = scaler.transform(X[mtr]), scaler.transform(X[mva])
    ytr, yva = y[mtr] - 1, y[mva] - 1                                # 1..6 -> 0..5

    cfg = {"arch": arch, "seed": seed, "fold": fold, **hp, "train_subjects": tr_subj,
           "val_subjects": va_subj, "n_train": int(mtr.sum()), "n_val": int(mva.sum()),
           "code_version": code_version(), "onednn": os.environ.get("TF_ENABLE_ONEDNN_OPTS")}
    sfx = hp.get("suffix", "")
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, group=arch + sfx,
                     job_type="cv" + sfx, name=f"{arch}{sfx}-s{seed}-f{fold}", config=cfg, mode=mode, reinit="finish_previous")

    model = build_model(arch, lr=hp["lr"], dropout=hp["dropout"])
    es = keras.callbacks.EarlyStopping(monitor=hp["monitor"], patience=hp["early_stopping_patience"],
                                       mode="min" if "loss" in hp["monitor"] else "max",
                                       restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=hp["epochs"],
                     batch_size=hp["batch_size"], callbacks=[es, WandbEpochLogger(run)], verbose=2)

    y_pred = model.predict(Xva, verbose=0).argmax(1)
    met = compute_metrics(yva, y_pred, names)
    hm = hist.history[hp["monitor"]]
    best_epoch = int((np.argmin(hm) if "loss" in hp["monitor"] else np.argmax(hm)) + 1)
    run.summary.update({"val_accuracy_final": met["accuracy"], "val_macro_f1_final": met["macro_f1"],
                        "best_epoch": best_epoch, "epochs_run": len(hist.history["loss"]),
                        "n_params": int(model.count_params())})
    run.log({"confusion_matrix": wandb.plot.confusion_matrix(y_true=yva.tolist(), preds=y_pred.tolist(),
                                                            class_names=names)})
    # Artefactos: estadísticos de normalización + modelo del fold
    rdir = OUT / "runs" / f"{arch}{sfx}-s{seed}-f{fold}"
    scaler.save(rdir / "scaler.json"); model.save(rdir / "model.keras"); dump_json(met, rdir / "metrics.json")
    art = wandb.Artifact(f"{arch}{sfx}-s{seed}-f{fold}", type="cv-model")
    art.add_file(str(rdir / "scaler.json")); art.add_file(str(rdir / "model.keras"))
    run.log_artifact(art)

    row = {"arch": arch, "seed": seed, "fold": fold, "accuracy": met["accuracy"], "macro_f1": met["macro_f1"],
           "best_epoch": best_epoch, "epochs_run": len(hist.history["loss"]), "n_params": int(model.count_params()),
           "run_id": run.id, "url": run.url if mode == "online" else ""}
    if mode == "online":
        append_log({"ID": next_id(), "fecha": now(), "fase": "4-CV" + sfx, "arquitectura": arch,
                    "hiperparametros": json.dumps(hp), "semilla": seed, "particion_fold": fold,
                    "exactitud": round(met["accuracy"], 4), "macro_f1": round(met["macro_f1"], 4),
                    "run_id_wandb": run.id, "url_wandb": run.url, "version_codigo": cfg["code_version"],
                    "observaciones": f"best_epoch={best_epoch}; val_subj={va_subj}"})
    run.finish()
    keras.backend.clear_session()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=["lstm", "conv_lstm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--smoke", action="store_true", help="1 época, 1 semilla, W&B offline")
    ap.add_argument("--monitor", default="val_loss", choices=["val_loss", "val_accuracy"],
                    help="métrica de EarlyStopping; val_accuracy = análisis de sensibilidad")
    a = ap.parse_args()

    hp = dict(HP); mode = "online"; seeds = a.seeds
    if a.monitor != HP["monitor"]:
        hp["monitor"] = a.monitor; hp["suffix"] = "-sens"
    if a.smoke:
        hp["epochs"] = 1; mode = "offline"; seeds = seeds[:1]

    X, y, s = load_or_cache("train")
    names = [load_activity_labels()[i] for i in range(1, 7)]
    rep = json.loads((OUT / "cv_repeticiones.json").read_text())
    reps = {r["seed"]: r["folds"] for r in rep["repeticiones"]}
    hold = rep["holdout_subjects"]

    rows = []
    for arch in a.archs:
        for seed in seeds:
            for f in reps[seed]:
                assert not set(f["train_subjects"] + f["val_subjects"]) & set(hold), "holdout en CV"
                print(f"\n=== {arch} | seed {seed} | fold {f['fold']} ===", flush=True)
                rows.append(run_one(arch, seed, f["fold"], X, y, s, f["train_subjects"],
                                    f["val_subjects"], names, hp, mode))

    df = pd.DataFrame(rows)
    tag = ("_smoke" if a.smoke else "") + ("_sens" if hp.get("suffix") else "")
    df.to_csv(OUT / f"resultados_cv{tag}.csv", index=False)
    # Tabla del enunciado: media ± desv sobre todos los (semilla, fold)
    g = df.groupby("arch")
    tab = pd.DataFrame({"n_runs": g.size(),
                        "exactitud_media": g.accuracy.mean(), "exactitud_desv": g.accuracy.std(ddof=1),
                        "macro_f1_media": g.macro_f1.mean(), "macro_f1_desv": g.macro_f1.std(ddof=1),
                        "n_params": g.n_params.first()})
    # Variabilidad entre semillas (media por semilla, luego desv entre semillas)
    tab["desv_entre_semillas"] = df.groupby(["arch", "seed"]).accuracy.mean().groupby("arch").std(ddof=1)
    tab.to_csv(OUT / f"tabla_cv{tag}.csv")
    print("\n", tab.round(4).to_string())


if __name__ == "__main__":
    main()

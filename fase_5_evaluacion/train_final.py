"""Fase 5 – Tarea (c): modelo final (LSTM) entrenado en DEV y evaluado en el HOLD-OUT de validación.

Uso (raíz del proyecto, .venv activo):
    python fase_5_evaluacion/train_final.py            # 3 semillas; la 42 se guarda como modelo desplegable
    python fase_5_evaluacion/train_final.py --smoke    # 1 época, W&B offline, no guarda en models/

Decisiones (ver fase_4_modelado/00_reporte_fase_4.md):
  - Arquitectura LSTM (opción A), 7 épocas fijas (mediana de best_epoch en la CV), sin early stopping:
    DEV completo no tiene validación interna y el HOLD-OUT NO se usa durante el entrenamiento.
  - El TEST oficial NO se toca aquí (eso es la tarea d).
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

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import wandb  # noqa: E402

from config import MODELS_DIR, SPLITS_FILE, WANDB_ENTITY, WANDB_PROJECT, set_global_seed  # noqa: E402
from data_io import CHANNELS, load_activity_labels, load_or_cache  # noqa: E402
from models import build_model  # noqa: E402
from prep import ChannelStandardizer, assert_disjoint_subjects, mask_by_subjects  # noqa: E402
from train_eval import (WandbEpochLogger, append_log, code_version, compute_metrics,  # noqa: E402
                        dump_json, next_id, now)

OUT = ROOT / "fase_5_evaluacion"
ARCH = "lstm"
HP = {"epochs": 7, "batch_size": 64, "lr": 1e-3, "dropout": 0.2, "early_stopping": "none (épocas fijas)"}
DEPLOY_SEED = 42


def plot_cm(cm, names, path, title):
    cm = np.array(cm); cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > .5 else "black")
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Predicha"); ax.set_ylabel("Real"); ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=.046); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    hp = dict(HP); mode = "online"; seeds = a.seeds
    if a.smoke:
        hp["epochs"] = 1; mode = "offline"; seeds = seeds[:1]

    sp = json.loads(SPLITS_FILE.read_text())
    X, y, s = load_or_cache("train")
    names = [load_activity_labels()[i] for i in range(1, 7)]
    m_dev, m_hold = mask_by_subjects(s, sp["dev_subjects"]), mask_by_subjects(s, sp["holdout_subjects"])
    assert_disjoint_subjects(s[m_dev], s[m_hold], "DEV", "HOLD-OUT")

    scaler = ChannelStandardizer().fit(X[m_dev])              # solo DEV
    Xd, Xh = scaler.transform(X[m_dev]), scaler.transform(X[m_hold])
    yd, yh = y[m_dev] - 1, y[m_hold] - 1

    rows = []
    for seed in seeds:
        set_global_seed(seed)
        cfg = {"arch": ARCH, "seed": seed, **hp, "train_subjects": sp["dev_subjects"],
               "eval_subjects": sp["holdout_subjects"], "n_train": int(m_dev.sum()),
               "n_eval": int(m_hold.sum()), "code_version": code_version()}
        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, group="final-lstm", job_type="final",
                         name=f"final-lstm-s{seed}", config=cfg, mode=mode, reinit="finish_previous")
        model = build_model(ARCH, lr=hp["lr"], dropout=hp["dropout"])
        model.fit(Xd, yd, epochs=hp["epochs"], batch_size=hp["batch_size"],
                  callbacks=[WandbEpochLogger(run)], verbose=2)          # sin validation_data: HOLD-OUT intacto
        proba = model.predict(Xh, verbose=0); y_pred = proba.argmax(1)
        met = compute_metrics(yh, y_pred, names)

        sdir = OUT / ("smoke" if a.smoke else "") / f"seed{seed}"
        dump_json(met, sdir / "metricas_holdout.json")
        plot_cm(met["confusion_matrix"], names, sdir / "matriz_confusion_holdout.png",
                f"LSTM final (semilla {seed}) - HOLD-OUT (suj. {sp['holdout_subjects']})")
        pc = pd.DataFrame(met["per_class"]).T; pc.to_csv(sdir / "metricas_por_clase_holdout.csv")
        run.summary.update({"holdout_accuracy": met["accuracy"], "holdout_macro_precision": met["macro_precision"],
                            "holdout_macro_recall": met["macro_recall"], "holdout_macro_f1": met["macro_f1"]})
        run.log({"holdout_confusion_matrix": wandb.plot.confusion_matrix(
                     y_true=yh.tolist(), preds=y_pred.tolist(), class_names=names),
                 "holdout_per_class": wandb.Table(dataframe=pc.reset_index().rename(columns={"index": "clase"}))})

        if seed == DEPLOY_SEED and not a.smoke:                 # artefactos para la interfaz
            MODELS_DIR.mkdir(exist_ok=True)
            model.save(MODELS_DIR / "final_lstm.keras"); scaler.save(MODELS_DIR / "scaler.json")
            dump_json({"labels": names, "channels": CHANNELS, "window_len": 128, "fs_hz": 50,
                       "arch": ARCH, "seed": seed, "hp": hp, "train_subjects": sp["dev_subjects"],
                       "run_id": run.id, "code_version": cfg["code_version"]}, MODELS_DIR / "model_card.json")
            art = wandb.Artifact("final-lstm", type="model")
            for f in ["final_lstm.keras", "scaler.json", "model_card.json"]:
                art.add_file(str(MODELS_DIR / f))
            run.log_artifact(art)

        if mode == "online":
            append_log({"ID": next_id(), "fecha": now(), "fase": "5-final-holdout", "arquitectura": ARCH,
                        "hiperparametros": json.dumps(hp, ensure_ascii=False), "semilla": seed,
                        "particion_fold": "DEV->HOLDOUT", "exactitud": round(met["accuracy"], 4),
                        "macro_f1": round(met["macro_f1"], 4), "run_id_wandb": run.id, "url_wandb": run.url,
                        "version_codigo": cfg["code_version"],
                        "observaciones": "modelo desplegado" if seed == DEPLOY_SEED else ""})
        rows.append({"seed": seed, "accuracy": met["accuracy"], "macro_precision": met["macro_precision"],
                     "macro_recall": met["macro_recall"], "macro_f1": met["macro_f1"], "run_id": run.id})
        run.finish()

    df = pd.DataFrame(rows)
    tag = "_smoke" if a.smoke else ""
    df.to_csv(OUT / f"resultados_holdout{tag}.csv", index=False)
    print("\n", df.round(4).to_string(index=False))
    if len(df) > 1:
        print("\nmedia ± desv:", {c: f"{df[c].mean():.4f} ± {df[c].std(ddof=1):.4f}"
                                  for c in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]})


if __name__ == "__main__":
    main()

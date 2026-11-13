"""Fase 5 – Tarea (d): evaluación ÚNICA del modelo final en el TEST oficial + 5 aciertos / 5 errores.

Uso (raíz del proyecto, .venv activo):
    python fase_5_evaluacion/eval_test.py            # evaluación real (una sola vez)
    python fase_5_evaluacion/eval_test.py --smoke    # prueba del pipeline usando el HOLD-OUT (no toca TEST), W&B offline

Regla de selección de ejemplos (pre-especificada, sin elegir a mano):
  - Correctos: 5 actividades distintas elegidas al azar (semilla 42); 1 ventana bien clasificada al azar de cada una.
  - Incorrectos: los 5 pares (real -> predicha) más frecuentes; 1 ventana al azar de cada par
    (si hay menos de 5 pares, se completa con el par más frecuente).
También exporta ventanas de TEST en CSV (128x9) para probar la interfaz (tarea f).
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
from data_io import CHANNELS, FS_HZ, load_activity_labels, load_or_cache  # noqa: E402
from models import build_model  # noqa: E402
from prep import ChannelStandardizer, mask_by_subjects  # noqa: E402
from train_eval import append_log, code_version, compute_metrics, dump_json, next_id, now, read_json  # noqa: E402

SEED = 42
SHORT = {"WALKING": "WALK", "WALKING_UPSTAIRS": "UP", "WALKING_DOWNSTAIRS": "DOWN",
         "SITTING": "SIT", "STANDING": "STAND", "LAYING": "LAY"}


def load_final_model():
    """Reconstruye la arquitectura y carga los pesos (robusto entre versiones menores de Keras)."""
    card = read_json(MODELS_DIR / "model_card.json")
    model = build_model(card["arch"], lr=card["hp"]["lr"], dropout=card["hp"]["dropout"])
    model.load_weights(MODELS_DIR / "final_lstm.keras")
    return model, ChannelStandardizer.load(MODELS_DIR / "scaler.json"), card


def plot_cm(cm, names, path, title):
    cm = np.array(cm); cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > .5 else "black")
    ax.set_xticks(range(6), names, rotation=35, ha="right", fontsize=8); ax.set_yticks(range(6), names, fontsize=8)
    ax.set_xlabel("Predicha"); ax.set_ylabel("Real"); ax.set_title(title, fontsize=9)
    fig.colorbar(im, fraction=.046); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_example(ax_row, x, proba, names, title):
    """x: ventana cruda (128, 9) en unidades físicas. ax_row: 4 ejes."""
    t = np.arange(x.shape[0]) / FS_HZ
    for k, (lab, off, unit) in enumerate([("body_acc", 0, "g"), ("body_gyro", 3, "rad/s"), ("total_acc", 6, "g")]):
        for j, axn in enumerate("xyz"):
            ax_row[k].plot(t, x[:, off + j], lw=.9, label=axn)
        ax_row[k].set_ylabel(f"{lab} [{unit}]", fontsize=7); ax_row[k].tick_params(labelsize=6)
    ax_row[0].set_title(title, fontsize=8, loc="left")
    ax_row[3].barh([SHORT[n] for n in names], proba, color=["C3" if p == proba.max() else "C7" for p in proba])
    ax_row[3].set_xlim(0, 1); ax_row[3].tick_params(labelsize=6); ax_row[3].set_xlabel("P(clase)", fontsize=7)


def grid(examples, X, P, names, path, suptitle):
    fig, axs = plt.subplots(len(examples), 4, figsize=(15, 2.4 * len(examples)),
                            gridspec_kw={"width_ratios": [3, 3, 3, 1.4]})
    for r, e in enumerate(examples):
        plot_example(axs[r], X[e["idx"]], P[e["idx"]], names,
                     f"#{e['idx']} suj {e['subject']} | real: {e['true']} | pred: {e['pred']} (p={e['p_pred']:.2f})")
    axs[0, 0].legend(fontsize=6, ncol=3)
    for c in range(3):
        axs[-1, c].set_xlabel("t [s]", fontsize=7)
    fig.suptitle(suptitle, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    set_global_seed(SEED); rng = np.random.default_rng(SEED)
    names = [load_activity_labels()[i] for i in range(1, 7)]
    model, scaler, card = load_final_model()

    if a.smoke:   # el HOLD-OUT sustituye al TEST solo para probar el pipeline
        sp = read_json(SPLITS_FILE); X, y, s = load_or_cache("train")
        m = mask_by_subjects(s, sp["holdout_subjects"]); X, y, s = X[m], y[m], s[m]
        OUT = ROOT / "fase_5_evaluacion" / "smoke_test"; mode = "offline"; split = "HOLDOUT(smoke)"
    else:
        X, y, s = load_or_cache("test"); OUT = ROOT / "fase_5_evaluacion" / "test"; mode = "online"; split = "TEST"
        assert not set(np.unique(s)) & set(card["train_subjects"]), "FUGA: sujetos de entrenamiento en TEST"
    OUT.mkdir(parents=True, exist_ok=True)

    P = model.predict(scaler.transform(X), verbose=0); yp = P.argmax(1); yt = y - 1
    met = compute_metrics(yt, yp, names); dump_json(met, OUT / "metricas.json")
    pd.DataFrame(met["per_class"]).T.to_csv(OUT / "metricas_por_clase.csv")
    plot_cm(met["confusion_matrix"], names, OUT / "matriz_confusion.png",
            f"LSTM final (semilla {card['seed']}) - {split} ({len(y)} ventanas)")
    pred_df = pd.DataFrame({"idx": np.arange(len(y)), "subject": s, "true": [names[i] for i in yt],
                            "pred": [names[i] for i in yp], "p_pred": P.max(1), "correct": yt == yp})
    for i, n in enumerate(names):
        pred_df[f"p_{n}"] = P[:, i]
    pred_df.to_csv(OUT / "predicciones.csv", index=False)

    # --- Selección pre-especificada de ejemplos
    ok = pred_df[pred_df.correct]
    classes = rng.choice(sorted(ok.true.unique()), size=min(5, ok.true.nunique()), replace=False)
    correct = [ok[ok.true == c].sample(1, random_state=SEED).iloc[0].to_dict() for c in classes]
    bad = pred_df[~pred_df.correct]
    pairs = bad.groupby(["true", "pred"]).size().sort_values(ascending=False)
    chosen = list(pairs.index[:5]) + [pairs.index[0]] * max(0, 5 - len(pairs))
    wrong, used = [], set()
    for (t_, p_) in chosen:
        cand = bad[(bad.true == t_) & (bad.pred == p_) & (~bad.idx.isin(used))]
        e = cand.sample(1, random_state=SEED + len(wrong)).iloc[0].to_dict(); used.add(e["idx"]); wrong.append(e)
    grid(correct, X, P, names, OUT / "ejemplos_correctos.png", f"5 ejemplos bien clasificados - {split}")
    grid(wrong, X, P, names, OUT / "ejemplos_incorrectos.png", f"5 ejemplos mal clasificados - {split}")
    pd.DataFrame(correct + wrong).assign(tipo=["correcto"] * len(correct) + ["incorrecto"] * len(wrong)) \
        .to_csv(OUT / "ejemplos_seleccionados.csv", index=False)

    # --- Análisis de confusiones: pares más frecuentes y efecto sujeto en posturas estáticas
    pairs.rename("n").reset_index().to_csv(OUT / "pares_confusion.csv", index=False)
    G = (X[..., 6:9] - X[..., 0:3]).mean(1)
    rows = []
    for sub in np.unique(s):
        k = (s == sub) & (yt == 4); j = (s == sub) & (yt == 3)
        rows.append({"subject": int(sub), "acc": float((yt[s == sub] == yp[s == sub]).mean()),
                     "n_standing": int(k.sum()), "recall_standing": float((yp[k] == 4).mean()),
                     "standing->sitting": float((yp[k] == 3).mean()),
                     "recall_sitting": float((yp[j] == 3).mean()),
                     "g_y_standing": float(G[k, 1].mean()), "g_y_sitting": float(G[j, 1].mean())})
    subj = pd.DataFrame(rows); subj.to_csv(OUT / "analisis_por_sujeto.csv", index=False)
    wrong_conf = pred_df[~pred_df.correct].p_pred
    summary = {"split": split, "n": int(len(y)), "accuracy": met["accuracy"], "macro_f1": met["macro_f1"],
               "macro_precision": met["macro_precision"], "macro_recall": met["macro_recall"],
               "conf_media_errores": float(wrong_conf.mean()) if len(wrong_conf) else None,
               "conf_media_aciertos": float(pred_df[pred_df.correct].p_pred.mean()),
               "top_pares_confusion": pairs.head(5).to_dict()}
    summary["top_pares_confusion"] = {f"{k[0]}->{k[1]}": int(v) for k, v in pairs.head(5).items()}
    dump_json(summary, OUT / "resumen.json")

    # --- Muestras para la interfaz (tarea f): 2 ventanas por clase, unidades físicas, columnas = CHANNELS
    if not a.smoke:
        mdir = ROOT / "fase_6_despliegue" / "muestras"; mdir.mkdir(parents=True, exist_ok=True)
        for c in names:
            for i in pred_df[pred_df.true == c].sample(2, random_state=SEED).idx:
                pd.DataFrame(X[i], columns=CHANNELS).to_csv(mdir / f"test_{i:04d}_suj{s[i]}_{c}.csv", index=False)

    # --- W&B
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, group="final-lstm", job_type="test",
                     name=f"test-final-lstm-s{card['seed']}", mode=mode,
                     config={"model_run_id": card["run_id"], "split": split, "code_version": code_version(),
                             "selection_rule": "ver docstring eval_test.py", "seed": SEED})
    run.summary.update({f"test_{k}": v for k, v in summary.items() if isinstance(v, (int, float))})
    run.log({"test_confusion_matrix": wandb.plot.confusion_matrix(y_true=yt.tolist(), preds=yp.tolist(),
                                                                 class_names=names),
             "ejemplos_correctos": wandb.Image(str(OUT / "ejemplos_correctos.png")),
             "ejemplos_incorrectos": wandb.Image(str(OUT / "ejemplos_incorrectos.png")),
             "analisis_por_sujeto": wandb.Table(dataframe=subj)})
    if not a.smoke:
        append_log({"ID": next_id(), "fecha": now(), "fase": "5-test", "arquitectura": card["arch"],
                    "hiperparametros": json.dumps(card["hp"], ensure_ascii=False), "semilla": card["seed"],
                    "particion_fold": "DEV->TEST", "exactitud": round(met["accuracy"], 4),
                    "macro_f1": round(met["macro_f1"], 4), "run_id_wandb": run.id, "url_wandb": run.url,
                    "version_codigo": code_version(), "observaciones": "evaluación única en TEST oficial"})
    run.finish()
    print(json.dumps(summary, indent=1, ensure_ascii=False)); print(subj.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

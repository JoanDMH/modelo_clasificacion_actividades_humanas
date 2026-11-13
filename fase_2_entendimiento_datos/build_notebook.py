"""Genera 05_eda_notebook.ipynb (se ejecuta luego con nbconvert)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# Fase 2 — Entendimiento de los datos (CRISP-DM)
**Proyecto 13:** Clasificación de actividades humanas desde señales inerciales (UCI HAR).

Objetivo de la fase: describir y **verificar** los datos crudos (`Inertial Signals`) antes de prepararlos.
No se entrena ningún modelo ni se ajustan estadísticos de normalización aquí (eso es Fase 3).

Etiquetas usadas: **[DATO]** (dataset / README UCI), **[SUPUESTO]**, **[HIPÓTESIS]**.""")

code("""import sys, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
from scipy import signal

ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))
from data_io import load_or_cache, load_activity_labels, CHANNELS, FS_HZ, WINDOW_LEN

FIG = Path("figuras"); FIG.mkdir(exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})
print("numpy", np.__version__, "| pandas", pd.__version__, "| matplotlib", matplotlib.__version__)""")

md("## 1. Carga y verificación estructural")
code("""Xtr, ytr, str_ = load_or_cache("train")
Xte, yte, ste = load_or_cache("test")
LABELS = load_activity_labels()
NAMES = [LABELS[i] for i in sorted(LABELS)]
print("Train:", Xtr.shape, ytr.shape, str_.shape)
print("Test :", Xte.shape, yte.shape, ste.shape)
print("Etiquetas:", LABELS)""")

code("""checks = {
    "train_shape": list(Xtr.shape), "test_shape": list(Xte.shape),
    "nan_train": int(np.isnan(Xtr).sum()), "nan_test": int(np.isnan(Xte).sum()),
    "inf_train": int(np.isinf(Xtr).sum()), "inf_test": int(np.isinf(Xte).sum()),
    "labels_train": sorted(map(int, np.unique(ytr))), "labels_test": sorted(map(int, np.unique(yte))),
    "subjects_train": sorted(map(int, np.unique(str_))), "subjects_test": sorted(map(int, np.unique(ste))),
}
inter = set(checks["subjects_train"]) & set(checks["subjects_test"])
checks["subject_overlap_train_test"] = sorted(inter)
assert not inter, "¡Fuga! sujetos compartidos entre train y test"
assert Xtr.shape[1:] == (WINDOW_LEN, 9) and Xte.shape[1:] == (WINDOW_LEN, 9)
# Duplicados exactos de ventanas
def n_dup(X):
    v = X.reshape(len(X), -1)
    return len(v) - len(np.unique(v, axis=0))
checks["dup_windows_train"] = n_dup(Xtr); checks["dup_windows_test"] = n_dup(Xte)
n_tot = len(set(checks["subjects_train"]) | set(checks["subjects_test"]))
checks["n_subjects_total"] = n_tot
checks["frac_windows_train"] = round(len(Xtr) / (len(Xtr) + len(Xte)), 4)
checks["frac_subjects_train"] = round(len(checks["subjects_train"]) / n_tot, 4)
for k, v in checks.items(): print(f"{k:28s}: {v}")""")

md("## 2. Distribución de clases y sujetos")
code("""def dist(y):
    s = pd.Series(y).map(LABELS).value_counts().reindex(NAMES)
    return s
dc = pd.DataFrame({"train_n": dist(ytr), "test_n": dist(yte)})
dc["train_%"] = (100*dc.train_n/dc.train_n.sum()).round(2)
dc["test_%"] = (100*dc.test_n/dc.test_n.sum()).round(2)
dc["ratio_max_min_train"] = ""
dc.iloc[0, -1] = round(dc.train_n.max()/dc.train_n.min(), 3)
display(dc)
dc.to_csv(FIG/"tabla_distribucion_clases.csv")

fig, ax = plt.subplots(figsize=(8,3.5))
x = np.arange(6); w=.4
ax.bar(x-w/2, dc["train_%"], w, label="train"); ax.bar(x+w/2, dc["test_%"], w, label="test")
ax.set_xticks(x, NAMES, rotation=20, ha="right"); ax.set_ylabel("% de ventanas"); ax.legend()
ax.set_title("Distribución de clases por partición oficial")
fig.tight_layout(); fig.savefig(FIG/"fig01_distribucion_clases.png")""")

code("""ct = pd.crosstab(pd.Series(np.r_[str_, ste], name="sujeto"),
                 pd.Series(np.r_[ytr, yte], name="actividad").map(LABELS)).reindex(columns=NAMES)
ct["total"] = ct.sum(1)
ct["split"] = ["train" if s in set(str_) else "test" for s in ct.index]
display(ct)
ct.to_csv(FIG/"tabla_sujeto_actividad.csv")
print("Ventanas por sujeto: min", ct.total.min(), "max", ct.total.max(), "media", round(ct.total.mean(),1))
print("¿Todo sujeto tiene las 6 actividades?", bool((ct[NAMES] > 0).all().all()))

fig, ax = plt.subplots(figsize=(10,3.5))
ct[NAMES].plot(kind="bar", stacked=True, ax=ax, width=.85)
ax.set_ylabel("n ventanas"); ax.set_title("Ventanas por sujeto y actividad (train y test)")
ax.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(.5,-.18))
fig.tight_layout(); fig.savefig(FIG/"fig02_ventanas_por_sujeto.png")""")

md("""## 3. Verificación del preprocesamiento declarado por UCI
### 3.1 Solapamiento del 50 %
Si las ventanas son consecutivas con 50 % de solapamiento, la 2.ª mitad de la ventana *i* debe coincidir con la 1.ª mitad de la ventana *i+1* (mismo sujeto y actividad).""")
code("""def overlap_check(X, y, s, tol=1e-6):
    same = (s[1:] == s[:-1]) & (y[1:] == y[:-1])
    d = np.abs(X[:-1, 64:, :] - X[1:, :64, :]).max(axis=(1, 2))
    return {"pares_consecutivos_mismo_sujeto_act": int(same.sum()),
            "con_solape_exacto": int((d[same] < tol).sum()),
            "pct": round(100 * (d[same] < tol).mean(), 2),
            "max_dif_en_pares_que_fallan": float(d[same][d[same] >= tol].max()) if (d[same] >= tol).any() else 0.0}
ov = {"train": overlap_check(Xtr, ytr, str_), "test": overlap_check(Xte, yte, ste)}
pd.DataFrame(ov).T""")

md("""### 3.2 Separación gravedad / cuerpo
UCI declara `body_acc = total_acc − gravedad`, con gravedad obtenida por Butterworth paso bajo a 0.3 Hz. Entonces `g = total − body` debería (a) tener módulo ≈ 1 g y (b) concentrar su energía bajo 0.3 Hz.""")
code("""def grav(X): return X[..., 6:9] - X[..., 0:3]
G = grav(Xtr)
gmag = np.linalg.norm(G, axis=-1)
f, P = signal.welch(G - G.mean(1, keepdims=True), fs=FS_HZ, nperseg=128, axis=1)
frac_low_g = (P[:, f <= 0.5, :].sum(1) / P.sum(1)).mean()
fb, Pb = signal.welch(Xtr[..., 0:3], fs=FS_HZ, nperseg=128, axis=1)
frac_low_b = (Pb[:, fb <= 0.5, :].sum(1) / Pb.sum(1)).mean()
grav_res = {"|g| media": float(gmag.mean()), "|g| desv": float(gmag.std()),
            "|g| p1": float(np.percentile(gmag, 1)), "|g| p99": float(np.percentile(gmag, 99)),
            "frac_energia_<=0.5Hz_gravedad(sin DC)": float(frac_low_g),
            "frac_energia_<=0.5Hz_body_acc": float(frac_low_b),
            "media |body_acc| por ventana (promedio)": float(np.abs(Xtr[..., :3]).mean())}
for k,v in grav_res.items(): print(f"{k:42s}: {v:.4f}")
print("Nota: con ventanas de 128 muestras, la resolución de Welch es 50/128 ≈ 0.39 Hz.")""")

md("## 4. Estadísticos por canal (solo descriptivos; NO se usan para normalizar)")
code("""def chan_stats(X):
    v = X.reshape(-1, 9)
    return pd.DataFrame({"media": v.mean(0), "desv": v.std(0), "min": v.min(0),
                         "p1": np.percentile(v,1,0), "p99": np.percentile(v,99,0), "max": v.max(0)}, index=CHANNELS)
st_tr, st_te = chan_stats(Xtr), chan_stats(Xte)
display(st_tr.round(4)); display(st_te.round(4))
st_tr.to_csv(FIG/"tabla_stats_canal_train.csv"); st_te.to_csv(FIG/"tabla_stats_canal_test.csv")
print("Razón de escalas desv (máx/mín) en train:", round(st_tr.desv.max()/st_tr.desv.min(), 2))""")

code("""fig, axs = plt.subplots(3,3, figsize=(11,7), sharey=False)
for k, ax in enumerate(axs.ravel()):
    ax.hist(Xtr[..., k].ravel(), bins=120, density=True, alpha=.6, label="train")
    ax.hist(Xte[..., k].ravel(), bins=120, density=True, alpha=.6, label="test")
    ax.set_title(CHANNELS[k], fontsize=9); ax.set_yscale("log")
axs[0,0].legend(); fig.suptitle("Histogramas por canal (escala log)"); fig.tight_layout()
fig.savefig(FIG/"fig03_histogramas_canales.png")""")

md("## 5. Señales de ejemplo por actividad")
code("""fig, axs = plt.subplots(6, 3, figsize=(12, 13), sharex=True)
t = np.arange(WINDOW_LEN) / FS_HZ
groups = [("body_acc [g]", 0), ("body_gyro [rad/s]", 3), ("total_acc [g]", 6)]
ex_idx = {}
for r, a in enumerate(sorted(LABELS)):
    i = int(rng.choice(np.where(ytr == a)[0])); ex_idx[LABELS[a]] = i
    for c, (nm, o) in enumerate(groups):
        for j, ax_lbl in enumerate("xyz"):
            axs[r, c].plot(t, Xtr[i, :, o+j], lw=1, label=ax_lbl)
        if r == 0: axs[r, c].set_title(nm)
    axs[r, 0].set_ylabel(f"{LABELS[a]}\\n(idx {i}, suj {str_[i]})", fontsize=8)
axs[0,0].legend(fontsize=7); [axs[-1,c].set_xlabel("t [s]") for c in range(3)]
fig.tight_layout(); fig.savefig(FIG/"fig04_senales_ejemplo.png"); print("Índices (train):", ex_idx)""")

md("## 6. Firma espectral por actividad (módulo de body_acc)")
code("""bmag = np.linalg.norm(Xtr[..., 0:3], axis=-1)
f, P = signal.welch(bmag - bmag.mean(1, keepdims=True), fs=FS_HZ, nperseg=128, axis=1)
fig, ax = plt.subplots(figsize=(8,4))
peak = {}
for a in sorted(LABELS):
    m = P[ytr == a].mean(0); ax.semilogy(f, m, label=LABELS[a]); peak[LABELS[a]] = float(f[m.argmax()])
ax.set_xlabel("Frecuencia [Hz]"); ax.set_ylabel("PSD media"); ax.legend(fontsize=7)
ax.set_title("PSD media de |body_acc| por actividad (train)")
fig.tight_layout(); fig.savefig(FIG/"fig05_psd_actividad.png"); print("Pico dominante [Hz]:", peak)""")

md("""## 7. Separabilidad preliminar: energía y orientación
Indicadores simples por ventana: desviación estándar del módulo de `body_acc` (intensidad del movimiento) y media de cada eje de la gravedad estimada (orientación del teléfono).""")
code("""energy = bmag.std(1)
gmean = G.mean(1)
df7 = pd.DataFrame({"act": pd.Series(ytr).map(LABELS), "std_|body_acc|": energy,
                    "g_x": gmean[:,0], "g_y": gmean[:,1], "g_z": gmean[:,2]})
summ = df7.groupby("act").agg(["mean","std"]).reindex(NAMES).round(4); display(summ)
summ.to_csv(FIG/"tabla_energia_orientacion.csv")

fig, axs = plt.subplots(1,2, figsize=(12,4))
df7.boxplot(column="std_|body_acc|", by="act", ax=axs[0], rot=20); axs[0].set_yscale("log")
axs[0].set_title("Intensidad de movimiento"); axs[0].set_xlabel("")
for a in [4,5,6]:
    m = ytr == a; axs[1].scatter(gmean[m,1], gmean[m,2], s=4, alpha=.4, label=LABELS[a])
axs[1].set_xlabel("g_y media [g]"); axs[1].set_ylabel("g_z media [g]"); axs[1].legend()
axs[1].set_title("Orientación (clases estáticas)"); fig.suptitle(""); fig.tight_layout()
fig.savefig(FIG/"fig06_energia_orientacion.png")""")

code("""# Solapamiento SITTING vs STANDING en orientación: exactitud de un umbral óptimo 1-D (solo descriptivo, en train)
def best_threshold_acc(a, b):
    v = np.r_[a, b]; lab = np.r_[np.zeros(len(a)), np.ones(len(b))]
    best = 0
    for th in np.quantile(v, np.linspace(0.01, .99, 199)):
        acc = max(((v > th) == lab).mean(), ((v <= th) == lab).mean()); best = max(best, acc)
    return round(float(best), 4)
sep = {}
for k, nm in enumerate(["g_x","g_y","g_z"]):
    sep[nm] = best_threshold_acc(gmean[ytr==4, k], gmean[ytr==5, k])
sep["std_|body_acc|"] = best_threshold_acc(energy[ytr==4], energy[ytr==5])
print("Exactitud de umbral único SITTING vs STANDING (train, optimista):", sep)""")

md("## 8. Variabilidad entre sujetos (motivo de la partición por sujeto)")
code("""ds = pd.DataFrame({"suj": str_, "act": pd.Series(ytr).map(LABELS), "gx": gmean[:,0], "e": energy})
w = ds[ds.act=="WALKING"].groupby("suj").e.mean()
print("std|body_acc| en WALKING — media entre sujetos:", round(w.mean(),4), "| CV entre sujetos:", round(w.std()/w.mean(),3))
# ANOVA simple: fracción de varianza explicada por sujeto dentro de una actividad
def eta2(df, col):
    g = df.groupby("suj")[col]; tot = ((df[col]-df[col].mean())**2).sum()
    return float(((g.transform("mean")-df[col].mean())**2).sum()/tot)
eta = {a: {"eta2_energia": round(eta2(ds[ds.act==a],"e"),3), "eta2_g_x": round(eta2(ds[ds.act==a],"gx"),3)} for a in NAMES}
display(pd.DataFrame(eta).T)""")

md("## 9. Resumen exportado")
code("""res = {"checks": checks, "overlap": ov, "gravity": grav_res, "psd_peak_hz": peak,
       "sit_vs_stand_threshold_acc": sep, "eta2_subject": eta,
       "class_dist": dc.drop(columns="ratio_max_min_train").to_dict(),
       "windows_per_subject": {"min": int(ct.total.min()), "max": int(ct.total.max()), "mean": float(ct.total.mean())}}
Path("resumen_eda.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
print("Guardado resumen_eda.json")""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).with_name("05_eda_notebook.ipynb"))

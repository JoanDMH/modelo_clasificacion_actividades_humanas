"""Fase 6 – Tarea (f): interfaz Streamlit para clasificar una ventana inercial (128 x 9).

Ejecutar desde la raíz del proyecto, con .venv activo:
    streamlit run fase_6_despliegue/app/app.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
APP = Path(__file__).resolve().parent
ROOT = APP.parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(APP))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from charts import PALETTE, proba_html, signals_html  # noqa: E402
from config import MODELS_DIR  # noqa: E402
from data_io import CHANNELS, WINDOW_LEN, load_or_cache  # noqa: E402
from models import build_model  # noqa: E402
from prep import ChannelStandardizer  # noqa: E402
from train_eval import read_json  # noqa: E402

ES = {"WALKING": "Caminar", "WALKING_UPSTAIRS": "Subir escaleras", "WALKING_DOWNSTAIRS": "Bajar escaleras",
      "SITTING": "Sentado", "STANDING": "De pie", "LAYING": "Acostado"}
EVAL = ROOT / "fase_5_evaluacion"

st.set_page_config(page_title="HAR · Clasificador de actividad", page_icon="◐", layout="wide")
P = PALETTE
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500&family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, .stMarkdown, .stButton, .stRadio {{ font-family:'Instrument Sans',system-ui,sans-serif; }}
.stApp {{ background:{P['bg']}; }}
header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:2.2rem; max-width:1180px; }}
.eyebrow {{ font:500 11px 'IBM Plex Mono',monospace; letter-spacing:.14em; text-transform:uppercase; color:{P['muted']}; margin-bottom:.35rem; }}
.title {{ font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:2.1rem; line-height:1.15; color:{P['ink']}; margin:.2rem 0 .3rem; }}
.sub {{ color:{P['muted']}; font-size:.95rem; max-width:62ch; }}
.card {{ background:#fff; border:1px solid {P['grid']}; border-radius:14px; padding:1.3rem 1.5rem; }}
.pred {{ font-family:'Fraunces',Georgia,serif; font-weight:600; font-size:2.6rem; line-height:1.05; color:{P['ink']}; margin:.35rem 0 .25rem; }}
.conf {{ font:500 .95rem 'IBM Plex Mono',monospace; color:{P['muted']}; }}
.chip {{ display:inline-block; margin-top:.9rem; padding:.28rem .75rem; border-radius:999px; font-size:.82rem; font-weight:500; }}
.ok {{ background:#E3F0EE; color:{P['accent']}; }} .bad {{ background:#F6E6DE; color:{P['error']}; }} .na {{ background:#EFECE6; color:{P['muted']}; }}
.kpi {{ border-left:2px solid {P['accent']}; padding:.15rem 0 .15rem .9rem; }}
.kpi .v {{ font:500 1.35rem 'IBM Plex Mono',monospace; color:{P['ink']}; }}
.kpi .k {{ font-size:.78rem; color:{P['muted']}; line-height:1.4; }}
.note {{ font-size:.82rem; color:{P['muted']}; line-height:1.55; border-top:1px solid {P['grid']}; padding-top:.9rem; margin-top:1.2rem; }}
section[data-testid="stSidebar"] {{ background:#F3EFE8; border-right:1px solid {P['grid']}; }}
.stButton>button, .stDownloadButton>button {{ border-radius:999px; border:1px solid {P['bar']}; background:#fff; color:{P['ink']}; }}
.stButton>button:hover, .stDownloadButton>button:hover {{ border-color:{P['accent']}; color:{P['accent']}; }}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------- datos y modelo (cacheados)
@st.cache_resource(show_spinner="Cargando modelo…")
def load_model():
    card = read_json(MODELS_DIR / "model_card.json")
    model = build_model(card["arch"], lr=card["hp"]["lr"], dropout=card["hp"]["dropout"])
    model.load_weights(MODELS_DIR / "final_lstm.keras")
    return model, ChannelStandardizer.load(MODELS_DIR / "scaler.json"), card


@st.cache_data(show_spinner=False)
def load_test():
    return load_or_cache("test")


@st.cache_data(show_spinner=False)
def load_metrics():
    """Métricas trazables: exactitud en validación por sujeto (3 semillas) y en la prueba oficial."""
    out = {}
    f = EVAL / "resultados_holdout.csv"
    if f.exists():
        d = pd.read_csv(f)
        out["val"] = (d.accuracy.mean(), d.accuracy.std(ddof=1), len(d))
    f = EVAL / "test" / "resumen.json"
    if f.exists():
        r = read_json(f); out["test"] = (r["accuracy"], r["n"])
    return out


def parse_csv(file) -> np.ndarray:
    df = pd.read_csv(file)
    if set(CHANNELS).issubset(df.columns):
        df = df[CHANNELS]
    else:
        file.seek(0); df = pd.read_csv(file, header=None)
    x = df.to_numpy(dtype=np.float32)
    if x.shape != (WINDOW_LEN, len(CHANNELS)):
        raise ValueError(f"se esperaban {WINDOW_LEN} filas × {len(CHANNELS)} columnas y se recibieron "
                         f"{x.shape[0]} × {x.shape[1]}")
    if not np.isfinite(x).all():
        raise ValueError("el archivo contiene valores vacíos o no numéricos")
    return x


def new_window(act_label=None):
    pool = np.where(yt == labels.index(act_label) + 1)[0] if act_label else np.arange(len(Xt))
    st.session_state.idx = int(np.random.default_rng().choice(pool))


try:
    model, scaler, card = load_model()
except FileNotFoundError:
    st.error("No se encontró el modelo entrenado. Ejecute primero `fase_5_evaluacion/train_final.py`.")
    st.stop()
labels = card["labels"]
Xt, yt, _ = load_test()
INV = {v: k for k, v in ES.items()}

# ----------------------------------------------------------------- entrada (barra lateral)
with st.sidebar:
    st.markdown('<div class="eyebrow">Señal de entrada</div>', unsafe_allow_html=True)
    modo = st.radio("Origen", ["Conjunto de prueba", "Subir archivo"], label_visibility="collapsed")
    x, true_idx, origen = None, None, ""
    if modo == "Conjunto de prueba":
        if "idx" not in st.session_state:
            new_window()
        st.caption("Elija una actividad para cargar una ventana al azar de ese tipo.")
        act = st.pills("Actividad", list(ES.values()), key="act", label_visibility="collapsed",
                       on_change=lambda: new_window(INV.get(st.session_state.act)))
        st.button("↻  Otra ventana", width="stretch", on_click=lambda: new_window(INV.get(st.session_state.act)))
        with st.expander("Elegir por número"):
            st.number_input(f"Ventana (0 – {len(Xt) - 1})", 0, len(Xt) - 1, key="idx", step=1)
        i = int(st.session_state.idx)
        x, true_idx, origen = Xt[i], int(yt[i]) - 1, f"ventana {i} del conjunto de prueba"
    else:
        up = st.file_uploader("CSV de 128 filas × 9 columnas", type="csv", label_visibility="collapsed")
        st.caption("128 muestras (2.56 s a 50 Hz) × 9 canales, en g y rad/s, en este orden: "
                   "body_acc x·y·z, body_gyro x·y·z, total_acc x·y·z. El encabezado es opcional.")
        if up is not None:
            try:
                x, origen = parse_csv(up), up.name
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo leer el archivo: {e}.")
    if x is not None:
        st.download_button("Descargar ventana (CSV)", pd.DataFrame(x, columns=CHANNELS).to_csv(index=False).encode(),
                           "ventana.csv", "text/csv", width="stretch")

    m = load_metrics()
    if m:
        st.markdown('<div class="eyebrow" style="margin-top:2.2rem">Desempeño del modelo · exactitud</div>',
                    unsafe_allow_html=True)
        if "val" in m:
            a, s, n = m["val"]
            st.markdown(f'<div class="kpi"><div class="v">{a:.3f} ± {s:.3f}</div><div class="k">Validación con '
                        f'sujetos no vistos · media ± desv. est. de {n} entrenamientos (semillas)</div></div>',
                        unsafe_allow_html=True)
        if "test" in m:
            a, n = m["test"]
            st.markdown(f'<div class="kpi" style="margin-top:1rem"><div class="v">{a:.3f}</div><div class="k">'
                        f'Prueba oficial · 9 sujetos, {n:,} ventanas · evaluación única del modelo desplegado'.replace("{:,}".format(n), "{:,}".format(n).replace(",", "\u202f"))
                        + '</div></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------- cabecera
st.markdown('<div class="eyebrow">Reconocimiento de actividad humana · UCI HAR</div>'
            '<div class="title">¿Qué estaba haciendo la persona?</div>'
            '<div class="sub">Una red LSTM analiza 2.56 s de señal del acelerómetro y el giroscopio de un teléfono '
            'llevado en la cintura y la asigna a una de seis actividades.</div>', unsafe_allow_html=True)
st.write("")

if x is None:
    st.info("Suba un archivo CSV desde el panel izquierdo o cambie a «Conjunto de prueba».")
    st.stop()

proba = model.predict(scaler.transform(x[None]), verbose=0)[0]
pred = int(proba.argmax())

# ----------------------------------------------------------------- resultado
left, right = st.columns([1, 1.3], gap="large")
with left:
    if true_idx is None:
        chip = '<span class="chip na">Sin etiqueta de referencia</span>'
    elif true_idx == pred:
        chip = '<span class="chip ok">✓ Coincide con la etiqueta real</span>'
    else:
        chip = f'<span class="chip bad">✗ Etiqueta real: {ES[labels[true_idx]]}</span>'
    st.markdown(f'<div class="card"><div class="eyebrow">Actividad predicha</div>'
                f'<div class="pred">{ES[labels[pred]]}</div>'
                f'<div class="conf">probabilidad {proba[pred]:.1%}</div>{chip}</div>', unsafe_allow_html=True)
with right:
    st.markdown('<div class="eyebrow">Probabilidad por clase' + (' · ● etiqueta real' if true_idx is not None else '')
                + '</div>', unsafe_allow_html=True)
    st.iframe(proba_html([ES[l] for l in labels], proba, pred, true_idx), height=212)  # HTML generado internamente

st.markdown(f'<div class="eyebrow" style="margin-top:1.4rem">Señal · {origen}</div>', unsafe_allow_html=True)
st.iframe(signals_html(x), height=416)  # solo valores numéricos de la señal

st.markdown(
    '<div class="note"><b>Cómo leer el resultado.</b> La probabilidad expresa la preferencia del modelo, no la '
    'certeza de acierto: en la evaluación hubo errores con probabilidad superior al 90 %. El modelo depende de la '
    'orientación del teléfono en la cintura, y su confusión más frecuente es entre <i>De pie</i> y <i>Sentado</i>.</div>',
    unsafe_allow_html=True)

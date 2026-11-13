# 01 · Preparación de los datos
**Fase CRISP-DM:** 3 – Preparación de los datos · **Fecha:** 2026-11-12
**Código:** `src/prep.py`, `src/config.py` · **Evidencia:** `02_preparacion_notebook.ipynb` · **Particiones congeladas:** `particiones.json`

## 1. Selección de datos
- Entradas: los 9 canales de `Inertial Signals` en el orden `src/data_io.CHANNELS`. Se excluyen las 561 características (decisión D1 de la Fase 2).
- Objetivo: la etiqueta `y` (1–6), que se re-indexará a 0–5 al construir los modelos.
- Grupo: `subject`. Solo se usa para particionar; nunca es una entrada del modelo.

## 2. Limpieza
- No se requiere: hay 0 NaN, 0 Inf y 0 duplicados [DATO: Fase 2, `02_descripcion_datos.md` §3].
- [SUPUESTO] No se eliminan valores extremos. Son movimientos reales (por ejemplo, los picos al bajar escaleras) y no errores del sensor.

## 3. Partición (sin fuga entre sujetos)
```
train oficial (21 sujetos, 7 352 ventanas)
 ├── HOLD-OUT de validación: 4 sujetos, 1 340 ventanas  -> tarea (c)
 └── DEV: 17 sujetos, 6 012 ventanas                    -> CV de 2 particiones (b) y entrenamiento del modelo final
test oficial (9 sujetos, 2 947 ventanas)                -> tareas (d) y (f), se usa una sola vez al final
```
Las particiones se generan con `StratifiedGroupKFold(shuffle=True, random_state=42)`, con `groups = subject`.

| Partición | Sujetos | Ventanas |
|---|---|---|
| HOLD-OUT (c) | 5, 7, 22, 25 | 1 340 |
| DEV | 1, 3, 6, 8, 11, 14, 15, 16, 17, 19, 21, 23, 26, 27, 28, 29, 30 | 6 012 |
| CV fold 0: train / val | {1, 3, 6, 11, 15, 16, 19, 30} / {8, 14, 17, 21, 23, 26, 27, 28, 29} | 2 766 / 3 246 |
| CV fold 1 | Roles invertidos respecto del fold 0 | 3 246 / 2 766 |
| TEST oficial | 2, 4, 9, 10, 12, 13, 18, 20, 24 | 2 947 |

- **Estratificación:** el porcentaje de cada clase en cualquier partición difiere como máximo 1.54 puntos del de DEV (`figuras/tabla_particiones.csv`).
- **Aserciones anti-fuga (todas pasan):**
  - Los sujetos de DEV, HOLD-OUT y TEST son disjuntos.
  - Cada ventana de entrenamiento pertenece exactamente a DEV o a HOLD-OUT.
  - Dentro de cada fold, los sujetos de train y val son disjuntos y juntos cubren todo DEV.
  - Ningún sujeto del HOLD-OUT aparece en la CV.
- [SUPUESTO] **La CV se hace sobre DEV y no sobre los 21 sujetos.** Así el HOLD-OUT no interviene en la elección de arquitectura, y la evaluación de la tarea (c) no queda sesgada por esa elección. El costo es entrenar con 4 sujetos menos.
- [SUPUESTO] **Los folds se cargan siempre desde `particiones.json`; no se recalculan.** El orden aleatorio de `StratifiedGroupKFold` puede cambiar entre versiones de scikit-learn, y este archivo se generó con scikit-learn 1.6.1, mientras que tu `.venv` tiene la 1.9.1. Usar el JSON garantiza que todas las fases usen las mismas particiones.

## 4. Construcción y formato: normalización por canal
- Se aplica un z-score por canal: `Z = (X − μ_c) / σ_c`. μ y σ se calculan sobre todas las ventanas y pasos de tiempo del **train del fold** (clase `ChannelStandardizer`).
- El mismo escalador se aplica a la validación del fold, al HOLD-OUT y al TEST.
- **Evidencia de que no hay fuga (fold 0, `figuras/tabla_normalizacion_fold0.csv`):**
  - Tras normalizar, el train del fold queda con media 0 y desviación 1 (|error| < 1e-3).
  - En validación, la media va de −0.137 a 0.015 y la desviación de 0.85 a 1.39, según el canal. Ese desvío es esperable y muestra que no se usó la validación para ajustar.
  - Las μ ajustadas solo con el train del fold difieren hasta 0.022 de las que se obtendrían con train + val.
- **Escaladores por usar en las fases siguientes:**
  - Uno por fold de CV, con el train de ese fold.
  - Uno para el modelo final, ajustado con DEV completo. Se guardará en `models/scaler.json` para la interfaz. La función `save`/`load` ya está verificada.
- Formato de salida: tensor `float32` de forma `(n, 128, 9)`, listo para `tf.keras`.

## 5. Integración
- No aplica: hay una sola fuente de datos y sus archivos ya están alineados fila a fila [DATO: `README.txt`].

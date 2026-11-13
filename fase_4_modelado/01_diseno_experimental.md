# 01 · Diseño experimental (tarea b)
**Fase CRISP-DM:** 4 – Modelado · **Estado:** listo para ejecutar · **Script:** `run_cv.py`

## Protocolo
- **CV repetida:** `StratifiedGroupKFold(2)` sobre DEV (17 sujetos) con 3 semillas (42, 43, 44). Son 6 evaluaciones por arquitectura y 12 runs en total.
  - Cada semilla cambia tanto el **reparto de sujetos** como la **inicialización** de los pesos.
  - Las particiones están congeladas en `cv_repeticiones.json`. La de la semilla 42 es idéntica a la de la Fase 3 (verificado con una aserción).
- **¿Por qué 3 semillas?** [SUPUESTO]
  - Con 2 particiones, una sola corrida da una desviación estándar calculada con n = 2, que no informa nada.
  - Con 3 repeticiones, n = 6, y se puede separar la varianza por partición de sujetos de la varianza entre semillas.
  - No se usan más semillas por el costo en CPU.
  - No mejora el modelo; mejora la **confiabilidad de la comparación** entre arquitecturas.
- **Normalización:** un escalador por (semilla, fold), ajustado solo con el train del fold.

## Arquitecturas (aprobadas por el usuario)
| ID | Arquitectura | Parámetros |
|---|---|---|
| A `lstm` | LSTM(64, seq) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(6, softmax) | 31 558 |
| B `conv_lstm` | Conv1D(64, 5) ×2 → MaxPool(2) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(6, softmax) | 56 902 |

## Hiperparámetros (aprobados)
- Adam con lr = 1e-3 y batch de 64.
- Máximo 50 épocas, con `EarlyStopping(val_loss, patience = 8, restore_best_weights)`.
- Pérdida: sparse categorical cross-entropy.
- Reproducibilidad: `TF_ENABLE_ONEDNN_OPTS = 0`, semillas fijas y `enable_op_determinism`.

**Nota metodológica:** la parada temprana usa la validación del fold, que es la misma con la que se reporta la exactitud de CV. Eso introduce un sesgo optimista leve e igual para ambas arquitecturas. Se declarará como limitación.

## Trazabilidad
- W&B:
  - `group` = arquitectura y `name` = `{arch}-s{seed}-f{fold}`.
  - Se registran las métricas por época, la matriz de confusión y los artefactos (modelo y `scaler.json`).
- Cada run agrega una fila a `registro_experimentos.csv`, con el ID, el run_id, la URL y el hash del código.
- Salidas: `resultados_cv.csv` (por run) y `tabla_cv.csv` (media ± desviación por arquitectura).

## Prueba de humo (2026-11-12, entorno Linux, W&B offline)
- Una época, semilla 42 y ambas arquitecturas: el flujo completo funciona de punta a punta.
- Costo observado: de ~6 a 12 s por época.
- **Las métricas de esta prueba no son resultados y no se registran.**
- Estimación para la corrida completa en tu equipo: entre ~40 y 100 minutos [SUPUESTO].

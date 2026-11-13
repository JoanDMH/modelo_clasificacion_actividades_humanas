# Reporte de la Fase 3 – Preparación de los datos
**Proyecto 13 · CRISP-DM fase 3 · Estado: CERRADA (2026-11-12)**
Detalle en `01_preparacion_datos.md` · Evidencia en `02_preparacion_notebook.ipynb` · Código en `src/prep.py` y `src/config.py`

## Objetivo
Particionar sin fuga entre sujetos, normalizar por canal y dejar los datos listos para `tf.keras`.

## Resultados
| Elemento | Resultado |
|---|---|
| HOLD-OUT de validación (c) | Sujetos 5, 7, 22, 25 · 1 340 ventanas |
| DEV (CV y modelo final) | 17 sujetos · 6 012 ventanas |
| TEST oficial (d, f) | 9 sujetos · 2 947 ventanas, sin tocar |
| Estratificación | Desviación máxima de 1.54 puntos porcentuales en la proporción de clases |
| Aserciones anti-fuga | Todas pasan |
| Normalización (fold 0) | Train con media 0 y desviación 1. Val con media entre −0.14 y 0.02 y desviación entre 0.85 y 1.39, lo esperable si no hubo fuga |

## Decisiones tomadas
| # | Decisión | Etiqueta / justificación |
|---|---|---|
| P1 | La CV se hace **solo sobre DEV**; el HOLD-OUT queda excluido de la selección de arquitectura | [SUPUESTO]: evita que la evaluación de (c) quede sesgada por esa elección |
| P2 | Las particiones se congelan en `particiones.json` y se leen desde ahí, nunca se recalculan | [SUPUESTO]: el orden aleatorio de scikit-learn puede variar entre versiones (1.6.1 frente a 1.9.1) |
| P3 | Un escalador por fold; el escalador final se ajusta con DEV y se guarda en `models/scaler.json` | Requisito del enunciado (interfaz) |
| P4 | No se eliminan valores extremos | [SUPUESTO]: son movimientos reales |
| P5 | Etiquetas re-indexadas a 0–5 solo dentro del modelo | Convención de Keras |

## Pendientes y riesgos
- Hay solo 4 sujetos en el HOLD-OUT, así que las métricas de (c) tendrán una varianza alta. Debe declararse como limitación.
- Al no usar el HOLD-OUT en la CV, se entrena con 4 sujetos menos.

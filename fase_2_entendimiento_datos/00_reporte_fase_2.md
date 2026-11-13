# Reporte de la Fase 2 – Entendimiento de los datos
**Proyecto 13 · CRISP-DM fase 2 · Estado: CERRADA (2026-11-12)**
Detalle en `01`–`04` · Evidencia en `05_eda_notebook.ipynb` y `resumen_eda.json`

## Objetivo
Verificar si los datos crudos (`Inertial Signals`) están realmente preprocesados como declara UCI, describirlos y extraer implicaciones para la preparación y el modelado.

## Hallazgos clave
| # | Hallazgo | Etiqueta |
|---|---|---|
| 1 | 7 352 ventanas de entrenamiento (21 sujetos) y 2 947 de prueba (9 sujetos), con forma (128, 9). Sujetos disjuntos. 0 NaN, 0 Inf, 0 duplicados | [DATO] |
| 2 | El preprocesamiento declarado se confirma: 50 % de solapamiento en el ~95 % de los pares consecutivos y gravedad estimada de 1.023 ± 0.013 g, concentrada en baja frecuencia | [DATO] |
| 3 | Las señales **no** están normalizadas: escalas de hasta 3.88× entre canales y `total_acc_x` con media 0.80 g | [DATO] |
| 4 | Desbalance leve: la razón entre la clase mayor y la menor es 1.43 | [DATO] |
| 5 | Efecto sujeto fuerte: η² entre 0.66 y 0.82 en las actividades dinámicas | [DATO] |
| 6 | Las clases dinámicas y estáticas se separan trivialmente por energía. LAYING se separa por orientación. SITTING y STANDING se solapan | [DATO] |
| 7 | La confusión principal del modelo será SITTING ↔ STANDING, y en segundo lugar WALKING ↔ WALKING_UPSTAIRS | [HIPÓTESIS] |
| 8 | El ~5 % de pares sin solapamiento corresponde a cortes entre grabaciones | [HIPÓTESIS], no verificable sin los datos crudos (RawData) |

## Decisiones tomadas
| # | Decisión | Justificación |
|---|---|---|
| D1 | Usar los 9 canales, incluido `total_acc` | `total_acc` contiene la gravedad (orientación), clave para las clases estáticas (H6) |
| D2 | z-score por canal ajustado solo con entrenamiento | Hallazgo 3 y requisito del enunciado |
| D3 | Partición por sujeto con `StratifiedGroupKFold` | Solapamiento (2) y efecto sujeto (5) |
| D4 | Validación *hold-out* por sujeto para (c); prueba oficial para (d) y (f) | [SUPUESTO] aprobado por el usuario |
| D5 | Reportar macro-F1 junto con la exactitud | Desbalance leve (4) |
| D6 | Sin aumento de datos ni remuestreo en la línea base | [SUPUESTO] |

## Pendientes y riesgos
- La licencia CC BY 4.0 del dataset está marcada como [REF-POR-VERIFICAR].
- La estimación del umbral SITTING/STANDING es optimista (se calculó sobre el mismo entrenamiento). Es solo descriptiva y no debe citarse como resultado.

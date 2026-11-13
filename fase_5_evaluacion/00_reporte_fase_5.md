# Reporte de la Fase 5 – Evaluación
**Proyecto 13 · CRISP-DM fase 5 · Estado: CERRADA (2026-11-13). Tareas (c) y (d) completadas**
Script `train_final.py` · Registro EXP-025 a EXP-027 · Modelo desplegable en `models/` (semilla 42, run `qbac5vxb`)

## 1. Protocolo de la tarea (c)
- LSTM (64→32) entrenada en DEV (17 sujetos, 6 012 ventanas) durante 7 épocas fijas, sin parada temprana.
- Normalización z-score ajustada solo con DEV (`models/scaler.json`).
- Evaluación en el HOLD-OUT de validación: sujetos 5, 7, 22 y 25, con 1 340 ventanas, que **no se usaron** ni en el entrenamiento ni en la selección de arquitectura.
- 3 semillas (42, 43, 44); se reporta media ± desv. est. (ddof = 1). El TEST oficial no se tocó.

## 2. Métricas globales en el HOLD-OUT [RESULTADO | qbac5vxb, 0pgykm9r, 0m661kzn]
| Métrica | Media ± desv. est. | Semilla 42 (desplegada) |
|---|---|---|
| Exactitud | **0.910 ± 0.016** | 0.892 |
| Precisión macro | 0.932 ± 0.008 | 0.924 |
| Exhaustividad macro | 0.916 ± 0.016 | 0.898 |
| F1 macro | **0.911 ± 0.015** | 0.894 |

Es coherente con la CV del protocolo principal (0.892 ± 0.037). El HOLD-OUT no muestra una degradación evidente.

## 3. Métricas por actividad (media ± desv. est., 3 semillas)
| Actividad | Precisión | Exhaustividad | F1 |
|---|---|---|---|
| WALKING | 0.975 ± 0.013 | 0.956 ± 0.026 | 0.965 ± 0.016 |
| WALKING_UPSTAIRS | 0.973 ± 0.014 | 0.979 ± 0.012 | 0.976 ± 0.013 |
| WALKING_DOWNSTAIRS | 0.984 ± 0.014 | 0.966 ± 0.035 | 0.975 ± 0.020 |
| SITTING | 0.683 ± 0.042 | 0.988 ± 0.010 | 0.807 ± 0.028 |
| STANDING | 0.987 ± 0.011 | **0.608 ± 0.042** | 0.752 ± 0.032 |
| LAYING | 0.988 ± 0.017 | 0.997 ± 0.005 | 0.993 ± 0.011 |

**Matriz de confusión (suma de las 3 semillas; filas = real, columnas = predicha)**
| | WALK | UP | DOWN | SIT | STAND | LAY |
|---|---|---|---|---|---|---|
| WALKING | 668 | 12 | 6 | 13 | 0 | 0 |
| UPSTAIRS | 12 | 602 | 1 | 0 | 0 | 0 |
| DOWNSTAIRS | 1 | 5 | 545 | 4 | 0 | 9 |
| SITTING | 1 | 0 | 1 | 649 | 6 | 0 |
| STANDING | 3 | 0 | 0 | **286** | 449 | 0 |
| LAYING | 0 | 0 | 1 | 1 | 0 | 745 |

Las figuras por semilla están en `seed{42,43,44}/matriz_confusion_holdout.png`.

## 4. Análisis de errores
1. **La confusión dominante es STANDING → SITTING y es asimétrica** [RESULTADO]. El 39 % de las ventanas de STANDING se predice como SITTING (286 de 738); en sentido inverso solo hay 6 de 657. Esto confirma la hipótesis H6 de la Fase 2. Las clases dinámicas y LAYING superan 0.96 de F1.
2. **El error depende del sujeto** [RESULTADO | modelo semilla 42; exactitud reproducida 0.8918 en otro entorno]

| Sujeto | Exhaustividad STANDING | g_y media STANDING | g_y media SITTING |
|---|---|---|---|
| 5 | 0.23 | −0.021 | +0.186 |
| 7 | 0.47 | −0.099 | +0.024 |
| 22 | **1.00** | −0.208 | +0.226 |
| 25 | 0.50 | −0.141 | +0.226 |
| *Referencia DEV* | — | −0.173 | +0.134 |

- [HIPÓTESIS] El modelo separa ambas posturas sobre todo por la inclinación del teléfono (g_y). Los sujetos cuya postura de pie produce una g_y cercana a la de sentado (sujeto 5) se clasifican como sentados. El sujeto 25 no encaja del todo (g_y típica y aun así 50 % de error), así que hay otros factores.
- Verificarlo requiere más sujetos; con 4 en el HOLD-OUT no es concluyente.
3. **Los errores se cometen con alta confianza** [RESULTADO]. La probabilidad media asignada a la clase errónea es 0.82 (sujeto 5) y 0.98 (sujeto 25). La salida softmax **no está calibrada**: la interfaz debe advertir que una probabilidad alta no garantiza acierto.
4. **Varianza entre semillas**: la exactitud va de 0.892 a 0.922. La semilla desplegada (42) es la peor de las tres. Se mantiene por estar pre-especificada [SUPUESTO]; elegir la mejor a posteriori sería selección sobre el HOLD-OUT.

## 5. Limitaciones
- Solo hay 4 sujetos en el HOLD-OUT: las métricas por clase dependen fuertemente de quiénes son.
- Las 7 épocas provienen de la CV sobre DEV, que es independiente del HOLD-OUT, así que no hay fuga.
- Compatibilidad: el archivo `final_lstm.keras` (Keras 3.15) no se deserializa completo en Keras 3.11, aunque sus pesos sí cargan. La interfaz debe usar las mismas versiones de `requirements.txt` o reconstruir la arquitectura y cargar los pesos.


## 6. Evaluación única en el TEST oficial – tarea (d) [RESULTADO | EXP-028, W&B f8j4v23e]
El modelo desplegado (LSTM, semilla 42) se evaluó una sola vez con `eval_test.py` sobre 9 sujetos y 2 947 ventanas. Resultados en `test/`.

| Métrica | TEST | HOLD-OUT (misma semilla) |
|---|---|---|
| Exactitud | **0.914** | 0.892 |
| F1 macro | **0.914** | 0.894 |
| Precisión macro | 0.914 | 0.924 |
| Exhaustividad macro | 0.916 | 0.898 |

**Por actividad (TEST)**
| Actividad | P | R | F1 |
|---|---|---|---|
| WALKING | 0.958 | 0.970 | 0.964 |
| UPSTAIRS | 0.869 | 0.915 | 0.891 |
| DOWNSTAIRS | 0.936 | 0.971 | 0.953 |
| SITTING | 0.819 | 0.864 | 0.840 |
| STANDING | 0.915 | 0.825 | 0.868 |
| LAYING | 0.990 | 0.950 | 0.970 |

**Matriz de confusión (filas = real)**
| | WALK | UP | DOWN | SIT | STAND | LAY |
|---|---|---|---|---|---|---|
| WALKING | 481 | 7 | 8 | 0 | 0 | 0 |
| UPSTAIRS | 21 | 431 | 19 | 0 | 0 | 0 |
| DOWNSTAIRS | 0 | 11 | 408 | 1 | 0 | 0 |
| SITTING | 0 | 20 | 1 | 424 | 41 | 5 |
| STANDING | 0 | 0 | 0 | 93 | 439 | 0 |
| LAYING | 0 | 27 | 0 | 0 | 0 | 510 |

**Ejemplos (regla pre-especificada, `test/ejemplos_seleccionados.csv`)**
- Correctos: #2037 STANDING (p = 0.93), #574 DOWNSTAIRS (0.89), #398 WALKING (0.99), #2829 LAYING (1.00) y #2939 UPSTAIRS (0.95).
- Incorrectos:

| Ventana | Sujeto | Real → Predicha | p | Observación en la señal |
|---|---|---|---|---|
| #631 | 9 | STANDING → SITTING | 0.74 | Señal estática; g ≈ (1.0, −0.02, 0.15), una inclinación típica de sentado en DEV |
| #45 | 2 | SITTING → STANDING | 0.87 | El giroscopio muestra deriva o asentamiento (−0.25 → 0 rad/s): probable fin de una transición |
| #1113 | 10 | LAYING → UPSTAIRS | 0.57 | Gravedad sobre −y (total_acc_y ≈ −0.93 g), una orientación ausente en DEV |
| #1499 | 12 | UPSTAIRS → WALKING | 0.99 | Marcha periódica de ~1.6 Hz, indistinguible de caminar en 2.56 s |
| #790 | 9 | SITTING → UPSTAIRS | 0.94 | Movimiento amplio (|body_acc| de hasta 0.3 g) y cambio de orientación: probable transición etiquetada como SITTING |

## 7. Análisis de las confusiones en TEST
1. **SITTING ↔ STANDING sigue siendo la confusión principal** (134 de 254 errores, 53 %) [RESULTADO]. A diferencia del HOLD-OUT, aquí es bidireccional: STANDING → SITTING 93 y SITTING → STANDING 41.
   - Se concentra en pocos sujetos. STANDING → SITTING: sujeto 9 (44, exhaustividad 0.02), sujeto 4 (28) y sujeto 10 (21). SITTING → STANDING: sujeto 2 (24).
   - 6 de 9 sujetos tienen exhaustividad 1.00 en STANDING.
   - Coincide con lo visto en el HOLD-OUT. En los sujetos con error, la g_y media de STANDING (sujeto 9: +0.008; sujeto 4: −0.058) está lejos del valor típico de DEV (−0.173) y cerca del de SITTING. En el sujeto 9, STANDING y SITTING tienen prácticamente la misma g_y (+0.008 y −0.023).
   - [HIPÓTESIS, reforzada en dos conjuntos independientes] el modelo separa estas posturas por la inclinación del teléfono, y la inclinación de cada sujeto (anatomía o colocación del teléfono) desplaza la frontera.
2. **Un nuevo modo de fallo: orientación fuera de distribución** [RESULTADO]. Las 27 ventanas LAYING → UPSTAIRS son todas del sujeto 10. Su gravedad media en LAYING es (0.02, −0.12, −0.04), con ventanas a g_y ≈ −0.93 g; en DEV, LAYING tiene g_y mínima de −0.48 y media de +0.65.
   - [HIPÓTESIS] el sujeto 10 se acostó en una orientación (de lado o boca abajo) no representada en entrenamiento, y la red asigna una clase arbitraria.
   - Muestra que el modelo **no es invariante a la orientación** del teléfono.
3. **Ventanas de transición** [HIPÓTESIS]. SITTING → UPSTAIRS (20, sobre todo del sujeto 10) y los ejemplos #45 y #790 tienen movimiento real dentro de ventanas etiquetadas como estáticas. Es coherente con transiciones posturales etiquetadas como la actividad siguiente. UCI HAR v1 no incluye etiquetas de transición; la versión HAPT sí, pero no se usó.
4. **Subir escaleras ↔ caminar / bajar escaleras**: 40 errores, con alta confianza (#1499, p = 0.99). La ventana de 2.56 s captura unos 4 pasos con firmas espectrales similares (Fase 2: pico de 1.56 Hz en ambas).
5. **Calibración**: confianza media de 0.95 en aciertos y de 0.79 en errores. Hay cierta separación, pero muchos errores tienen p > 0.9. La interfaz debe advertirlo.
6. **Generalización**: el TEST (0.914) está dentro del rango del HOLD-OUT con 3 semillas (0.910 ± 0.016) y de la CV (0.892 ± 0.037). No hay evidencia de sobreajuste a la selección de modelo.
   - La exactitud por sujeto en TEST va de 0.653 (sujeto 10) a 0.997 (sujeto 24): **la variabilidad entre sujetos es la principal fuente de error**.

## 8. Limitaciones adicionales para el artículo
- El TEST se evaluó con un solo modelo (semilla 42). La variabilidad entre semillas se estima solo en el HOLD-OUT.
- La comparación con la literatura en el mismo dataset debe hacerse con cautela: los protocolos de partición y de entrada difieren (señales crudas frente a 561 características).

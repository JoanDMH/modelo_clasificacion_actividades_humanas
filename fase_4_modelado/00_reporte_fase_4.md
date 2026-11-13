# Reporte de la Fase 4 – Modelado (tarea b)
**Proyecto 13 · CRISP-DM fase 4 · Estado: CERRADA (2026-11-13). Decisión: B como análisis de sensibilidad, luego A (LSTM), según el usuario. Ver §6 y §7**
Protocolo en `01_diseno_experimental.md` · Datos en `resultados_cv.csv` y `tabla_cv.csv` · Registro EXP-001 a EXP-012

## 1. Tabla del enunciado
CV estratificada por sujeto, 2 particiones × 3 semillas (42, 43, 44), n = 6 por arquitectura, evaluada sobre DEV (17 sujetos).

| Arquitectura | Exactitud (media ± desv. est.) | Macro-F1 (media ± desv. est.) | Parámetros |
|---|---|---|---|
| LSTM (64→32) | **0.892 ± 0.037** | 0.894 ± 0.037 | 31 558 |
| Conv1D + LSTM | **0.892 ± 0.064** | 0.889 ± 0.075 | 56 902 |

[RESULTADO | EXP-001 a EXP-006 y EXP-007 a EXP-012; W&B `group = lstm` / `conv_lstm`; código `src/` con el hash registrado; desv. est. con ddof = 1]

## 2. Resultados por run
| Semilla / fold | LSTM | Conv-LSTM | Diferencia (Conv − LSTM) | Runs W&B (LSTM / Conv) |
|---|---|---|---|---|
| 42 / 0 | 0.860 | 0.910 | +0.050 | 562v72lr / hor2cnkb |
| 42 / 1 | 0.918 | 0.920 | +0.002 | hqbx2trn / 5xpep7yp |
| 43 / 0 | 0.919 | 0.933 | +0.014 | rjw0kgwo / 2pqtlz2t |
| 43 / 1 | 0.857 | **0.777** | −0.080 | 3zzdgzlj / n8yu16s8 |
| 44 / 0 | 0.861 | 0.861 | 0.000 | 66vcnme4 / ucben3ku |
| 44 / 1 | 0.939 | 0.950 | +0.011 | 9arrgmvu / mlkpiphu |

- Diferencia pareada media: ≈ 0.000. Conv-LSTM es mejor en 4 de 6 pares, empata en 1 y pierde en 1.
- La desviación entre semillas (media por semilla) es 0.007 en LSTM y 0.032 en Conv-LSTM.

## 3. Análisis
1. **No hay diferencia demostrable entre arquitecturas** [RESULTADO]. Las medias son idénticas al tercer decimal y la dispersión (0.04–0.06) es unas 100 veces mayor que la diferencia. Con n = 6 no correlacionados no puede afirmarse superioridad de ninguna.
2. **El reparto de sujetos pesa más que la arquitectura** [RESULTADO]. El mismo fold produce exactitudes similares en ambos modelos (por ejemplo, 44/0: 0.861 y 0.861; 44/1: 0.939 y 0.950). Esto es coherente con el efecto sujeto de la Fase 2 (H4).
3. **Un valor atípico se debe a la parada temprana, no a la arquitectura** [RESULTADO | n8yu16s8]. En Conv-LSTM 43/1, la mínima `val_loss` se dio en la época 1 (0.539, val_acc 0.777), mientras que en épocas posteriores val_acc llegó a ~0.88 con mayor `val_loss`.
   - [HIPÓTESIS] El modelo se vuelve sobreconfiado (la pérdida castiga los errores con alta confianza) aunque acierte más. Monitorear `val_loss` con `restore_best_weights` puede devolver pesos peores en exactitud.
   - Sin ese run, la media de Conv-LSTM sería 0.915. **No se excluye**: excluirlo después de ver los resultados sería selección a posteriori.
4. **Hay sobreajuste temprano** [RESULTADO]. La exactitud de entrenamiento llega a 0.95–0.98 mientras la de validación se estanca en 0.86–0.95. La mejor época está entre 1 y 12; ningún run llegó a 50 épocas.
5. **La reproducibilidad está verificada** [RESULTADO]. La época 1 de `lstm-s42-f0` es idéntica en la prueba de humo y en la corrida real (acc 0.5278, loss 1.2715) y coincide con el entorno Linux.

## 4. Decisión pendiente (requiere al usuario)
| Opción | Qué implica | A favor | En contra |
|---|---|---|---|
| **A (recomendada)** | Elegir LSTM con el protocolo actual | Exactitud igual, la mitad de la varianza y 45 % menos parámetros (parsimonia). No cambia el protocolo después de ver datos | La CV no demuestra que sea mejor; la elección se justifica por estabilidad y simplicidad, no por exactitud |
| B | Repetir la CV de ambas con la parada temprana monitorizando `val_accuracy` y un mínimo de épocas | Corrige el mecanismo que causó el valor atípico | ~35 min más de CPU. Es un cambio de protocolo tras ver resultados: debe declararse como análisis de sensibilidad, no como resultado principal |

Para el modelo final (tarea c) hace falta además definir el número de épocas, porque DEV completo no tiene un conjunto propio para la parada temprana.
- [SUPUESTO] propuesto: usar épocas fijas iguales a la mediana de la mejor época en la CV de la arquitectura elegida (LSTM: {10, 8, 10, 6, 6, 4} → mediana **7**).
- Alternativa: separar 2 sujetos de DEV como validación interna.

## 5. Limitaciones para el artículo
- Hay sesgo optimista: la parada temprana usa la misma validación con la que se reporta la exactitud (afecta igual a ambas arquitecturas).
- Los folds de distintas semillas comparten sujetos, así que las 6 evaluaciones no son independientes. La desviación estándar es descriptiva, no un intervalo de confianza.
- No hubo búsqueda de hiperparámetros; ambas arquitecturas usan la configuración aprobada a priori.

## 6. Análisis de sensibilidad B: parada temprana sobre `val_accuracy` (EXP-013 a EXP-024)
Mismas particiones, semillas e hiperparámetros; solo cambia la métrica que monitoriza la parada temprana. W&B: `group = lstm-sens` / `conv_lstm-sens`.

| Arquitectura | Exactitud (media ± desv. est.) | Macro-F1 | Desv. entre semillas |
|---|---|---|---|
| LSTM | 0.908 ± 0.035 | 0.910 ± 0.034 | 0.010 |
| Conv1D + LSTM | 0.920 ± 0.025 | 0.923 ± 0.025 | 0.008 |

**Comparación pareada (Conv − LSTM, n = 6)** [RESULTADO, calculado desde `resultados_cv*.csv`]

| Protocolo | Diferencia media | Desv. est. | t pareada | t corregida (Nadeau–Bengio) | Wilcoxon |
|---|---|---|---|---|---|
| Principal (`val_loss`) | −0.0005 | 0.043 | p = 0.98 | p = 0.99 | p = 0.63 |
| Sensibilidad (`val_accuracy`) | +0.012 | 0.020 | p = 0.20 | p = 0.60 | p = 0.31 |

**Interpretación**
- Con `val_accuracy` el run atípico se corrige (Conv-LSTM 43/1: 0.777 → 0.881). Esto confirma que el valor atípico era un artefacto de la métrica de parada [RESULTADO | hfax76ib].
- Ambas arquitecturas suben (+1.5 y +2.8 puntos). Parte de esa mejora es **sesgo de selección**: ahora se elige la mejor época con la misma métrica y la misma partición con que se reporta. Por eso B es más optimista que el protocolo principal y **no reemplaza** la tabla del §1.
- Conv-LSTM gana en 5 de 6 pares, pero **la diferencia no es significativa** con ninguna prueba. La corrección de Nadeau–Bengio es la apropiada, porque los conjuntos de entrenamiento de distintas semillas se solapan.
- Con n = 6 la potencia estadística es baja; no puede descartarse una ventaja real de alrededor de 1 punto.

## 7. Decisión final de la fase
- **Arquitectura para el modelo final: LSTM.** Es la opción A, pre-especificada antes de ver B y confirmada por el usuario.
  - Justificación: ningún protocolo muestra una diferencia significativa; LSTM tiene 45 % menos parámetros y su varianza en el protocolo principal es menor.
  - El artículo reportará el §1 como resultado principal y el §6 como análisis de sensibilidad, con la observación de que Conv-LSTM fue numéricamente superior en B.
- **Épocas del modelo final: 7**, la mediana de la mejor época de LSTM en el protocolo principal (decisión del usuario).

## 8. Ratificación de la decisión (2026-11-13, tras revisar la literatura)
**Se mantiene LSTM.** Justificación, por principio de parsimonia:
- **Ganancia no demostrada.** Diferencia pareada de −0.0005 en el protocolo principal y +0.012 en la sensibilidad B, con p ≥ 0.20 en todas las pruebas y p = 0.60 con la corrección de Nadeau–Bengio [RESULTADO]. La literatura no establece una superioridad general de los modelos convolucional-recurrentes en HAR; el beneficio depende de las actividades y del conjunto de datos (Hammerla et al., 2016) [REF].
- **Complejidad no justificada a esta escala.** Conv1D + LSTM tiene un 80 % más de parámetros (56 902 frente a 31 558) y añade hiperparámetros de diseño (número de filtros, tamaño de kernel, pooling) que tendrían que ajustarse con solo 17 sujetos de DEV. Esto aumenta el riesgo de sobreajuste a la selección de modelo.
- **Precisión sobre el costo:** en tiempo de entrenamiento, Conv1D + LSTM fue **más rápida** en esta CPU (~7–10 s por época frente a ~11–15 s), porque el MaxPool reduce a 64 los pasos que procesa la LSTM. Por lo tanto, el argumento es de complejidad del modelo y de parsimonia, **no** de costo en tiempo. Ambos modelos son igual de viables para la interfaz (~125 KB frente a ~225 KB de pesos).
- **Coherencia del protocolo:** LSTM se pre-especificó antes del análisis B. Cambiar de modelo a partir del protocolo más optimista sería una decisión condicionada por los datos.
- **El error dominante (STANDING → SITTING) no es atribuible a la arquitectura** [HIPÓTESIS]: depende de la orientación del teléfono de cada sujeto (Fase 5, §4).

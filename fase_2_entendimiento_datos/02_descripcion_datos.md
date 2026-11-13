# 02 · Descripción de los datos
**Fase CRISP-DM:** 2 · **Tarea:** descripción · Todas las cifras provienen de `05_eda_notebook.ipynb`, exportadas en `resumen_eda.json`.

## 1. Volumen y forma
| Partición | Ventanas | Forma tensor | Sujetos | % ventanas | % sujetos |
|---|---|---|---|---|---|
| Entrenamiento | 7 352 | (7352, 128, 9) | 21 | 71.39 % | 70 % |
| Prueba | 2 947 | (2947, 128, 9) | 9 | 28.61 % | 30 % |

[DATO: archivos `Inertial Signals`, verificado en el notebook §1.]
- Sujetos de prueba: {2, 4, 9, 10, 12, 13, 18, 20, 24}. **Intersección con entrenamiento: vacía.**
- Ventanas por sujeto: mínimo 281, máximo 409, media 343.3.
- Todos los sujetos tienen las 6 actividades. El mínimo en entrenamiento por actividad y sujeto es 36 ventanas (WALKING_DOWNSTAIRS).

## 2. Distribución de clases
| Actividad (id) | Train n | Train % | Test n | Test % |
|---|---|---|---|---|
| WALKING (1) | 1226 | 16.68 | 496 | 16.83 |
| WALKING_UPSTAIRS (2) | 1073 | 14.59 | 471 | 15.98 |
| WALKING_DOWNSTAIRS (3) | 986 | 13.41 | 420 | 14.25 |
| SITTING (4) | 1286 | 17.49 | 491 | 16.66 |
| STANDING (5) | 1374 | 18.69 | 532 | 18.05 |
| LAYING (6) | 1407 | 19.14 | 537 | 18.22 |

- El desbalance es leve: la razón entre la clase mayor y la menor en entrenamiento es 1.43.
- [SUPUESTO] No se requiere remuestreo. Se reportarán métricas macro junto con la exactitud.

## 3. Calidad
| Verificación | Resultado |
|---|---|
| NaN / Inf | 0 / 0 en ambas particiones |
| Etiquetas | {1..6} en ambas particiones |
| Ventanas duplicadas exactas | 0 / 0 |
| Fuga de sujetos entre train y test | Ninguna (aserción en el notebook) |

## 4. Verificación del preprocesamiento declarado
**Solapamiento del 50 %**
- Se compararon pares de ventanas consecutivas del mismo sujeto y la misma actividad.
- En entrenamiento, el 95.48 % de los pares (6 752 de 7 072) comparte exactamente 64 muestras. En prueba, el 95.15 % (2 690 de 2 827).
- [HIPÓTESIS] El ~5 % restante corresponde a cortes entre grabaciones o repeticiones de una misma actividad, es decir, ventanas consecutivas en el archivo pero no en el tiempo. Debe confirmarse con `RawData/labels.txt` de la versión extendida (HAPT), que no está en el zip.

**Gravedad = total_acc − body_acc**
- El módulo de la gravedad estimada vale 1.0229 ± 0.0132 g (p1 = 0.990, p99 = 1.048), coherente con 1 g.
- Al quitar la componente continua, el 78.5 % de su energía está en ≤ 0.5 Hz; en `body_acc` esa fracción es solo 18.6 %.
- Esto es consistente con el filtro paso bajo declarado. La resolución espectral de una ventana es 0.39 Hz, así que no se puede verificar el corte exacto de 0.3 Hz.

**Conclusión:** los datos **sí** están preprocesados como se declara.
- Ya están filtrados, segmentados en ventanas y separados en cuerpo/gravedad.
- **No** están normalizados por canal. Las desviaciones estándar en entrenamiento van de 0.107 (`body_acc_z`) a 0.414 (`total_acc_x`), una razón de 3.88, y `total_acc_x` tiene media 0.80 g. La normalización es necesaria en la Fase 3.

## 5. Estadísticos por canal (entrenamiento; solo descriptivos)
| Canal | Media | Desv. | p1 | p99 |
|---|---|---|---|---|
| body_acc_x | −0.0006 | 0.1948 | −0.475 | 0.704 |
| body_acc_y | −0.0003 | 0.1224 | −0.430 | 0.312 |
| body_acc_z | −0.0003 | 0.1068 | −0.352 | 0.308 |
| body_gyro_x | 0.0005 | 0.4067 | −1.253 | 1.194 |
| body_gyro_y | −0.0008 | 0.3817 | −1.120 | 1.261 |
| body_gyro_z | 0.0001 | 0.2556 | −0.845 | 0.712 |
| total_acc_x | 0.8040 | 0.4141 | −0.166 | 1.693 |
| total_acc_y | 0.0288 | 0.3908 | −0.678 | 0.976 |
| total_acc_z | 0.0865 | 0.3577 | −0.752 | 0.921 |

Unidades: g para la aceleración y rad/s para el giroscopio [DATO: `README.txt`]. Los estadísticos de prueba están en `figuras/tabla_stats_canal_test.csv` y **no** se usarán para ninguna decisión.

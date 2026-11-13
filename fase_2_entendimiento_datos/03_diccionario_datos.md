# 03 · Diccionario de datos
**Fase CRISP-DM:** 2 · Fuente: [DATO: `README.txt`, `activity_labels.txt`] y verificación propia.

## 1. Tensor de entrada `X` — forma `(n_ventanas, 128, 9)`, `float32`
| Índice | Canal | Archivo origen (`{split}` = train/test) | Unidad | Descripción |
|---|---|---|---|---|
| 0 | body_acc_x | `Inertial Signals/body_acc_x_{split}.txt` | g | Aceleración corporal = total − gravedad (eje X) |
| 1 | body_acc_y | `.../body_acc_y_{split}.txt` | g | Ídem, eje Y |
| 2 | body_acc_z | `.../body_acc_z_{split}.txt` | g | Ídem, eje Z |
| 3 | body_gyro_x | `.../body_gyro_x_{split}.txt` | rad/s | Velocidad angular del giroscopio, eje X |
| 4 | body_gyro_y | `.../body_gyro_y_{split}.txt` | rad/s | Ídem, eje Y |
| 5 | body_gyro_z | `.../body_gyro_z_{split}.txt` | rad/s | Ídem, eje Z |
| 6 | total_acc_x | `.../total_acc_x_{split}.txt` | g | Aceleración total del acelerómetro (incluye gravedad), eje X |
| 7 | total_acc_y | `.../total_acc_y_{split}.txt` | g | Ídem, eje Y |
| 8 | total_acc_z | `.../total_acc_z_{split}.txt` | g | Ídem, eje Z |

- Cada fila de cada `.txt` es una ventana de 128 valores (2.56 s a 50 Hz).
- La fila *k* de los 9 archivos, de `y_{split}.txt` y de `subject_{split}.txt` corresponde a la misma ventana.
- El orden de los canales está fijado en `src/data_io.CHANNELS` y **debe mantenerse** en entrenamiento, normalización e interfaz.

## 2. Variables auxiliares
| Variable | Archivo | Tipo | Dominio | Uso |
|---|---|---|---|---|
| `y` | `{split}/y_{split}.txt` | int | 1–6 | Etiqueta objetivo |
| `subject` | `{split}/subject_{split}.txt` | int | 1–30 | **Grupo** para partición sin fuga; no es una entrada del modelo |

## 3. Etiquetas (`activity_labels.txt`)
| id | Nombre | Español | Tipo |
|---|---|---|---|
| 1 | WALKING | Caminar | Dinámica |
| 2 | WALKING_UPSTAIRS | Subir escaleras | Dinámica |
| 3 | WALKING_DOWNSTAIRS | Bajar escaleras | Dinámica |
| 4 | SITTING | Sentado | Estática |
| 5 | STANDING | De pie | Estática |
| 6 | LAYING | Acostado | Estática |

- [SUPUESTO] En los modelos, las etiquetas se re-indexarán a 0–5 (`y − 1`).
- El mapeo se guardará junto con el modelo para la interfaz.

## 4. Variables derivadas usadas solo en el EDA (no son entradas del modelo)
| Variable | Definición |
|---|---|
| g estimada | `total_acc − body_acc` (3 ejes) |
| `std_|body_acc|` | Desviación estándar del módulo de `body_acc` en la ventana (intensidad de movimiento) |
| `g_x, g_y, g_z` medias | Media de la gravedad estimada en la ventana (orientación del teléfono) |

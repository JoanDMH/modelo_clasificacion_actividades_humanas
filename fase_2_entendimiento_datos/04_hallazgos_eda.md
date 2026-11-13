# 04 · Hallazgos del EDA e implicaciones
**Fase CRISP-DM:** 2 → insumos para la Fase 3. Las figuras están en `figuras/`. Ninguna cifra de este documento es un resultado de modelo.

## H1. Datos íntegros y partición oficial sin fuga
- [DATO] No hay NaN, Inf ni duplicados.
- [DATO] 21 sujetos en entrenamiento y 9 en prueba, disjuntos.
- **Implicación:** se puede usar la partición oficial tal cual.

## H2. Los datos están preprocesados pero no normalizados
- [DATO] Ver `02_descripcion_datos.md` §4. Las escalas por canal difieren hasta 3.88 veces y `total_acc_x` está centrada en ~0.8 g.
- **Implicación (Fase 3):** se aplicará estandarización z por canal.
  - μ y σ se calcularán sobre todas las muestras y pasos de tiempo de las ventanas de **entrenamiento del fold**.
  - Luego se aplicarán a validación y prueba.
  - Se guardarán como artefacto para la interfaz.

## H3. Ventanas solapadas → riesgo de fuga si se divide por ventana
- [DATO] El ~95 % de las ventanas consecutivas comparte 64 de sus 128 muestras.
- **Implicación:** una partición aleatoria por ventana pondría la mitad de una señal en entrenamiento y la otra mitad en validación. Esto confirma la necesidad de `StratifiedGroupKFold` con `groups = subject`, con aserciones de disjunción.

## H4. Fuerte efecto sujeto
- [DATO] Dentro de cada actividad dinámica, el sujeto explica el 66–73 % de la varianza de la intensidad de movimiento (η² entre 0.664 y 0.734) y el 68–82 % de la de `g_x` media.
- **Implicaciones:**
  - Es otra razón para validar por sujeto.
  - Con solo 21 sujetos de entrenamiento y CV de 2 particiones (~10–11 sujetos por fold), se espera una **varianza alta entre folds**.
  - [SUPUESTO] Se propone repetir la CV con ≥ 3 semillas (pendiente de tu aprobación por el costo en CPU).

## H5. Separación dinámica / estática trivial; dentro de cada grupo, no
- [DATO] La desviación estándar de |body_acc| es de ~0.011–0.016 g en las clases estáticas y de 0.125–0.212 g en las dinámicas (fig06, izquierda).
- [DATO] Picos espectrales dominantes de |body_acc|:
  - WALKING y WALKING_UPSTAIRS: 1.56 Hz.
  - WALKING_DOWNSTAIRS: 3.52 Hz.
  - Clases estáticas: ≤ 0.39 Hz.
  - La resolución es 0.39 Hz (fig05).
- [HIPÓTESIS] Las confusiones del modelo se concentrarán dentro de cada grupo: entre caminar y subir escaleras (mismo pico de frecuencia), y entre las clases estáticas.

## H6. LAYING se distingue por orientación; SITTING y STANDING se solapan
- [DATO] g_x media: LAYING 0.07 g, SITTING 0.95 g, STANDING 1.00 g. Es decir, el teléfono está horizontal al acostarse.
- [DATO] El mejor umbral único en una sola variable separa SITTING de STANDING con un 88.8 % de exactitud usando g_y y un 55.4 % usando la intensidad de movimiento. Esta medida es optimista porque se calculó sobre todo el entrenamiento y es solo descriptiva.
- [HIPÓTESIS] El par SITTING–STANDING será la principal fuente de error, porque solo difiere por una leve inclinación de la cadera. Esto se analizará en la tarea (d).

## H7. Los canales `total_acc` son informativos
- **Implicación:** contienen la orientación (gravedad), que es la clave para las clases estáticas.
- [SUPUESTO] Se mantienen los 9 canales. Eliminar `total_acc` como "redundante" (porque body + g = total) le quitaría al modelo la información de gravedad, que no está en ningún otro canal.

## Decisiones que pasan a la Fase 3
| # | Decisión | Etiqueta |
|---|---|---|
| D1 | Usar los 9 canales en el orden de `CHANNELS` | [SUPUESTO] (H7) |
| D2 | Estandarización z por canal, ajustada solo con el entrenamiento de cada fold | Requisito del enunciado + H2 |
| D3 | CV con `StratifiedGroupKFold(n_splits=2)`, grupos = sujeto | Requisito + H3/H4 |
| D4 | Validación *hold-out* por sujeto para la tarea (c); prueba oficial solo para (d) y (f) | [SUPUESTO] aprobado por el usuario |
| D5 | Reportar exactitud junto con macro-F1, por el desbalance leve | [SUPUESTO] (§2 de 02) |
| D6 | Sin aumento de datos ni remuestreo en la línea base | [SUPUESTO] |

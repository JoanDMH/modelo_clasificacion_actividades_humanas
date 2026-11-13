# Clasificación de actividades humanas desde señales de acelerómetro y giroscopio

Proyecto 13 · Reconocimiento de actividad humana (HAR) con redes recurrentes sobre las **señales inerciales crudas** del conjunto [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) (30 sujetos, 6 actividades, ventanas de 2,56 s × 9 canales). Metodología CRISP-DM, evaluación **independiente del sujeto** y trazabilidad completa en Weights & Biases.

## Resultados principales

| Etapa | Métrica | Valor | Registro |
|---|---|---|---|
| Validación cruzada por sujeto (2 particiones × 3 semillas) | Exactitud LSTM | 0,892 ± 0,037 | EXP-001 a 006 |
| | Exactitud Conv1D+LSTM | 0,892 ± 0,064 | EXP-007 a 012 |
| Modelo final (LSTM) en validación con 4 sujetos reservados | Exactitud (3 semillas) | 0,910 ± 0,016 | EXP-025 a 027 |
| Prueba oficial (9 sujetos, evaluación única) | Exactitud / F1 macro | 0,914 / 0,914 | EXP-028 |

- La diferencia entre arquitecturas no es significativa: p = 0,99 con la prueba t corregida de Nadeau–Bengio.
- Se eligió la LSTM por parsimonia: tiene 31 558 parámetros, frente a 56 902 de la Conv1D+LSTM.
- El error dominante es la confusión entre *de pie* y *sentado*, concentrada en pocos sujetos y asociada a la inclinación del teléfono.
- El detalle está en los reportes de cada fase y en `articulo/main.pdf`.

## Estructura

```
src/                         Código compartido: carga, partición, normalización, modelos, métricas, registro
fase_2_entendimiento_datos/  EDA y verificación del preprocesamiento (reportes .md + notebook)
fase_3_preparacion_datos/    Partición por sujeto y normalización; particiones.json (congelado)
fase_4_modelado/             Validación cruzada (run_cv.py), cv_repeticiones.json (congelado), resultados
fase_5_evaluacion/           Modelo final (train_final.py), evaluación en prueba (eval_test.py), resultados
fase_6_despliegue/           Interfaz Streamlit + D3 (app/) y ventanas de ejemplo en CSV (muestras/)
models/                      Modelo desplegado: pesos, escalador z y ficha del modelo
articulo/                    Artículo LNCS (main.tex, main.pdf) y sus figuras
registro_experimentos.csv    Registro de los 28 experimentos (ID, configuración, métricas, run de W&B)
```

Cada carpeta `fase_*` abre con un `00_reporte_fase_N.md` que resume los hallazgos y las decisiones tomadas.

## Reproducción

1. **Entorno.** Usar Python 3.12; TensorFlow 2.21 no es compatible con 3.14.
   ```
   py -3.12 -m venv .venv
   .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Datos.** Descargar `UCI HAR Dataset.zip` desde UCI y descomprimirlo en `human+activity+recognition+using+smartphones/UCI HAR Dataset/`. No se versiona por su tamaño.
3. **Weights & Biases.** Ejecutar `wandb login` y ajustar `WANDB_ENTITY` y `WANDB_PROJECT` en `src/config.py`.
4. **Pipeline**, desde la raíz del proyecto:
   ```
   jupyter nbconvert --execute --to notebook --inplace fase_2_entendimiento_datos/05_eda_notebook.ipynb
   jupyter nbconvert --execute --to notebook --inplace fase_3_preparacion_datos/02_preparacion_notebook.ipynb
   python fase_4_modelado/run_cv.py                          # validación cruzada principal (12 runs)
   python fase_4_modelado/run_cv.py --monitor val_accuracy   # análisis de sensibilidad (12 runs)
   python fase_5_evaluacion/train_final.py                   # modelo final, 3 semillas
   python fase_5_evaluacion/eval_test.py                     # evaluación ÚNICA en la prueba oficial
   streamlit run fase_6_despliegue/app/app.py                # interfaz
   ```
   Todos los scripts aceptan `--smoke` para una prueba rápida sin registrar en W&B.

## Decisiones metodológicas clave

- **Partición por sujeto en todas las etapas.** Las ventanas se solapan un 50 % y el efecto sujeto es fuerte, así que partir por ventana filtraría información. Aserciones automáticas verifican que ningún sujeto aparezca en dos conjuntos.
- **Particiones congeladas.** `particiones.json` y `cv_repeticiones.json` se leen, no se recalculan, porque el orden aleatorio de `StratifiedGroupKFold` puede variar entre versiones de scikit-learn. `fase_4_modelado/gen_cv_repeticiones.py` documenta cómo se generaron.
- **Normalización z por canal** ajustada solo con los datos de entrenamiento de cada partición.
- **Prueba oficial usada una sola vez**, con el modelo elegido de antemano (semilla 42).

## Trazabilidad

- Cada fila de `registro_experimentos.csv` remite a un run de W&B: proyecto `clasificacion-actividades-humanas`, equipo `joan-martinez-universidad-de-los-llanos`, visibilidad de equipo.
- La columna `version_codigo` es un hash del contenido de `src/*.py`. Este repositorio es la referencia para el código de los scripts de cada fase.

## Datos y cita

UCI HAR se distribuye con licencia CC BY 4.0. Al usarlo, cite:

> Anguita, D., Ghio, A., Oneto, L., Parra, X., Reyes-Ortiz, J.L.: A public domain dataset for human activity recognition using smartphones. In: ESANN 2013, pp. 437–442 (2013).

## Autor

Joan David Martínez Hernández · Universidad de los Llanos

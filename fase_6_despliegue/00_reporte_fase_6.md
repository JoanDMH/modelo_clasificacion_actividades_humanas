# Reporte de la Fase 6 – Despliegue (tarea f)
**Proyecto 13 · CRISP-DM fase 6 · Estado: interfaz implementada y probada automáticamente; falta la validación del usuario en su equipo**
App: `app/app.py` (Streamlit) · Muestras: `muestras/` (12 ventanas del TEST, 2 por actividad)

## Ejecución
```
streamlit run fase_6_despliegue/app/app.py
```
Se ejecuta desde la raíz del proyecto, con `.venv` activo. Se abre en http://localhost:8501.

## Funcionalidad
| Requisito del enunciado | Implementación |
|---|---|
| Cargar un segmento de señal inercial del TEST | Tres orígenes: (1) índice de ventana del TEST oficial, 0–2946, con botón de ventana aleatoria; (2) muestras CSV incluidas; (3) CSV propio de 128 × 9 (con o sin encabezado `CHANNELS`, en g y rad/s), con validación de forma y de NaN |
| Mostrar la actividad predicha | Métrica con la clase predicha y su probabilidad. Si la ventana es del TEST, muestra la etiqueta real y si hubo acierto |
| Distribución de probabilidad por clase | Barras horizontales ordenadas y tabla |
| Extra | Gráfica de los 9 canales, descarga de la ventana en CSV, aviso de interpretación (no calibración, dependencia de la orientación, confusión de pie ↔ sentado) y ficha del modelo con métricas trazables (EXP-028) |

## Decisiones técnicas
- La entrada se espera **en unidades físicas, sin normalizar**. La app aplica el mismo `scaler.json` del entrenamiento, el que se ajustó con DEV.
- Se reconstruye la arquitectura y se cargan los pesos (`load_weights`) en lugar de `load_model`. Esto evita la incompatibilidad de deserialización entre versiones menores de Keras detectada en la Fase 5.
- El modelo y el TEST se cargan una sola vez gracias a `st.cache_resource` y `st.cache_data`.

## Verificación (2026-11-13, `streamlit.testing.AppTest`, Streamlit 1.5x en Linux)
| Caso | Esperado | Obtenido |
|---|---|---|
| Arranque | Sin excepciones | OK |
| TEST #0 | Coherente con `predicciones.csv` | De pie, p = 0.946, acierto |
| TEST #631 (ejemplo incorrecto de la tarea d) | SITTING, p = 0.744 | Sentado, p = 0.744, marcado como error ✔ |
| Muestra CSV incluida | Predicción y etiqueta real | De pie, p = 0.948, acierto |
| Modo "subir CSV" sin archivo | Mensaje de espera | OK |

La predicción de #631 coincide exactamente con la evaluación de la Fase 5, lo que confirma la consistencia entre el pipeline de evaluación y la interfaz.

## Pendiente
- El usuario debe ejecutarla en Windows (Streamlit 1.67) y tomar capturas para el artículo y la presentación.

## Limitaciones
- Demo local, no un despliegue productivo: no hay autenticación, ni monitoreo, ni procesamiento de flujo continuo (una ventana a la vez).
- Las probabilidades no están calibradas; se advierte en la interfaz.

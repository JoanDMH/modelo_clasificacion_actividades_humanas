# 01 · Recolección inicial de datos
**Fase CRISP-DM:** 2 – Entendimiento de los datos · **Tarea:** recolección inicial · **Fecha:** 2026-11-12

## 1. Fuente
| Campo | Valor |
|---|---|
| Conjunto | Human Activity Recognition Using Smartphones, v1.0 [DATO: `README.txt`] |
| Repositorio | UCI ML Repository, id 240 — https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones |
| Autores | Reyes-Ortiz, Anguita, Ghio, Oneto, Parra (Univ. de Génova / UPC) [DATO: `README.txt`] |
| Archivo local | `human+activity+recognition+using+smartphones/UCI HAR Dataset.zip` (60 999 314 bytes) |
| Licencia | CC BY 4.0 según la ficha UCI [REF-POR-VERIFICAR en la página de UCI] |

## 2. Protocolo de adquisición declarado [DATO: `README.txt`]
- 30 voluntarios de 19–48 años; teléfono Samsung Galaxy S II **en la cintura**.
- Acelerómetro y giroscopio triaxiales a **50 Hz** constantes; etiquetado manual a partir de video.
- Partición aleatoria **por voluntario**: 70 % de los sujetos para entrenamiento y 30 % para prueba.
- Preprocesamiento: filtros de ruido; separación cuerpo/gravedad con Butterworth paso bajo a 0.3 Hz; ventanas deslizantes de 2.56 s (128 muestras) con 50 % de solapamiento.

## 3. Extracción
- Se descomprimió en `human+activity+recognition+using+smartphones/UCI HAR Dataset/`, excluyendo `__MACOSX/` y `.DS_Store` (metadatos de macOS, no son datos).
- Se usan **únicamente** los 18 archivos de `train|test/Inertial Signals/` (9 canales × 2 particiones), junto con `y_*.txt`, `subject_*.txt` y `activity_labels.txt`.
- Las 561 características (`X_*.txt`) **no se usan**, por decisión del proyecto (Fase 1).
- Para acelerar las cargas siguientes se crea una caché `data_cache/{train,test}_raw.npz` (función `src/data_io.load_or_cache`). La caché es reproducible desde los `.txt` y no forma parte de los datos fuente.

## 4. Problemas encontrados en la recolección
- Ninguno bloqueante. Los `.txt` son texto plano separado por espacios y se cargan sin errores con `numpy.loadtxt`.
- La carpeta `documentos_fuente/` no existe. El usuario confirmó que la única fuente de requisitos es el enunciado del Proyecto 13.

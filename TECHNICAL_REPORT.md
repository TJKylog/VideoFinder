# 🎬 FindDuplicatedVideos — Reporte Técnico

> Documentación técnica sobre el pipeline de extracción de frames, generación de huellas digitales y comparación de similitud entre videos.

---

## 📐 Arquitectura General

El proyecto sigue un pipeline secuencial de 5 etapas:

```
Escaneo → Extracción de Frames → Hashing/Embedding → Comparación → Reporte
```

| Etapa | Módulo | Descripción |
|-------|--------|-------------|
| 1. Escaneo | `video_scanner.py` | Recorre carpeta y subcarpetas buscando archivos de video |
| 2. Caché | `fingerprint_cache.py` | Carga/guarda fingerprints ya procesados (pickle) |
| 3. Extracción | `frame_extractor.py` / `gpu_extractor.py` | Extrae frames y genera huellas digitales |
| 4. Comparación | `hash_comparator.py` | Compara todas las combinaciones de pares de videos |
| 5. Reporte | `report_generator.py` | Genera HTML con resultados |

---

## 🎞️ Extracción de Frames

### Método de Extracción

Se usa **OpenCV** (`cv2.VideoCapture`) para abrir cada archivo de video. La extracción de frames se realiza mediante **seek por tiempo en milisegundos** (`CAP_PROP_POS_MSEC`), que es significativamente más rápido que el seek por número de frame, especialmente en archivos grandes.

```
Inicio (0ms) → Frame 1 → +1000ms → Frame 2 → +1000ms → Frame 3 → ...
```

### Parámetros de Extracción

| Parámetro | Valor Default | Descripción |
|-----------|:------------:|-------------|
| `FRAME_INTERVAL_SECONDS` | `1.0` | Segundos entre cada frame extraído |
| `MAX_HASHES_PER_VIDEO` | `300` | Máximo de frames/hashes por video |
| `FRAME_RESIZE` | `32×32` | Resolución a la que se redimensiona cada frame (modo CPU) |
| `HASH_SIZE` | `8` | Tamaño del hash perceptual (8 → 64 bits) |

### Auto-ajuste de Intervalo

Si un video es largo y generaría más de `MAX_HASHES_PER_VIDEO` frames, el intervalo se incrementa automáticamente:

$$
\text{intervalo\_ajustado} = \frac{\text{duración\_video}}{\text{MAX\_HASHES\_PER\_VIDEO}}
$$

**Ejemplo:** Un video de 600 segundos con `MAX_HASHES_PER_VIDEO=300` e intervalo default de 1s generaría 600 hashes. Se auto-ajusta a intervalo = 2s para generar exactamente 300 hashes.

### Metadatos Capturados por Video

Cada `VideoFingerprint` almacena:
- Ruta del archivo
- Duración (segundos), FPS, conteo de frames
- Resolución (width × height)
- Lista de hashes/embeddings
- Tamaño del archivo (MB)
- Estado de error (si aplica)

---

## 🔐 Modo CPU — Hash Perceptual (pHash)

### Pipeline de Hashing

Para cada frame extraído se aplica el siguiente proceso:

```
Frame BGR (OpenCV)
  │
  ├─ 1. cv2.resize() → 32×32 px (INTER_AREA)
  │
  ├─ 2. cv2.cvtColor() → BGR a RGB
  │
  ├─ 3. PIL.Image.fromarray() → Imagen PIL
  │
  ├─ 4. imagehash.phash(hash_size=8) → Hash perceptual de 64 bits
  │
  └─ 5. hash.flatten().astype(uint8) → Array plano de 64 valores (0/1)
```

### ¿Qué es pHash?

El **hash perceptual** (pHash) es un algoritmo que genera una huella digital compacta de una imagen, resistente a:

- Cambios de resolución
- Compresión / recodificación
- Pequeños cambios de brillo/contraste

**Proceso interno de pHash (hash_size=8):**
1. Redimensiona la imagen a 32×32 píxeles en escala de grises
2. Aplica la **DCT** (Transformada Discreta del Coseno) — similar a JPEG
3. Toma el bloque de 8×8 frecuencias bajas (esquina superior izquierda)
4. Calcula la mediana de esos 64 valores
5. Genera 64 bits: `1` si el valor DCT ≥ mediana, `0` si no

**Resultado:** Un vector de 64 bits que representa la "esencia visual" del frame.

### Bibliotecas Utilizadas (CPU)

| Biblioteca | Versión Mín. | Función |
|-----------|:-----------:|---------|
| `opencv-python` | ≥ 4.8.0 | Lectura de video, resize, conversión de color |
| `Pillow` | ≥ 10.0.0 | Conversión a imagen PIL para imagehash |
| `imagehash` | ≥ 4.3.1 | Cálculo de pHash |
| `numpy` | ≥ 1.24.0 | Arrays de bits, operaciones vectorizadas |
| `tqdm` | ≥ 4.65.0 | Barras de progreso |

---

## 🚀 Modo GPU — Embeddings CNN (MobileNetV3)

### Red Neuronal

Se utiliza **MobileNetV3-Small** preentrenada en ImageNet:

| Propiedad | Valor |
|-----------|-------|
| Modelo | `torchvision.models.mobilenet_v3_small` |
| Parámetros | ~2.5 millones |
| Dimensión de embedding | **576** |
| Pesos | `MobileNet_V3_Small_Weights.DEFAULT` (ImageNet) |
| Clasificador | Removido (`nn.Identity`) — solo feature extractor |
| Normalización de salida | L2 (para similitud coseno) |

### Pipeline de Embedding

```
Frame BGR (OpenCV)
  │
  ├─ 1. cvtColor → RGB
  │
  ├─ 2. ToPILImage()
  │
  ├─ 3. Resize(256) + CenterCrop(224) — estándar ImageNet
  │
  ├─ 4. ToTensor() + Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])
  │
  ├─ 5. MobileNetV3-Small (sin clasificador) → vector de 576 dims
  │
  └─ 6. Normalización L2 → embedding unitario
```

### Backends GPU Soportados

| Backend | Plataforma | Prioridad |
|---------|-----------|:---------:|
| **MPS** (Metal Performance Shaders) | macOS Apple Silicon | 1 (más alta) |
| **CUDA** | NVIDIA GPUs | 2 |
| **CPU** | Cualquiera (fallback) | 3 |

### Procesamiento por Batches

Los frames se procesan en lotes de `GPU_BATCH_SIZE` (default: 32) para maximizar el throughput de la GPU. Toda la inferencia se ejecuta en modo `torch.no_grad()` para eficiencia.

### Bibliotecas Adicionales (GPU)

| Biblioteca | Función |
|-----------|---------|
| `torch` ≥ 2.0.0 | Backend de deep learning, tensor operations |
| `torchvision` ≥ 0.15.0 | MobileNetV3, transforms de ImageNet |

---

## 🔍 Comparación de Similitud

### Estrategia General

Se comparan **todos los pares posibles** de videos (combinaciones):

$$
\text{pares} = \binom{n}{2} = \frac{n \times (n-1)}{2}
$$

Para cada par, el video con **menos hashes** se designa como Video A (corto) y el otro como Video B (largo). Se busca qué porcentaje de frames de A tienen un frame similar en B.

### Modo CPU: Distancia Hamming

La **distancia Hamming** cuenta el número de bits diferentes entre dos hashes:

$$
d_H(a, b) = \sum_{i=1}^{64} a_i \oplus b_i
$$

#### Algoritmo Vectorizado

1. Se construyen **matrices de hashes**: `mat_a` (m × 64) y `mat_b` (n × 64)
2. Se calcula la **matriz completa de distancias** mediante XOR + sum con broadcasting de NumPy:
   ```
   XOR: (m, 1, 64) ⊕ (1, n, 64) → (m, n, 64)
   Distancias: sum(axis=2) → (m, n)
   ```
3. Para cada hash de A, se toma la **distancia mínima** al hash más cercano de B
4. Si `min_distance ≤ HAMMING_THRESHOLD` → se cuenta como **match**

#### Optimización para Matrices Grandes

Si la matriz de distancias excedería 50 millones de elementos, se procesa en **bloques** para evitar consumo excesivo de memoria:

$$
\text{block\_size} = \left\lfloor \frac{50{,}000{,}000}{|B|} \right\rfloor
$$

#### Parámetros de Hamming

| Parámetro | Default | Descripción |
|-----------|:-------:|-------------|
| `HAMMING_THRESHOLD` | `10` | Distancia máxima para considerar dos frames iguales |
| Rango recomendado (hash_size=8) | 5–12 | Menor = más estricto |
| Rango recomendado (hash_size=16) | 10–25 | Más bits → umbral mayor |

### Modo GPU: Similitud Coseno

Para embeddings CNN normalizados L2, la **similitud coseno** se reduce a un **producto punto**:

$$
\text{sim}(\vec{a}, \vec{b}) = \vec{a} \cdot \vec{b} = \sum_{i=1}^{576} a_i \times b_i
$$

Dado que los vectores están normalizados ($\|\vec{a}\| = \|\vec{b}\| = 1$), el resultado está en el rango $[-1, 1]$, donde:
- **1.0** = idénticos
- **0.0** = sin relación
- **-1.0** = opuestos

#### Algoritmo

1. **Matrices de embeddings**: `mat_a` (m × 576) y `mat_b` (n × 576)
2. **Producto matricial**: `mat_a @ mat_b.T` → matriz de similitudes (m × n)
3. Para cada embedding de A: **máxima similitud** con algún embedding de B
4. Si `max_similarity ≥ COSINE_THRESHOLD` → **match**

#### Parámetros de Coseno

| Parámetro | Default | Descripción |
|-----------|:-------:|-------------|
| `COSINE_THRESHOLD` | `0.85` | Similitud mínima para considerar frames iguales |
| 0.90 | — | Casi idéntico |
| 0.85 | — | Muy similar |
| 0.75 | — | Algo parecido |

---

## 📊 Resultado de la Comparación

Cada par evaluado produce un `MatchResult` con:

| Campo | Descripción |
|-------|-------------|
| `matched_frames` | Frames de A que encontraron coincidencia en B |
| `total_frames_a` | Total de frames (hashes) del video más corto |
| `match_ratio` | `matched_frames / total_frames_a` (0.0 a 1.0) |
| `avg_hamming` | Distancia Hamming promedio de los matches (CPU) / (1 − similitud) × 100 (GPU) |

### Criterio de Duplicado

Un par se declara **duplicado/contenido** si:

$$
\text{match\_ratio} \geq \text{MATCH\_RATIO\_THRESHOLD}
$$

**Default:** `MATCH_RATIO_THRESHOLD = 0.40` (40% de los frames del video corto deben coincidir).

---

## 💾 Sistema de Caché

La caché almacena fingerprints procesados en disco usando **pickle**. La clave es un compuesto de:

```
clave = ruta_absoluta | tamaño_archivo | fecha_modificación
```

Si el archivo cambia (tamaño o fecha), se re-procesa automáticamente.

| Parámetro | Default | Descripción |
|-----------|:-------:|-------------|
| `CACHE_ENABLED` | `True` | Activar/desactivar caché |
| `CACHE_FILENAME` | `.fingerprint_cache.pkl` | Nombre del archivo de caché |

---

## ⚡ Paralelismo

| Modo | Estrategia | Workers |
|------|-----------|:-------:|
| **CPU** | `ProcessPoolExecutor` | Hasta 8 (auto) |
| **GPU** | Secuencial (la GPU ya paraleliza internamente) | 1 |

En modo CPU, cada proceso extrae frames y calcula hashes de un video independientemente. En modo GPU, los frames de cada video se procesan en batches por la CNN.

---

## 🎯 Extensiones de Video Soportadas

```
.mp4  .avi  .mkv  .mov  .wmv  .flv  .webm
.m4v  .mpg  .mpeg .3gp  .ogv  .ts   .vob
```

---

## 📈 Comparativa CPU vs GPU

| Aspecto | CPU (pHash) | GPU (MobileNetV3) |
|---------|:-----------:|:-----------------:|
| Dimensión del descriptor | 64 bits | 576 floats |
| Métrica de similitud | Distancia Hamming | Similitud Coseno |
| Velocidad | Moderada | 5–10× más rápido |
| Precisión | Buena para duplicados exactos | Superior para variaciones |
| Dependencias | opencv + imagehash | + torch + torchvision |
| Resistencia a ediciones | Media | Alta |
| Memoria | Baja | Alta (modelo + tensors en VRAM) |

---

## 🔧 Resumen de Configuración por Defecto

```python
# Extracción
FRAME_INTERVAL_SECONDS = 1.0      # 1 frame/segundo
FRAME_RESIZE            = (32, 32) # resize para pHash
HASH_SIZE               = 8        # pHash de 64 bits
MAX_HASHES_PER_VIDEO    = 300      # límite de frames

# Comparación CPU
HAMMING_THRESHOLD       = 10       # distancia máx entre frames
MATCH_RATIO_THRESHOLD   = 0.40     # 40% mínimo de coincidencia

# GPU
USE_GPU                 = False    # desactivado por defecto
GPU_BATCH_SIZE          = 32       # frames por batch
COSINE_THRESHOLD        = 0.85     # similitud mínima coseno
```

---

*Generado automáticamente para el proyecto FindDuplicatedVideos.*

"""
Configuración global del proyecto FindDuplicatedVideos.
Ajusta estos valores según tus necesidades.
"""

# ─── Extensiones de video soportadas ────────────────────────────────────────
VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".mpg", ".mpeg", ".3gp", ".ogv", ".ts", ".vob",
}

# ─── Extracción de frames ───────────────────────────────────────────────────
# Intervalo en segundos entre cada frame extraído.
# Un valor menor = más precisión pero más lento.
FRAME_INTERVAL_SECONDS = 1.0

# Tamaño al que se redimensionan los frames antes de calcular el hash.
FRAME_RESIZE = (32, 32)

# Tamaño del hash perceptual (bits = hash_size^2).
# 8  →  64 bits  — rápido,  buena precisión para duplicados.
# 16 → 256 bits  — lento,   mayor precisión.
HASH_SIZE = 8

# Máximo de hashes (frames) por video.
# Para videos muy largos, se aumenta automáticamente el intervalo
# para no superar este límite. 0 = sin límite.
MAX_HASHES_PER_VIDEO = 300

# ─── Comparación ────────────────────────────────────────────────────────────
# Distancia Hamming máxima para considerar dos frames como "iguales".
# Cuanto menor, más estricto. Rango recomendado:
#   hash_size=8:  5-12
#   hash_size=16: 10-25
HAMMING_THRESHOLD = 10

# Porcentaje mínimo de frames coincidentes del video más corto
# para declarar que un video "contiene" a otro.
# 0.0 – 1.0  (0.40 = 40% de los frames del video corto deben coincidir)
MATCH_RATIO_THRESHOLD = 0.40

# ─── Caché ──────────────────────────────────────────────────────────────────
# Activar caché de fingerprints en disco.
# Así al re-ejecutar se saltan los videos ya procesados.
CACHE_ENABLED = True
CACHE_FILENAME = ".fingerprint_cache.pkl"

# ─── GPU ────────────────────────────────────────────────────────────────────
# Activar modo GPU (requiere PyTorch + torchvision).
# Usa una CNN (MobileNetV3) para generar embeddings en vez de pHash.
# Backends: MPS (macOS Apple Silicon), CUDA (NVIDIA), CPU (fallback).
USE_GPU = False

# Tamaño de batch para procesar frames en GPU.
# Mayor batch = más rápido pero más VRAM. Ajustar si se queda sin memoria.
GPU_BATCH_SIZE = 32

# Umbral de similitud coseno para modo GPU (0.0 – 1.0).
# Los embeddings CNN se comparan por coseno, no por Hamming.
# 0.85 = muy similar, 0.90 = casi idéntico, 0.75 = algo parecido.
COSINE_THRESHOLD = 0.85

# ─── Paralelismo ────────────────────────────────────────────────────────────
# Número de workers para procesar videos en paralelo.
# None  = usa todos los CPUs disponibles (hasta 8).
# En modo GPU se usa 1 worker (la GPU ya paraleliza internamente).
MAX_WORKERS = None

# ─── Reporte ────────────────────────────────────────────────────────────────
# (El reporte HTML fue eliminado; los resultados se guardan en SQLite)

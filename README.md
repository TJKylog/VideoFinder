# 🎬 FindDuplicatedVideos

Herramienta para detectar **videos duplicados** o que **contienen fragmentos de otros videos** en una carpeta y subcarpetas.

## ¿Cómo funciona?

1. **Escaneo recursivo**: Recorre la carpeta indicada y todas sus subcarpetas buscando archivos de video (.mp4, .avi, .mkv, .mov, etc.)
2. **Extracción de frames**: Extrae un frame a intervalos regulares (por defecto cada 1 segundo) de cada video
3. **Hash perceptual (pHash)**: Calcula un "hash visual" de cada frame que resiste redimensionamiento, recompresión y pequeños cambios
4. **Comparación inteligente**: Compara los hashes entre todos los pares de videos usando:
   - **Ventana deslizante**: Encuentra la mejor alineación temporal (detecta si un clip corto es parte de uno largo)
   - **Búsqueda greedy**: Busca frames similares sin importar el orden (detecta ediciones/reordenamientos)
5. **Reporte HTML**: Genera un reporte visual con todos los duplicados encontrados

## Instalación

```bash
# Clonar o descargar el proyecto
cd FindDuplicatedVideos

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Básico
```bash
python main.py /ruta/a/carpeta/de/videos
```

### Con opciones
```bash
# Más preciso (intervalo de 0.5s entre frames)
python main.py ~/Videos -i 0.5

# Más permisivo (umbral de similitud más bajo)
python main.py ~/Videos -t 0.30

# Más estricto (distancia Hamming menor)
python main.py ~/Videos -H 8

# Combinar opciones
python main.py ~/Videos -i 0.5 -H 10 -t 0.45 -w 4

# Sin reporte HTML
python main.py ~/Videos --no-report

# Guardar reporte en otra carpeta
python main.py ~/Videos -o ~/Desktop
```

### Opciones disponibles

| Opción | Corto | Default | Descripción |
|--------|-------|---------|-------------|
| `folder` | — | — | Carpeta raíz donde buscar videos |
| `--interval` | `-i` | `1.0` | Segundos entre frames extraídos |
| `--hamming` | `-H` | `12` | Distancia Hamming máxima (menor = más estricto) |
| `--threshold` | `-t` | `0.40` | Porcentaje mínimo de similitud (0.0 – 1.0) |
| `--workers` | `-w` | auto | Procesos paralelos para extracción |
| `--output` | `-o` | carpeta de videos | Carpeta de salida para el reporte |
| `--no-report` | — | — | No generar el reporte HTML |

## Configuración avanzada

Edita `config.py` para ajustar valores por defecto:

- `VIDEO_EXTENSIONS`: Extensiones de video soportadas
- `FRAME_INTERVAL_SECONDS`: Intervalo entre frames (menor = más preciso, más lento)
- `HASH_SIZE`: Tamaño del hash (mayor = más preciso, más lento)
- `HAMMING_THRESHOLD`: Tolerancia de diferencia entre hashes
- `MATCH_RATIO_THRESHOLD`: % mínimo de frames coincidentes para declarar duplicado

## Guía de umbrales

| Escenario | `--interval` | `--hamming` | `--threshold` |
|-----------|:---:|:---:|:---:|
| Búsqueda rápida | 2.0 | 15 | 0.35 |
| **Balanceado (default)** | **1.0** | **12** | **0.40** |
| Alta precisión | 0.5 | 8 | 0.50 |
| Videos muy similares | 0.5 | 6 | 0.60 |

## ¿Qué detecta?

- ✅ Videos exactamente iguales (mismo archivo, diferente nombre)
- ✅ Videos recomprimidos (distinta calidad/resolución)
- ✅ Un video corto que es un fragmento de uno más largo
- ✅ Videos recortados (crop) o con marcos
- ✅ Videos con pequeñas diferencias de color/brillo
- ⚠️ Videos con ediciones significativas (poca detección)
- ❌ Videos completamente diferentes pero del mismo tema

## Estructura del proyecto

```
FindDuplicatedVideos/
├── main.py               # Punto de entrada (CLI)
├── config.py             # Configuración global
├── video_scanner.py      # Escaneo de archivos de video
├── frame_extractor.py    # Extracción de frames y cálculo de hashes
├── hash_comparator.py    # Comparación de hashes entre videos
├── report_generator.py   # Generación de reporte HTML
├── requirements.txt      # Dependencias de Python
└── README.md             # Este archivo
```

## Dependencias

- **Python 3.10+**
- **OpenCV** – Lectura de videos y extracción de frames
- **Pillow** – Procesamiento de imágenes
- **imagehash** – Cálculo de hashes perceptuales
- **NumPy** – Operaciones numéricas
- **tqdm** – Barras de progreso

## Licencia

MIT

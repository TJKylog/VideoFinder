"""
Módulo para escanear carpetas y subcarpetas en busca de archivos de video.
"""

import os
from pathlib import Path
from typing import List

from config import VIDEO_EXTENSIONS


def scan_videos(root_folder: str) -> List[Path]:
    """
    Recorre *root_folder* y todas sus subcarpetas y devuelve una lista
    de rutas absolutas a archivos de video reconocidos.

    Parameters
    ----------
    root_folder : str
        Ruta a la carpeta raíz donde comenzar la búsqueda.

    Returns
    -------
    List[Path]
        Lista ordenada de archivos de video encontrados.
    """
    root = Path(root_folder).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"La ruta no es un directorio válido: {root}")

    videos: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            # Ignorar archivos de metadatos de macOS (._*)
            if fname.startswith("._"):
                continue
            if Path(fname).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(Path(dirpath) / fname)

    videos.sort()
    return videos

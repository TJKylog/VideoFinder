"""
Sistema de caché para fingerprints de video.

Guarda los fingerprints procesados en disco (pickle) para que al
re-ejecutar el análisis se salten los videos ya procesados.

La clave de caché es:  ruta_absoluta + tamaño_archivo + fecha_modificación
Así si el archivo cambia, se re-procesa automáticamente.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

from config import CACHE_ENABLED, CACHE_FILENAME
from frame_extractor import VideoFingerprint


def _file_key(video_path: Path) -> str:
    """
    Genera una clave única para un archivo de video basada en:
      - Ruta absoluta
      - Tamaño del archivo
      - Fecha de última modificación
    """
    try:
        stat = video_path.stat()
        return f"{video_path.resolve()}|{stat.st_size}|{stat.st_mtime}"
    except OSError:
        return str(video_path.resolve())


class FingerprintCache:
    """Caché en disco para VideoFingerprint."""

    def __init__(self, cache_dir: str):
        self.cache_path = Path(cache_dir) / CACHE_FILENAME
        self._data: Dict[str, VideoFingerprint] = {}
        self._dirty = False

        if CACHE_ENABLED:
            self._load()

    def _load(self) -> None:
        """Carga la caché desde disco."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    self._data = pickle.load(f)
            except Exception:
                # Caché corrupta, empezar de cero
                self._data = {}

    def save(self) -> None:
        """Guarda la caché a disco (solo si hubo cambios)."""
        if not CACHE_ENABLED or not self._dirty:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
            self._dirty = False
        except Exception as exc:
            print(f"[AVISO] No se pudo guardar la cache: {exc}")

    def get(self, video_path: Path) -> Optional[VideoFingerprint]:
        """
        Devuelve el fingerprint cacheado si existe y el archivo no cambió.
        """
        if not CACHE_ENABLED:
            return None
        key = _file_key(video_path)
        fp = self._data.get(key)
        if fp is not None:
            # Verificar que el archivo sigue existiendo
            if not video_path.exists():
                del self._data[key]
                self._dirty = True
                return None
            return fp
        return None

    def put(self, video_path: Path, fp: VideoFingerprint) -> None:
        """Guarda un fingerprint en la caché."""
        if not CACHE_ENABLED:
            return
        key = _file_key(video_path)
        self._data[key] = fp
        self._dirty = True

    @property
    def size(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        """Limpia toda la caché."""
        self._data.clear()
        self._dirty = True
        if self.cache_path.exists():
            self.cache_path.unlink()

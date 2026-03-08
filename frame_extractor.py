"""
Extrae frames de un video a intervalos regulares y calcula
su hash perceptual (pHash) para comparaciones posteriores.

Optimizaciones clave:
  - Seek por tiempo (ms) en lugar de por frame (mucho más rápido).
  - Límite máximo de hashes por video (auto-ajusta el intervalo).
  - Resize con OpenCV (más rápido que PIL).
  - average_hash como opción rápida.
"""

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import imagehash
import numpy as np
from PIL import Image

from config import (
    FRAME_INTERVAL_SECONDS,
    FRAME_RESIZE,
    HASH_SIZE,
    MAX_HASHES_PER_VIDEO,
)

# Máximo de lecturas fallidas consecutivas antes de abortar un video
_MAX_CONSECUTIVE_FAILURES = 10


@contextmanager
def suppress_stderr():
    """Suprime stderr (warnings de ffmpeg/libav) mientras se usa OpenCV."""
    stderr_fd = sys.stderr.fileno()
    old_stderr_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(old_stderr_fd, stderr_fd)
        os.close(old_stderr_fd)
        os.close(devnull)


@dataclass
class VideoFingerprint:
    """Huella digital de un video: metadatos + hashes de sus frames."""
    path: Path
    duration_seconds: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    hashes: List[np.ndarray] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def filesize_mb(self) -> float:
        try:
            return self.path.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0

    def __repr__(self) -> str:
        return (
            f"VideoFingerprint({self.path.name}, "
            f"dur={self.duration_seconds:.1f}s, "
            f"hashes={len(self.hashes)})"
        )


def _compute_hash(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Calcula el hash perceptual de un frame BGR de OpenCV.
    Devuelve un array de bits (uint8, 0/1) para comparación rápida con numpy.
    """
    # Redimensionar con OpenCV (mucho más rápido que PIL)
    small = cv2.resize(frame_bgr, FRAME_RESIZE, interpolation=cv2.INTER_AREA)
    # BGR → RGB → PIL
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    h = imagehash.phash(pil_img, hash_size=HASH_SIZE)
    # Convertir a array plano de bits para comparación vectorizada
    return h.hash.flatten().astype(np.uint8)


def extract_fingerprint(video_path: Path) -> VideoFingerprint:
    """
    Abre un archivo de video, extrae frames a intervalos regulares,
    calcula el pHash de cada uno y devuelve un ``VideoFingerprint``.

    Usa seek por milisegundos (CAP_PROP_POS_MSEC) que es mucho más
    rápido que seek por frame, especialmente en archivos grandes.

    Si el video generaría más de MAX_HASHES_PER_VIDEO hashes, se
    aumenta automáticamente el intervalo para no exceder ese límite.
    """
    fp = VideoFingerprint(path=video_path)

    with suppress_stderr():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            fp.error = "No se pudo abrir el archivo de video."
            return fp

        try:
            fp.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fp.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fp.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fp.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fp.duration_seconds = fp.frame_count / fp.fps if fp.fps > 0 else 0.0

            # Validar metadatos: descartar videos con datos inválidos
            if fp.duration_seconds <= 0 or fp.width == 0 or fp.height == 0:
                fp.error = "Video con metadatos inválidos (duración/resolución 0)."
                return fp

            # Calcular intervalo real (auto-ajustar si excedería el máximo)
            interval = FRAME_INTERVAL_SECONDS
            if MAX_HASHES_PER_VIDEO > 0 and fp.duration_seconds > 0:
                estimated_hashes = fp.duration_seconds / interval
                if estimated_hashes > MAX_HASHES_PER_VIDEO:
                    interval = fp.duration_seconds / MAX_HASHES_PER_VIDEO

            # Seek por tiempo (ms) — mucho más rápido que por frame
            current_ms = 0.0
            interval_ms = interval * 1000.0
            end_ms = fp.duration_seconds * 1000.0
            consecutive_failures = 0

            while current_ms < end_ms:
                cap.set(cv2.CAP_PROP_POS_MSEC, current_ms)
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                        # Video probablemente corrupto/truncado
                        if not fp.hashes:
                            fp.error = "Video corrupto: no se pudo leer ningún frame."
                        break
                    current_ms += interval_ms
                    continue

                consecutive_failures = 0
                h = _compute_hash(frame)
                fp.hashes.append(h)

                current_ms += interval_ms

                # Límite de seguridad
                if MAX_HASHES_PER_VIDEO > 0 and len(fp.hashes) >= MAX_HASHES_PER_VIDEO:
                    break

        except Exception as exc:
            fp.error = str(exc)
        finally:
            cap.release()

    return fp

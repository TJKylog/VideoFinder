"""
Extractor de fingerprints usando GPU con PyTorch + CNN preentrenada.

Usa MobileNetV3-Small para generar embeddings densos de 576 dimensiones
por frame. Es ~5-10x más rápido que pHash cuando hay GPU disponible y
produce embeddings de mayor calidad para detección de duplicados.

Optimizaciones:
  - Pipeline con ThreadPoolExecutor: lectura de video (I/O) en paralelo
    mientras la GPU procesa el batch anterior.
  - Procesamiento de frames en batches grandes.

Backends GPU soportados:
  - MPS  (macOS Apple Silicon)
  - CUDA (NVIDIA)
  - CPU  (fallback automático)
"""

import gc
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import (
    FRAME_INTERVAL_SECONDS,
    MAX_HASHES_PER_VIDEO,
    GPU_BATCH_SIZE,
)

# Máximo de lecturas fallidas consecutivas antes de abortar un video
_MAX_CONSECUTIVE_FAILURES = 10


@contextmanager
def _suppress_stderr():
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

# Importación late de torch — solo cuando se usa este módulo
_torch_loaded = False
_torch = None
_transforms = None
_models = None
_device = None
_model = None
_preprocess = None


def _lazy_load_torch() -> str:
    """
    Carga PyTorch y el modelo de forma lazy.
    Devuelve el nombre del dispositivo usado.
    """
    global _torch_loaded, _torch, _transforms, _models
    global _device, _model, _preprocess

    if _torch_loaded:
        return str(_device)

    try:
        import torch
        import torchvision.transforms as transforms
        import torchvision.models as models
    except ImportError:
        raise ImportError(
            "Para usar el modo GPU necesitas instalar PyTorch:\n"
            "  pip install torch torchvision\n\n"
            "macOS Apple Silicon:  pip install torch torchvision\n"
            "NVIDIA CUDA:          pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu121"
        )

    _torch = torch
    _transforms = transforms
    _models = models

    # Seleccionar dispositivo: MPS (macOS) > CUDA (NVIDIA) > CPU
    if torch.backends.mps.is_available():
        _device = torch.device("mps")
    elif torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    # Cargar MobileNetV3-Small (muy ligero: ~2.5M parámetros)
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    _model = models.mobilenet_v3_small(weights=weights)
    # Quitar el clasificador: nos quedamos con el feature extractor
    # La salida del avgpool es un vector de 576 dims
    _model.classifier = _torch.nn.Identity()
    _model = _model.to(_device)
    _model.eval()

    # Preprocesamiento estándar de ImageNet
    _preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    _torch_loaded = True
    return str(_device)


def get_gpu_device_name() -> str:
    """Devuelve el nombre del dispositivo GPU disponible (sin cargar el modelo)."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "MPS (Apple Metal)"
        elif torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        else:
            return "CPU (sin GPU detectada)"
    except ImportError:
        return "No disponible (PyTorch no instalado)"


def _extract_frames(video_path: Path) -> Tuple[List[np.ndarray], dict]:
    """
    Extrae los frames crudos de un video (sin procesarlos todavía).
    Devuelve (frames_rgb, metadata).
    """
    with _suppress_stderr():
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("No se pudo abrir el archivo de video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0

        metadata = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": duration,
        }

        # Validar metadatos: descartar videos con datos inválidos
        if duration <= 0 or width == 0 or height == 0:
            cap.release()
            raise RuntimeError("Video con metadatos inválidos (duración/resolución 0).")

        # Calcular intervalo (auto-ajustar si excedería el máximo)
        interval = FRAME_INTERVAL_SECONDS
        if MAX_HASHES_PER_VIDEO > 0 and duration > 0:
            estimated = duration / interval
            if estimated > MAX_HASHES_PER_VIDEO:
                interval = duration / MAX_HASHES_PER_VIDEO

        frames = []
        current_ms = 0.0
        interval_ms = interval * 1000.0
        end_ms = duration * 1000.0
        consecutive_failures = 0

        try:
            while current_ms < end_ms:
                cap.set(cv2.CAP_PROP_POS_MSEC, current_ms)
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                        break  # Video probablemente corrupto/truncado
                    current_ms += interval_ms
                    continue

                consecutive_failures = 0
                # BGR → RGB
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                current_ms += interval_ms
                if MAX_HASHES_PER_VIDEO > 0 and len(frames) >= MAX_HASHES_PER_VIDEO:
                    break
        finally:
            cap.release()

    if not frames:
        raise RuntimeError("Video corrupto: no se pudo leer ningún frame.")

    return frames, metadata


def _frames_to_embeddings(frames: List[np.ndarray]) -> List[np.ndarray]:
    """
    Procesa una lista de frames RGB y devuelve embeddings usando la CNN en GPU.
    Procesa en batches para eficiencia de GPU.
    """
    if not frames:
        return []

    device_name = _lazy_load_torch()

    # Preprocesar todos los frames
    tensors = []
    for frame in frames:
        t = _preprocess(frame)
        tensors.append(t)

    embeddings = []
    batch_size = GPU_BATCH_SIZE

    with _torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = _torch.stack(tensors[i:i + batch_size]).to(_device)
            out = _model(batch)  # (batch, 576)
            # Normalizar L2 para usar similitud coseno después
            out = out / out.norm(dim=1, keepdim=True)
            embeddings.append(out.cpu().numpy())
            # Liberar tensor de GPU inmediatamente
            del batch, out

    # Liberar tensores preprocesados
    del tensors

    # Concatenar y separar en lista
    all_emb = np.concatenate(embeddings, axis=0)  # (n_frames, 576)
    del embeddings
    return [all_emb[i] for i in range(all_emb.shape[0])]


# Reutilizar el dataclass de frame_extractor
from frame_extractor import VideoFingerprint


def _clear_gpu_cache():
    """Limpia la caché de memoria GPU (MPS/CUDA) y ejecuta garbage collector."""
    gc.collect()
    if _torch is not None:
        if _device is not None and _device.type == "cuda":
            _torch.cuda.empty_cache()
        elif _device is not None and _device.type == "mps":
            try:
                _torch.mps.empty_cache()
            except AttributeError:
                pass


def extract_fingerprint_gpu(video_path: Path) -> VideoFingerprint:
    """
    Extrae fingerprint de un video usando GPU (CNN).
    Los "hashes" son embeddings de 576 dimensiones (float32 normalizado L2).
    """
    fp = VideoFingerprint(path=video_path)

    try:
        frames, meta = _extract_frames(video_path)
        fp.fps = meta["fps"]
        fp.frame_count = meta["frame_count"]
        fp.width = meta["width"]
        fp.height = meta["height"]
        fp.duration_seconds = meta["duration_seconds"]

        fp.hashes = _frames_to_embeddings(frames)

        del frames
        _clear_gpu_cache()

    except Exception as exc:
        fp.error = str(exc)
        _clear_gpu_cache()

    return fp


# ─── Pipeline: lectura paralela + GPU ────────────────────────────────────────

def extract_fingerprints_gpu_pipeline(
    video_paths: List[Path],
    on_complete=None,
    read_workers: int = 4,
) -> List[VideoFingerprint]:
    """
    Procesa múltiples videos con pipeline paralelo:
      - ThreadPool (read_workers hilos) lee videos del disco (I/O bound)
      - Mientras tanto, la GPU procesa los frames del video anterior

    Parameters
    ----------
    video_paths : List[Path]
        Videos a procesar.
    on_complete : callable, optional
        Callback(fp: VideoFingerprint) llamado después de procesar cada video.
    read_workers : int
        Hilos para lectura paralela de video (default: 4).

    Returns
    -------
    List[VideoFingerprint]
    """
    _lazy_load_torch()
    results: List[VideoFingerprint] = []

    # Usamos un ThreadPool para pre-leer videos mientras la GPU trabaja
    with ThreadPoolExecutor(max_workers=read_workers) as reader_pool:
        # Crear un buffer de futuros de lectura (pre-fetch)
        prefetch_size = read_workers + 1
        pending_reads: deque = deque()
        path_iter = iter(video_paths)

        # Llenar el buffer inicial de pre-fetch
        for _ in range(min(prefetch_size, len(video_paths))):
            try:
                path = next(path_iter)
                future = reader_pool.submit(_extract_frames, path)
                pending_reads.append((path, future))
            except StopIteration:
                break

        # Procesar: tomar resultado de lectura → GPU → encolar siguiente lectura
        while pending_reads:
            path, read_future = pending_reads.popleft()
            fp = VideoFingerprint(path=path)

            try:
                frames, meta = read_future.result()
                fp.fps = meta["fps"]
                fp.frame_count = meta["frame_count"]
                fp.width = meta["width"]
                fp.height = meta["height"]
                fp.duration_seconds = meta["duration_seconds"]

                # Mientras la GPU procesa, el ThreadPool ya está leyendo
                # el siguiente video en background
                fp.hashes = _frames_to_embeddings(frames)
                del frames

            except Exception as exc:
                fp.error = str(exc)

            results.append(fp)
            if on_complete:
                on_complete(fp)

            # Encolar siguiente lectura
            try:
                next_path = next(path_iter)
                future = reader_pool.submit(_extract_frames, next_path)
                pending_reads.append((next_path, future))
            except StopIteration:
                pass

        # Limpiar GPU al final
        _clear_gpu_cache()

    return results

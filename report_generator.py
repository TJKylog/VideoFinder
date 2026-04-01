"""
Genera thumbnails de los videos analizados y puebla la base de datos
SQLite con los resultados del analisis de duplicados.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

import db_manager
from frame_extractor import VideoFingerprint
from hash_comparator import MatchResult


def _silence_opencv():
    """Silencia los logs de OpenCV de forma thread-safe."""
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        pass


# Numero de thumbnails a extraer por video
_NUM_FRAMES = 4
_THUMB_WIDTH = 160
_THUMB_QUALITY = 50
_THUMBS_DIR_NAME = "report_thumbs"


def _extract_thumbnails(
    video_path: Path,
    thumbs_dir: Path,
    num_frames: int = _NUM_FRAMES,
) -> List[Tuple[str, float]]:
    """
    Extrae *num_frames* thumbnails equidistantes de un video y los guarda
    como archivos JPEG en thumbs_dir.

    Returns
    -------
    list[(absolute_path_str, timestamp_seconds)]
    """
    thumbnails: List[Tuple[str, float]] = []
    try:
        _silence_opencv()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return thumbnails

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0.0

            if duration <= 0:
                return thumbnails

            interval = duration / (num_frames + 1)
            safe_name = video_path.stem.replace(" ", "_")[:40]
            vid_hash = abs(hash(str(video_path))) % 0xFFFF

            for i in range(1, num_frames + 1):
                ts = interval * i
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
                ret, frame = cap.read()
                if not ret:
                    continue

                h, w = frame.shape[:2]
                new_w = _THUMB_WIDTH
                new_h = int(h * (new_w / w))
                thumb = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                fname = f"{safe_name}_{vid_hash:04x}_{i:02d}.jpg"
                fpath = thumbs_dir / fname
                cv2.imwrite(
                    str(fpath), thumb,
                    [cv2.IMWRITE_JPEG_QUALITY, _THUMB_QUALITY],
                )
                thumbnails.append((str(fpath), ts))
        finally:
            cap.release()
    except Exception:
        pass

    return thumbnails


def _prefetch_all_thumbnails(
    matches: List[MatchResult],
    thumbs_dir: Path,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extrae thumbnails de todos los videos en los matches en paralelo.
    """
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    unique_paths = set()
    for m in matches:
        unique_paths.add(str(m.video_a.path))
        unique_paths.add(str(m.video_b.path))

    cache: Dict[str, List[Tuple[str, float]]] = {}
    path_list = list(unique_paths)

    workers = min(8, len(path_list))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_extract_thumbnails, Path(p), thumbs_dir): p
            for p in path_list
        }
        for future in futures:
            p = futures[future]
            try:
                cache[p] = future.result(timeout=60)
            except Exception:
                cache[p] = []

    return cache


def generate_report(
    matches: List[MatchResult],
    all_fingerprints: List[VideoFingerprint],
    output_dir: str,
) -> str:
    """
    Genera thumbnails y puebla la BD SQLite con los resultados.

    Returns
    -------
    str
        Ruta de la base de datos generada.
    """
    print("   [THUMBS] Extrayendo thumbnails...")

    # -- Inicializar BD --
    db_path = db_manager.init_db(output_dir)
    db_manager.populate_videos(db_path, all_fingerprints)
    db_manager.populate_matches(db_path, matches)

    # -- Extraer thumbnails y guardarlos en BD --
    if matches:
        thumbs_dir = Path(output_dir) / _THUMBS_DIR_NAME
        thumbs_cache = _prefetch_all_thumbnails(matches, thumbs_dir)
        db_manager.populate_thumbnails(db_path, thumbs_cache, str(thumbs_dir))
        print(f"   [THUMBS] {sum(len(v) for v in thumbs_cache.values())} thumbnails extraidos")
    else:
        print("   [THUMBS] Sin duplicados, no se generaron thumbnails")

    print(f"   [BD] Base de datos guardada: {db_path}")
    return db_path

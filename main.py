#!/usr/bin/env python3
"""
FindDuplicatedVideos — Encuentra videos duplicados o que contengan
fragmentos de otros videos en una carpeta y sus subcarpetas.

Uso:
    python main.py /ruta/a/carpeta/de/videos [opciones]

Ejemplo:
    python main.py ~/Videos --threshold 0.35 --hamming 15 --interval 2
"""

import argparse
import gc
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

from tqdm import tqdm

import config
from fingerprint_cache import FingerprintCache
from frame_extractor import VideoFingerprint, extract_fingerprint
from hash_comparator import MatchResult, compare_all, compare_pair
from report_generator import generate_html_report
from video_scanner import scan_videos

# GPU extractor se importa condicionalmente
_gpu_extract_fn = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "🎬 FindDuplicatedVideos — Detecta videos duplicados o que "
            "contienen fragmentos de otros videos usando hash perceptual."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "folder",
        help="Carpeta raíz donde buscar videos (incluye subcarpetas).",
    )
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=config.FRAME_INTERVAL_SECONDS,
        help=(
            f"Intervalo en segundos entre frames extraídos "
            f"(default: {config.FRAME_INTERVAL_SECONDS})."
        ),
    )
    parser.add_argument(
        "-H", "--hamming",
        type=int,
        default=config.HAMMING_THRESHOLD,
        help=(
            f"Distancia Hamming máxima para considerar dos frames iguales "
            f"(default: {config.HAMMING_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=config.MATCH_RATIO_THRESHOLD,
        help=(
            f"Porcentaje mínimo de coincidencia (0.0-1.0) "
            f"(default: {config.MATCH_RATIO_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "-m", "--max-hashes",
        type=int,
        default=config.MAX_HASHES_PER_VIDEO,
        help=(
            f"Máximo de hashes por video. Videos largos auto-ajustan intervalo "
            f"(default: {config.MAX_HASHES_PER_VIDEO}, 0=sin límite)."
        ),
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Número de procesos paralelos (default: auto).",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Carpeta de salida para el reporte (default: misma que folder).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="No generar el reporte HTML.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Desactivar la caché de fingerprints.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Borrar la caché existente antes de empezar.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Usar GPU (PyTorch + CNN) para extraer embeddings. "
             "Más rápido y preciso. Requiere: pip install torch torchvision",
    )
    parser.add_argument(
        "--gpu-batch",
        type=int,
        default=config.GPU_BATCH_SIZE,
        help=f"Tamaño de batch para GPU (default: {config.GPU_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--cosine",
        type=float,
        default=config.COSINE_THRESHOLD,
        help=f"Umbral de similitud coseno para modo GPU (default: {config.COSINE_THRESHOLD}).",
    )
    return parser.parse_args()


def apply_cli_overrides(args: argparse.Namespace) -> None:
    """Aplica los argumentos CLI a la configuración global."""
    config.FRAME_INTERVAL_SECONDS = args.interval
    config.HAMMING_THRESHOLD = args.hamming
    config.MATCH_RATIO_THRESHOLD = args.threshold
    config.MAX_HASHES_PER_VIDEO = args.max_hashes
    config.GPU_BATCH_SIZE = args.gpu_batch
    config.COSINE_THRESHOLD = args.cosine
    if args.workers is not None:
        config.MAX_WORKERS = args.workers
    if args.no_cache:
        config.CACHE_ENABLED = False
    if args.gpu:
        config.USE_GPU = True


def main() -> None:
    args = parse_args()
    apply_cli_overrides(args)

    # Suprimir logs de OpenCV/ffmpeg a nivel global
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET

    folder = Path(args.folder).resolve()
    output_dir = args.output or str(folder)

    # ── 1. Escanear videos ───────────────────────────────────────────────
    print(f"\n📁 Escaneando: {folder}")
    videos = scan_videos(str(folder))

    if not videos:
        print("⚠️  No se encontraron archivos de video.")
        sys.exit(0)

    print(f"   Encontrados: {len(videos)} videos\n")

    # ── Modo GPU: inicializar y mostrar dispositivo ──────────────────────
    extract_fn = extract_fingerprint  # default: CPU + pHash
    if config.USE_GPU:
        try:
            from gpu_extractor import extract_fingerprint_gpu, get_gpu_device_name
            device_name = get_gpu_device_name()
            print(f"🚀 Modo GPU activado: {device_name}")
            print(f"   Batch size: {config.GPU_BATCH_SIZE} | Cosine threshold: {config.COSINE_THRESHOLD}\n")
            extract_fn = extract_fingerprint_gpu
        except ImportError as e:
            print(f"⚠️  No se pudo activar GPU: {e}")
            print("   Continuando en modo CPU...\n")
            config.USE_GPU = False

    # ── 2. Inicializar caché ─────────────────────────────────────────────
    cache = FingerprintCache(str(folder))
    if args.clear_cache:
        cache.clear()
        print("🗑  Caché eliminada.\n")

    # Separar videos ya cacheados de los que faltan
    cached_fps: List[VideoFingerprint] = []
    videos_to_process: List[Path] = []
    for v in videos:
        fp = cache.get(v)
        if fp is not None:
            cached_fps.append(fp)
        else:
            videos_to_process.append(v)

    if cached_fps:
        print(f"💾 Caché: {len(cached_fps)} videos ya procesados, "
              f"{len(videos_to_process)} pendientes.\n")

    # ── 3. Extraer fingerprints (paralelo) ───────────────────────────────
    fingerprints: List[VideoFingerprint] = list(cached_fps)
    t_start = time.time()

    if videos_to_process:
        print(f"🔍 Extrayendo huellas digitales de {len(videos_to_process)} videos...")

        if config.USE_GPU:
            # GPU: procesar secuencialmente (la GPU ya paraleliza internamente)
            save_every = max(10, len(videos_to_process) // 20)

            interrupted = False
            original_sigint = signal.getsignal(signal.SIGINT)

            def _sigint_handler(sig, frame):
                nonlocal interrupted
                interrupted = True
                print("\n\n⚠️  Interrumpido. Guardando caché...")

            signal.signal(signal.SIGINT, _sigint_handler)

            processed_count = 0
            try:
                with tqdm(total=len(videos_to_process), unit="video", ncols=90) as pbar:
                    for v in videos_to_process:
                        if interrupted:
                            break

                        fp = extract_fn(v)
                        fingerprints.append(fp)
                        cache.put(fp.path, fp)
                        processed_count += 1

                        if processed_count % save_every == 0:
                            cache.save()

                        status = (f"✅ {len(fp.hashes)} emb"
                                  if not fp.error else f"❌ {fp.error[:30]}")
                        pbar.set_postfix_str(f"{fp.path.name[:25]} → {status}")
                        pbar.update(1)
            finally:
                signal.signal(signal.SIGINT, original_sigint)
                cache.save()
                print(f"\n   💾 Caché guardada ({cache.size} videos totales)")

            if interrupted:
                print("\n⚠️  Proceso interrumpido. Se guardó el progreso en caché.")
                print("   Ejecuta el mismo comando de nuevo para continuar.\n")
                sys.exit(130)
        else:
            # CPU: procesar en paralelo con ProcessPoolExecutor
            workers = config.MAX_WORKERS or min(8, len(videos_to_process))
            save_every = max(10, len(videos_to_process) // 20)

            # Manejo de Ctrl+C para guardar caché antes de salir
            interrupted = False
            original_sigint = signal.getsignal(signal.SIGINT)

            def _sigint_handler(sig, frame):
                nonlocal interrupted
                interrupted = True
                print("\n\n⚠️  Interrumpido. Guardando caché...")

            signal.signal(signal.SIGINT, _sigint_handler)

            processed_count = 0
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(extract_fn, v): v for v in videos_to_process
                    }
                    with tqdm(total=len(videos_to_process), unit="video", ncols=90) as pbar:
                        for future in as_completed(futures):
                            if interrupted:
                                for f in futures:
                                    f.cancel()
                                break

                            fp = future.result()
                            fingerprints.append(fp)
                            cache.put(fp.path, fp)
                            processed_count += 1

                            if processed_count % save_every == 0:
                                cache.save()

                            status = (f"✅ {len(fp.hashes)} hashes"
                                      if not fp.error else f"❌ {fp.error[:30]}")
                            pbar.set_postfix_str(f"{fp.path.name[:25]} → {status}")
                            pbar.update(1)
            finally:
                signal.signal(signal.SIGINT, original_sigint)
                cache.save()
                print(f"\n   💾 Caché guardada ({cache.size} videos totales)")

            if interrupted:
                print("\n⚠️  Proceso interrumpido. Se guardó el progreso en caché.")
                print("   Ejecuta el mismo comando de nuevo para continuar.\n")
                sys.exit(130)

    t_extract = time.time() - t_start
    valid = [fp for fp in fingerprints if fp.hashes and not fp.error]
    errored = [fp for fp in fingerprints if fp.error]

    print(f"\n   ✅ Procesados: {len(valid)} videos correctamente")
    if errored:
        print(f"   ⚠️  Con errores: {len(errored)} videos")
        for fp in errored[:10]:
            print(f"      └─ {fp.path.name}: {fp.error}")
        if len(errored) > 10:
            print(f"      ... y {len(errored) - 10} más")

    if len(valid) < 2:
        print("\n⚠️  Se necesitan al menos 2 videos válidos para comparar.")
        sys.exit(0)

    # ── 4. Comparar todos los pares ──────────────────────────────────────
    n_pairs = len(valid) * (len(valid) - 1) // 2
    print(f"\n🔄 Comparando {n_pairs:,} pares de videos...")
    t_compare = time.time()

    matches = compare_all(fingerprints)

    t_compare = time.time() - t_compare
    t_total = time.time() - t_start

    # ── 5. Mostrar resultados ────────────────────────────────────────────
    print(f"\n{'='*60}")
    if matches:
        print(f"🎯 ¡Se encontraron {len(matches)} pares duplicados/contenidos!\n")
        for i, m in enumerate(matches, 1):
            print(f"  [{i}] {m.summary()}\n")
    else:
        print("✅ No se encontraron videos duplicados.")

    print(f"{'='*60}")
    print(f"⏱  Extracción: {t_extract:.1f}s | Comparación: {t_compare:.1f}s | Total: {t_total:.1f}s")

    # ── 6. Generar reporte HTML ──────────────────────────────────────────
    if not args.no_report:
        report_path = generate_html_report(
            matches=matches,
            all_fingerprints=fingerprints,
            output_dir=output_dir,
            scan_time_seconds=t_total,
        )
        print(f"📄 Reporte generado: {report_path}")

    print()


if __name__ == "__main__":
    main()

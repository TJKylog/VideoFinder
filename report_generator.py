"""
Genera un reporte HTML visual con los resultados de la comparación de videos.

Incluye:
  - Thumbnails de frames extraídos de cada video.
  - Comparación visual lado a lado de cada par duplicado.
  - Gauge animado de similitud.
  - Panel de metodología y parámetros utilizados.
"""

import gc
import html
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

import config
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

# Número de thumbnails a extraer por video en el reporte
_REPORT_NUM_FRAMES = 4
_THUMB_WIDTH = 160
_THUMB_QUALITY = 50
_THUMBS_DIR_NAME = "report_thumbs"


# ─── Utilidades ──────────────────────────────────────────────────────────────

def _format_duration(seconds: float) -> str:
    """Formatea segundos a HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _extract_thumbnails(
    video_path: Path,
    thumbs_dir: Path,
    num_frames: int = _REPORT_NUM_FRAMES,
) -> List[Tuple[str, float]]:
    """
    Extrae *num_frames* thumbnails equidistantes de un video y los guarda
    como archivos JPEG en thumbs_dir.

    Returns
    -------
    list[(relative_path_str, timestamp_seconds)]
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
            # Nombre base seguro para el archivo
            safe_name = video_path.stem.replace(' ', '_')[:40]
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
                # Ruta relativa al HTML (que estará junto a thumbs_dir)
                rel_path = f"{_THUMBS_DIR_NAME}/{fname}"
                thumbnails.append((rel_path, ts))
        finally:
            cap.release()
    except Exception:
        pass

    return thumbnails


def _get_thumbnails_for_pair(
    m: MatchResult,
    cache: Dict[str, List[Tuple[str, float]]],
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extrae thumbnails solo para los 2 videos de un match.
    Usa *cache* para no re-extraer un video que ya apareció antes.
    """
    for p in (str(m.video_a.path), str(m.video_b.path)):
        if p not in cache:
            cache[p] = _extract_thumbnails(Path(p))
    return cache


def _prefetch_all_thumbnails(
    matches: List[MatchResult],
    thumbs_dir: Path,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extrae thumbnails de todos los videos en los matches en paralelo
    y los guarda como archivos JPEG en thumbs_dir.
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


def _similarity_color(ratio: float) -> str:
    if ratio >= 0.80:
        return "#e74c3c"
    elif ratio >= 0.60:
        return "#e67e22"
    elif ratio >= 0.40:
        return "#f39c12"
    else:
        return "#2ecc71"


def _similarity_label(ratio: float) -> str:
    if ratio >= 0.90:
        return "Casi idéntico"
    elif ratio >= 0.75:
        return "Muy similar"
    elif ratio >= 0.60:
        return "Similar"
    elif ratio >= 0.40:
        return "Parcialmente similar"
    else:
        return "Baja similitud"


def _build_filmstrip(
    thumbs: List[Tuple[str, float]],
    label: str,
) -> str:
    """Genera HTML de una tira de thumbnails con timestamps."""
    if not thumbs:
        return (
            '<div class="filmstrip-empty">'
            f'No se pudieron extraer frames de {html.escape(label)}'
            '</div>'
        )

    parts: List[str] = []
    for rel_path, ts in thumbs:
        parts.append(
            '<div class="frame-cell">'
            f'<img src="{html.escape(rel_path)}" alt="frame" loading="lazy">'
            f'<span class="frame-ts">{_format_duration(ts)}</span>'
            '</div>'
        )

    return f'<div class="filmstrip">{"" .join(parts)}</div>'


def _build_match_card(
    idx: int,
    m: MatchResult,
    thumbs_map: Dict[str, List[Tuple[str, float]]],
) -> str:
    """Genera el HTML de una tarjeta visual para un par duplicado."""
    pct = min(m.match_ratio * 100, 100)
    color = _similarity_color(m.match_ratio)
    label = _similarity_label(m.match_ratio)

    thumbs_a = thumbs_map.get(str(m.video_a.path), [])
    thumbs_b = thumbs_map.get(str(m.video_b.path), [])

    metric_name = "Similitud Coseno" if config.USE_GPU else "Hamming Promedio"
    if config.USE_GPU:
        metric_value = (
            f"{(100.0 - m.avg_hamming) / 100.0:.3f}"
            if m.avg_hamming < 999 else "N/A"
        )
    else:
        metric_value = f"{m.avg_hamming:.1f}"

    threshold_str = (
        f"coseno ≥ {config.COSINE_THRESHOLD}"
        if config.USE_GPU
        else f"hamming ≤ {config.HAMMING_THRESHOLD}"
    )

    return f'''
    <div class="match-card">
        <div class="match-header">
            <span class="match-num">#{idx}</span>
            <span class="match-label" style="color:{color}">{label}</span>
        </div>

        <!-- Gauge de similitud -->
        <div class="gauge-container">
            <div class="gauge-ring">
                <svg viewBox="0 0 120 120">
                    <circle cx="60" cy="60" r="52" fill="none"
                            stroke="#2a2a4a" stroke-width="10"/>
                    <circle cx="60" cy="60" r="52" fill="none"
                            stroke="{color}" stroke-width="10"
                            stroke-dasharray="{pct * 3.267} 326.7"
                            stroke-dashoffset="0" stroke-linecap="round"
                            transform="rotate(-90 60 60)"
                            class="gauge-arc"/>
                </svg>
                <div class="gauge-text">
                    <span class="gauge-pct" style="color:{color}">{pct:.1f}%</span>
                    <span class="gauge-sub">coincidencia</span>
                </div>
            </div>
            <div class="gauge-details">
                <div class="gauge-detail-item">
                    <span class="detail-label">Frames coincidentes</span>
                    <span class="detail-value">{m.matched_frames} / {m.total_frames_a}</span>
                </div>
                <div class="gauge-detail-item">
                    <span class="detail-label">{metric_name}</span>
                    <span class="detail-value">{metric_value}</span>
                </div>
                <div class="gauge-detail-item">
                    <span class="detail-label">Umbral usado</span>
                    <span class="detail-value">{threshold_str}</span>
                </div>
            </div>
        </div>

        <!-- Video A -->
        <div class="video-section">
            <div class="video-info-bar video-a-bar">
                <span class="video-badge">A</span>
                <span class="video-name"
                      title="{html.escape(str(m.video_a.path))}"
                >{html.escape(m.video_a.path.name)}</span>
                <span class="video-meta">
                    {_format_duration(m.video_a.duration_seconds)}
                    &nbsp;·&nbsp; {m.video_a.filesize_mb:.1f} MB
                    &nbsp;·&nbsp; {m.video_a.width}×{m.video_a.height}
                    &nbsp;·&nbsp; {len(m.video_a.hashes)} hashes
                </span>
            </div>
            {_build_filmstrip(thumbs_a, m.video_a.path.name)}
        </div>

        <!-- Video B -->
        <div class="video-section">
            <div class="video-info-bar video-b-bar">
                <span class="video-badge">B</span>
                <span class="video-name"
                      title="{html.escape(str(m.video_b.path))}"
                >{html.escape(m.video_b.path.name)}</span>
                <span class="video-meta">
                    {_format_duration(m.video_b.duration_seconds)}
                    &nbsp;·&nbsp; {m.video_b.filesize_mb:.1f} MB
                    &nbsp;·&nbsp; {m.video_b.width}×{m.video_b.height}
                    &nbsp;·&nbsp; {len(m.video_b.hashes)} hashes
                </span>
            </div>
            {_build_filmstrip(thumbs_b, m.video_b.path.name)}
        </div>

        <div class="match-paths">
            <div><strong>A:</strong> <code>{html.escape(str(m.video_a.path))}</code></div>
            <div><strong>B:</strong> <code>{html.escape(str(m.video_b.path))}</code></div>
        </div>
    </div>
    '''


# ─── Generador principal ────────────────────────────────────────────────────

def generate_html_report(
    matches: List[MatchResult],
    all_fingerprints: List[VideoFingerprint],
    output_dir: str,
    scan_time_seconds: float = 0.0,
    max_visual_matches: int = 100,
) -> Path:
    """
    Genera un archivo HTML visual con los duplicados encontrados.
    Thumbnails se guardan como archivos JPEG en una subcarpeta.

    Parameters
    ----------
    max_visual_matches : int
        Máximo de comparaciones visuales con thumbnails en el reporte.
        Las demás se muestran solo en la tabla resumen.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_videos = len(all_fingerprints)
    errored = sum(1 for fp in all_fingerprints if fp.error)
    total_matches = len(matches)
    valid_videos = total_videos - errored
    total_hashes = sum(len(fp.hashes) for fp in all_fingerprints if not fp.error)
    n_pairs = valid_videos * (valid_videos - 1) // 2 if valid_videos > 1 else 0

    print("   📸 Extrayendo thumbnails para el reporte...")
    visual_matches = matches[:max_visual_matches]
    if len(matches) > max_visual_matches:
        print(f"   ℹ️  Mostrando thumbnails para los {max_visual_matches} mejores matches "
              f"(de {len(matches)} totales). Usa --max-report para ajustar.")

    # ── Tabla resumen rápida ─────────────────────────────────────────────
    summary_parts: List[str] = []
    for i, m in enumerate(matches, 1):
        pct = min(m.match_ratio * 100, 100)
        color = _similarity_color(m.match_ratio)
        summary_parts.append(f'''
        <tr>
            <td>{i}</td>
            <td title="{html.escape(str(m.video_a.path))}">{html.escape(m.video_a.path.name)}</td>
            <td>{_format_duration(m.video_a.duration_seconds)}</td>
            <td>{m.video_a.filesize_mb:.1f} MB</td>
            <td title="{html.escape(str(m.video_b.path))}">{html.escape(m.video_b.path.name)}</td>
            <td>{_format_duration(m.video_b.duration_seconds)}</td>
            <td>{m.video_b.filesize_mb:.1f} MB</td>
            <td>
                <div class="mini-bar">
                    <div class="mini-bar-fill"
                         style="width:{pct:.1f}%;background:{color}"></div>
                </div>
                <span style="color:{color};font-weight:700">{pct:.1f}%</span>
            </td>
            <td>{m.matched_frames}/{m.total_frames_a}</td>
            <td>{m.avg_hamming:.1f}</td>
        </tr>''')
    summary_rows = "".join(summary_parts)

    # ── Tabla de errores ─────────────────────────────────────────────────
    error_section = ""
    if errored:
        err_parts: List[str] = []
        for fp in all_fingerprints:
            if fp.error:
                err_parts.append(f'''
                <tr>
                    <td title="{html.escape(str(fp.path))}"
                    >{html.escape(fp.path.name)}</td>
                    <td>{html.escape(fp.error)}</td>
                </tr>''')
        error_rows = "".join(err_parts)
        error_section = f'''
        <h2 id="errors"><span class="section-icon">⚠️</span>
            Videos con Errores</h2>
        <table>
        <thead><tr><th>Archivo</th><th>Error</th></tr></thead>
        <tbody>{error_rows}</tbody>
        </table>'''

    # ── Tabla de todos los videos ────────────────────────────────────────
    all_parts: List[str] = []
    for fp in sorted(all_fingerprints, key=lambda f: f.path.name):
        status = "⚠️ Error" if fp.error else "✅ OK"
        all_parts.append(f'''
        <tr>
            <td title="{html.escape(str(fp.path))}">{html.escape(fp.path.name)}</td>
            <td>{_format_duration(fp.duration_seconds)}</td>
            <td>{fp.filesize_mb:.1f} MB</td>
            <td>{fp.width}×{fp.height}</td>
            <td>{fp.fps:.1f}</td>
            <td>{len(fp.hashes)}</td>
            <td>{status}</td>
        </tr>''')
    all_rows = "".join(all_parts)

    # ── Modo y parámetros ────────────────────────────────────────────────
    if config.USE_GPU:
        mode_badge = '<span class="mode-badge mode-gpu">🚀 GPU (CNN)</span>'
        mode_name = "GPU — MobileNetV3-Small"
        mode_desc = (
            "Embeddings de 576 dimensiones generados con una CNN preentrenada,"
            " comparados por similitud coseno."
        )
        metric_info = f'''
            <div class="param-card">
                <div class="param-label">Modelo</div>
                <div class="param-value">MobileNetV3-Small</div>
            </div>
            <div class="param-card">
                <div class="param-label">Embedding</div>
                <div class="param-value">576 dims</div>
            </div>
            <div class="param-card">
                <div class="param-label">Métrica</div>
                <div class="param-value">Similitud Coseno</div>
            </div>
            <div class="param-card accent">
                <div class="param-label">Umbral Coseno</div>
                <div class="param-value">≥ {config.COSINE_THRESHOLD}</div>
            </div>
            <div class="param-card">
                <div class="param-label">Batch Size GPU</div>
                <div class="param-value">{config.GPU_BATCH_SIZE}</div>
            </div>'''
        how_step2 = (
            '<strong>2.</strong> Cada frame pasa por '
            '<strong style="color:var(--accent)">MobileNetV3-Small</strong> '
            'para generar un embedding de '
            '<strong style="color:var(--accent)">576 dimensiones</strong>.'
        )
        how_step3 = (
            '<strong>3.</strong> Para cada par de videos, se calcula la '
            '<strong style="color:var(--accent)">similitud coseno</strong> '
            'entre todos los embeddings.'
        )
        how_step4 = (
            '<strong>4.</strong> Un frame coincide si su similitud con algún '
            'frame del otro video es '
            f'<strong style="color:var(--accent)">≥ {config.COSINE_THRESHOLD}</strong>.'
        )
    else:
        mode_badge = '<span class="mode-badge mode-cpu">🖥️ CPU (pHash)</span>'
        mode_name = "CPU — Hash Perceptual (pHash)"
        mode_desc = (
            "Hash perceptual de 64 bits basado en DCT. Los frames se comparan"
            " mediante distancia Hamming."
        )
        metric_info = f'''
            <div class="param-card">
                <div class="param-label">Hash</div>
                <div class="param-value">pHash {config.HASH_SIZE}×{config.HASH_SIZE} = {config.HASH_SIZE**2} bits</div>
            </div>
            <div class="param-card">
                <div class="param-label">Resize</div>
                <div class="param-value">{config.FRAME_RESIZE[0]}×{config.FRAME_RESIZE[1]} px</div>
            </div>
            <div class="param-card">
                <div class="param-label">Métrica</div>
                <div class="param-value">Distancia Hamming</div>
            </div>
            <div class="param-card accent">
                <div class="param-label">Umbral Hamming</div>
                <div class="param-value">≤ {config.HAMMING_THRESHOLD}</div>
            </div>'''
        how_step2 = (
            '<strong>2.</strong> Cada frame se redimensiona a '
            f'<strong style="color:var(--accent)">{config.FRAME_RESIZE[0]}×{config.FRAME_RESIZE[1]}</strong> px'
            ' y se calcula su '
            f'<strong style="color:var(--accent)">hash perceptual (pHash)</strong>'
            f' de <strong style="color:var(--accent)">{config.HASH_SIZE**2} bits</strong> vía DCT.'
        )
        how_step3 = (
            '<strong>3.</strong> Para cada par de videos, se calcula la '
            '<strong style="color:var(--accent)">distancia Hamming</strong> '
            'entre todos los hashes (XOR vectorizado).'
        )
        how_step4 = (
            '<strong>4.</strong> Un frame coincide si su distancia Hamming '
            'al más cercano es '
            f'<strong style="color:var(--accent)">≤ {config.HAMMING_THRESHOLD}</strong>.'
        )

    # ── Pipeline visual ──────────────────────────────────────────────────
    pipeline_html = f'''
    <div class="pipeline">
        <div class="pipe-step">
            <div class="pipe-icon">📁</div>
            <div class="pipe-title">Escaneo</div>
            <div class="pipe-detail">{total_videos} videos</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="pipe-icon">🎞️</div>
            <div class="pipe-title">Extracción</div>
            <div class="pipe-detail">Cada {config.FRAME_INTERVAL_SECONDS}s · máx {config.MAX_HASHES_PER_VIDEO}</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="pipe-icon">{"🧠" if config.USE_GPU else "🔐"}</div>
            <div class="pipe-title">{"Embedding CNN" if config.USE_GPU else "pHash (DCT)"}</div>
            <div class="pipe-detail">{total_hashes:,} hashes totales</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="pipe-icon">🔍</div>
            <div class="pipe-title">Comparación</div>
            <div class="pipe-detail">{n_pairs:,} pares · {"coseno" if config.USE_GPU else "hamming"}</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step highlight">
            <div class="pipe-icon">🎯</div>
            <div class="pipe-title">Resultados</div>
            <div class="pipe-detail">{total_matches} duplicados</div>
        </div>
    </div>'''

    cosine_metric = "coseno" if config.USE_GPU else "hamming"

    # ── Escribir HTML por bloques al disco (streaming) ───────────────────
    output_path = Path(output_dir) / config.REPORT_FILENAME
    with open(output_path, "w", encoding="utf-8") as f:
        # ── Head + CSS + apertura body ───────────────────────────────────
        f.write(f'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🎬 Reporte de Videos Duplicados</title>
<style>
  :root {{
    --bg-primary: #0f0f1a;
    --bg-secondary: #161625;
    --bg-card: #1c1c30;
    --bg-hover: #22223a;
    --accent: #e94560;
    --accent2: #0f3460;
    --text: #e8e8f0;
    --text-dim: #8888aa;
    --text-muted: #555577;
    --border: #2a2a42;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont,
                 Helvetica, Arial, sans-serif;
    background: var(--bg-primary); color: var(--text);
    padding: 2rem; line-height: 1.5;
  }}

  /* ── Header ── */
  .header {{ margin-bottom: 2rem; }}
  .header h1 {{ font-size: 2rem; color: var(--accent); margin-bottom: .3rem; }}
  .header .subtitle {{ color: var(--text-dim); font-size: .9rem; }}

  /* ── Nav ── */
  .nav {{ display: flex; gap: .5rem; flex-wrap: wrap; margin: 1.5rem 0; }}
  .nav a {{
    background: var(--bg-card); color: var(--text-dim);
    text-decoration: none; padding: .45rem 1rem;
    border-radius: 6px; font-size: .82rem; transition: all .2s;
  }}
  .nav a:hover {{ background: var(--accent); color: #fff; }}

  /* ── Section headings ── */
  h2 {{
    font-size: 1.15rem; color: var(--text); margin: 2.2rem 0 1rem;
    padding-bottom: .5rem; border-bottom: 2px solid var(--accent);
    display: flex; align-items: center; gap: .5rem;
  }}
  .section-icon {{ font-size: 1.2rem; }}

  /* ── Stats ── */
  .stats {{ display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.2rem 0 2rem; }}
  .stat-card {{
    background: var(--bg-card); border-left: 4px solid var(--accent);
    padding: .9rem 1.3rem; border-radius: 8px; min-width: 160px; flex: 1;
  }}
  .stat-card .label {{
    font-size: .78rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: .03em;
  }}
  .stat-card .value {{
    font-size: 1.8rem; font-weight: 700; color: var(--accent);
    margin-top: .1rem;
  }}

  /* ── Pipeline ── */
  .pipeline {{
    display: flex; align-items: center; gap: .5rem; flex-wrap: wrap;
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 10px; padding: 1.2rem 1.5rem; margin: 1.5rem 0;
  }}
  .pipe-step {{
    text-align: center; padding: .6rem .8rem; border-radius: 8px;
    background: var(--bg-card); min-width: 110px; flex: 1;
  }}
  .pipe-step.highlight {{ border: 2px solid var(--accent); }}
  .pipe-icon {{ font-size: 1.5rem; margin-bottom: .2rem; }}
  .pipe-title {{ font-size: .8rem; font-weight: 700; color: var(--text); }}
  .pipe-detail {{ font-size: .7rem; color: var(--text-dim); margin-top: .15rem; }}
  .pipe-arrow {{ color: var(--accent); font-size: 1.3rem; font-weight: 700; }}

  /* ── Mode badge ── */
  .mode-badge {{
    display: inline-block; padding: .3rem .9rem; border-radius: 20px;
    font-size: .82rem; font-weight: 600; margin-bottom: .5rem;
  }}
  .mode-gpu {{ background: #1a3a1a; color: #4ecdc4; border: 1px solid #4ecdc4; }}
  .mode-cpu {{ background: #1a2a3a; color: #74b9ff; border: 1px solid #74b9ff; }}

  /* ── Param cards ── */
  .params-grid {{
    display: flex; gap: .7rem; flex-wrap: wrap; margin: .8rem 0 1.5rem;
  }}
  .param-card {{
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: .6rem 1rem; min-width: 140px; flex: 1;
  }}
  .param-card.accent {{ border-color: var(--accent); }}
  .param-label {{
    font-size: .7rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: .03em;
  }}
  .param-value {{
    font-size: .95rem; font-weight: 600; color: var(--text);
    margin-top: .1rem;
  }}
  .param-card.accent .param-value {{ color: var(--accent); }}
  .method-desc {{
    color: var(--text-dim); font-size: .85rem; margin: .5rem 0 1rem;
    max-width: 700px; line-height: 1.5;
  }}

  /* ── Match card ── */
  .match-card {{
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1.8rem;
    transition: border-color .2s;
  }}
  .match-card:hover {{ border-color: var(--accent); }}
  .match-header {{
    display: flex; align-items: center; gap: .8rem; margin-bottom: 1rem;
  }}
  .match-num {{
    background: var(--accent); color: #fff; font-weight: 700;
    padding: .2rem .7rem; border-radius: 6px; font-size: .9rem;
  }}
  .match-label {{ font-weight: 600; font-size: 1rem; }}

  /* ── Gauge ── */
  .gauge-container {{
    display: flex; align-items: center; gap: 2rem; margin: 1rem 0 1.5rem;
    flex-wrap: wrap;
  }}
  .gauge-ring {{
    position: relative; width: 120px; height: 120px; flex-shrink: 0;
  }}
  .gauge-ring svg {{ width: 100%; height: 100%; }}
  .gauge-arc {{ transition: stroke-dasharray .8s ease-out; }}
  .gauge-text {{
    position: absolute; top: 50%; left: 50%;
    transform: translate(-50%, -50%); text-align: center;
  }}
  .gauge-pct {{ font-size: 1.4rem; font-weight: 800; display: block; }}
  .gauge-sub {{ font-size: .65rem; color: var(--text-dim); display: block; }}
  .gauge-details {{ display: flex; flex-direction: column; gap: .5rem; }}
  .gauge-detail-item {{ display: flex; flex-direction: column; }}
  .detail-label {{
    font-size: .7rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: .03em;
  }}
  .detail-value {{ font-size: .9rem; font-weight: 600; color: var(--text); }}

  /* ── Video section ── */
  .video-section {{ margin: .8rem 0; }}
  .video-info-bar {{
    display: flex; align-items: center; gap: .6rem; flex-wrap: wrap;
    padding: .5rem .8rem; border-radius: 8px 8px 0 0; font-size: .82rem;
  }}
  .video-a-bar {{ background: #1a2540; border-left: 3px solid #3498db; }}
  .video-b-bar {{ background: #2a1a30; border-left: 3px solid #9b59b6; }}
  .video-badge {{
    font-weight: 800; font-size: .75rem; padding: .15rem .5rem;
    border-radius: 4px; color: #fff;
  }}
  .video-a-bar .video-badge {{ background: #3498db; }}
  .video-b-bar .video-badge {{ background: #9b59b6; }}
  .video-name {{ font-weight: 600; color: var(--text); }}
  .video-meta {{ color: var(--text-dim); font-size: .75rem; }}

  /* ── Filmstrip ── */
  .filmstrip {{
    display: flex; gap: 6px; overflow-x: auto; padding: .6rem;
    background: var(--bg-card); border-radius: 0 0 8px 8px;
    scrollbar-width: thin; scrollbar-color: var(--accent) var(--bg-card);
  }}
  .filmstrip::-webkit-scrollbar {{ height: 6px; }}
  .filmstrip::-webkit-scrollbar-track {{
    background: var(--bg-card); border-radius: 3px;
  }}
  .filmstrip::-webkit-scrollbar-thumb {{
    background: var(--accent); border-radius: 3px;
  }}
  .frame-cell {{ flex-shrink: 0; text-align: center; position: relative; }}
  .frame-cell img {{
    border-radius: 4px; display: block;
    border: 2px solid transparent; transition: border-color .2s;
    height: 100px; width: auto;
  }}
  .frame-cell img:hover {{ border-color: var(--accent); }}
  .frame-ts {{
    display: block; font-size: .65rem; color: var(--text-dim);
    margin-top: .2rem; font-family: monospace;
  }}
  .filmstrip-empty {{
    padding: 1.5rem; text-align: center; color: var(--text-muted);
    background: var(--bg-card); border-radius: 0 0 8px 8px;
    font-size: .85rem;
  }}

  /* ── Match paths ── */
  .match-paths {{
    margin-top: .8rem; padding: .6rem .8rem; background: var(--bg-card);
    border-radius: 6px; font-size: .72rem; color: var(--text-dim);
  }}
  .match-paths code {{
    color: var(--text-muted); font-size: .7rem; word-break: break-all;
  }}

  /* ── Tables ── */
  table {{
    width: 100%; border-collapse: collapse; margin-bottom: 1.5rem;
    background: var(--bg-card); border-radius: 8px; overflow: hidden;
  }}
  th {{
    background: var(--accent2); color: var(--accent); padding: .65rem .5rem;
    text-align: left; font-size: .78rem; text-transform: uppercase;
    letter-spacing: .03em; font-weight: 600;
  }}
  td {{
    padding: .5rem .5rem; border-bottom: 1px solid var(--bg-primary);
    font-size: .82rem; max-width: 250px; overflow: hidden;
    text-overflow: ellipsis; white-space: nowrap;
  }}
  tr:hover td {{ background: var(--bg-hover); }}

  .mini-bar {{
    background: #2a2a42; border-radius: 3px; overflow: hidden;
    width: 80px; height: 8px; display: inline-block;
    vertical-align: middle; margin-right: .4rem;
  }}
  .mini-bar-fill {{
    height: 100%; border-radius: 3px; transition: width .3s;
  }}

  /* ── No matches ── */
  .no-matches {{
    text-align: center; padding: 3rem; color: var(--text-dim);
    background: var(--bg-secondary); border-radius: 12px;
    border: 1px dashed var(--border);
  }}
  .no-matches-icon {{ font-size: 3rem; margin-bottom: .5rem; }}

  /* ── How-it-works box ── */
  .how-box {{
    background: var(--bg-card); border-radius: 8px; padding: 1rem;
    font-size: .8rem; color: var(--text-dim); margin-bottom: 1rem;
    line-height: 1.6;
  }}
  .how-box strong.hl {{ color: var(--accent); }}

  /* ── Footer ── */
  footer {{
    margin-top: 3rem; padding-top: 1rem;
    border-top: 1px solid var(--border);
    text-align: center; color: var(--text-muted); font-size: .75rem;
  }}

  @media (max-width: 768px) {{
    body {{ padding: 1rem; }}
    .stats {{ flex-direction: column; }}
    .pipeline {{ flex-direction: column; }}
    .pipe-arrow {{ transform: rotate(90deg); }}
    .gauge-container {{ flex-direction: column; align-items: flex-start; }}
  }}
</style>
</head>
<body>

<div class="header">
    <h1>🎬 Reporte de Videos Duplicados</h1>
    <div class="subtitle">
        Generado: {now} &nbsp;·&nbsp;
        Tiempo de análisis: {scan_time_seconds:.1f}s
    </div>
</div>

<nav class="nav">
    <a href="#methodology">🧪 Metodología</a>
    <a href="#results">🎯 Resultados</a>
    <a href="#summary">📊 Tabla Resumen</a>
    <a href="#all-videos">📋 Todos los Videos</a>
    {"<a href='#errors'>⚠️ Errores</a>" if errored else ""}
</nav>

<!-- ═══ Estadísticas ═══ -->
<div class="stats">
    <div class="stat-card">
        <div class="label">Videos analizados</div>
        <div class="value">{total_videos}</div>
    </div>
    <div class="stat-card">
        <div class="label">Duplicados encontrados</div>
        <div class="value">{total_matches}</div>
    </div>
    <div class="stat-card">
        <div class="label">Hashes generados</div>
        <div class="value">{total_hashes:,}</div>
    </div>
    <div class="stat-card">
        <div class="label">Pares comparados</div>
        <div class="value">{n_pairs:,}</div>
    </div>
    <div class="stat-card">
        <div class="label">Con errores</div>
        <div class="value">{errored}</div>
    </div>
    <div class="stat-card">
        <div class="label">Tiempo total</div>
        <div class="value">{scan_time_seconds:.1f}s</div>
    </div>
</div>

<!-- ═══ Metodología ═══ -->
<h2 id="methodology">
    <span class="section-icon">🧪</span> Metodología y Parámetros
</h2>

{pipeline_html}

<div style="margin-top:1.5rem">
    {mode_badge}
    <h3 style="font-size:.95rem;margin:.4rem 0 .2rem;color:var(--text)">
        {mode_name}
    </h3>
    <p class="method-desc">{mode_desc}</p>
</div>

<div class="params-grid">
    <div class="param-card">
        <div class="param-label">Intervalo de Frames</div>
        <div class="param-value">Cada {config.FRAME_INTERVAL_SECONDS}s</div>
    </div>
    <div class="param-card">
        <div class="param-label">Máx. Hashes/Video</div>
        <div class="param-value">{config.MAX_HASHES_PER_VIDEO if config.MAX_HASHES_PER_VIDEO > 0 else "Sin límite"}</div>
    </div>
    {metric_info}
    <div class="param-card accent">
        <div class="param-label">Ratio Mínimo</div>
        <div class="param-value">≥ {config.MATCH_RATIO_THRESHOLD:.0%}</div>
    </div>
</div>

<div class="how-box">
    <strong style="color:var(--text)">¿Cómo funciona?</strong><br>
    <strong>1.</strong> Se extraen frames del video cada
    <strong class="hl">{config.FRAME_INTERVAL_SECONDS}s</strong>
    (máximo <strong class="hl">{config.MAX_HASHES_PER_VIDEO}</strong>
    por video).<br>
    {how_step2}<br>
    {how_step3}<br>
    {how_step4}<br>
    <strong>5.</strong> Si el
    <strong class="hl">{config.MATCH_RATIO_THRESHOLD:.0%}</strong>
    o más de los frames del video corto coinciden →
    <strong class="hl">DUPLICADO</strong>.
</div>

<!-- ═══ Resultados visuales ═══ -->
<h2 id="results">
    <span class="section-icon">🎯</span> Comparación Visual de Duplicados
</h2>
<p style="color:var(--text-dim);font-size:.82rem;margin-bottom:1.2rem">
    Cada tarjeta muestra thumbnails extraídos de ambos videos para
    comparación visual. Los timestamps indican el momento exacto del
    frame en el video.
</p>
''')
        # Liberar variables de secciones ya escritas
        del pipeline_html, mode_badge, mode_name, mode_desc
        del metric_info, how_step2, how_step3, how_step4

        # ── Match cards: thumbnails guardados como archivos ─────────────
        if visual_matches:
            thumbs_dir = Path(output_dir) / _THUMBS_DIR_NAME
            thumbs_cache = _prefetch_all_thumbnails(visual_matches, thumbs_dir)
            for i, m in enumerate(visual_matches, 1):
                card_html = _build_match_card(i, m, thumbs_cache)
                f.write(card_html)
                del card_html
            del thumbs_cache
            if len(matches) > max_visual_matches:
                f.write(f'''
        <div class="no-matches" style="border-color:var(--accent)">
            <div class="no-matches-icon">📊</div>
            <p>Se muestran {max_visual_matches} comparaciones visuales de
               {len(matches)} totales.<br>
               El resto se puede consultar en la
               <a href="#summary" style="color:var(--accent)">tabla resumen</a>.
               Usa <code>--max-report N</code> para ver más.</p>
        </div>''')
        else:
            f.write('''
        <div class="no-matches">
            <div class="no-matches-icon">✅</div>
            <p>No se encontraron videos duplicados.
               ¡Todos los videos son únicos!</p>
        </div>''')

        # ── Resto del documento ──────────────────────────────────────────
        f.write(f'''
<!-- ═══ Tabla resumen ═══ -->
<h2 id="summary">
    <span class="section-icon">📊</span> Tabla Resumen
</h2>
<table>
<thead>
<tr>
    <th>#</th>
    <th>Video A (corto)</th><th>Dur. A</th><th>Tam. A</th>
    <th>Video B (largo)</th><th>Dur. B</th><th>Tam. B</th>
    <th>Similitud</th><th>Frames</th>
    <th>{"Coseno" if config.USE_GPU else "Hamming"} Avg</th>
</tr>
</thead>
<tbody>{summary_rows if summary_rows else '<tr><td colspan="10" style="text-align:center;color:var(--text-muted)">— Sin resultados —</td></tr>'}
</tbody>
</table>

{error_section}

<!-- ═══ Todos los videos ═══ -->
<h2 id="all-videos">
    <span class="section-icon">📋</span> Todos los Videos Procesados
</h2>
<table>
<thead>
<tr>
    <th>Archivo</th><th>Duración</th><th>Tamaño</th>
    <th>Resolución</th><th>FPS</th><th>Hashes</th><th>Estado</th>
</tr>
</thead>
<tbody>{all_rows}</tbody>
</table>

<footer>
    FindDuplicatedVideos &nbsp;·&nbsp;
    {"GPU: MobileNetV3 + Coseno" if config.USE_GPU else "CPU: pHash + Hamming"}
    &nbsp;·&nbsp;
    {total_videos} videos · {total_hashes:,} hashes · {n_pairs:,} pares
    &nbsp;·&nbsp; {now}
</footer>

</body>
</html>''')

    return output_path

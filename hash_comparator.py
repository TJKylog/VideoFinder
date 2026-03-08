"""
Compara las huellas digitales (fingerprints) de los videos para encontrar
duplicados o videos que contengan fragmentos de otros.

Optimizaciones:
  - Hashes como numpy arrays → distancias Hamming vectorizadas.
  - Matriz de distancias precalculada por par de videos.
  - Descarte temprano de pares que no pueden alcanzar el umbral.
  - Barra de progreso en la comparación.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import config
from frame_extractor import VideoFingerprint


@dataclass
class MatchResult:
    """Resultado de la comparación entre dos videos."""
    video_a: VideoFingerprint          # video más corto (o con menos hashes)
    video_b: VideoFingerprint          # video más largo
    matched_frames: int                # cantidad de frames de A que coinciden en B
    total_frames_a: int                # total de frames (hashes) de A
    best_offset: int                   # desplazamiento en B donde mejor coincide A
    match_ratio: float                 # matched_frames / total_frames_a
    avg_hamming: float                 # distancia Hamming promedio de los matches

    @property
    def is_duplicate(self) -> bool:
        return self.match_ratio >= config.MATCH_RATIO_THRESHOLD

    def summary(self) -> str:
        return (
            f"{'✅ DUPLICADO' if self.is_duplicate else '❌ Sin coincidencia'}  "
            f"similitud={self.match_ratio:.1%}  "
            f"frames={self.matched_frames}/{self.total_frames_a}  "
            f"hamming_avg={self.avg_hamming:.1f}\n"
            f"  ├─ Corto : {self.video_a.path.name} "
            f"({self.video_a.duration_seconds:.1f}s)\n"
            f"  └─ Largo : {self.video_b.path.name} "
            f"({self.video_b.duration_seconds:.1f}s)"
        )


def _build_hash_matrix(hashes: List[np.ndarray]) -> np.ndarray:
    """Apila una lista de hash arrays en una matriz 2D (n_hashes × hash_bits)."""
    if not hashes:
        return np.empty((0, 0), dtype=np.uint8)
    return np.stack(hashes)  # shape: (n, bits)


def _hamming_matrix(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de distancias Hamming entre todas las filas de mat_a
    y mat_b usando operaciones vectorizadas.

    Returns: shape (len_a, len_b) con distancias Hamming.
    """
    # XOR + sum  — completamente vectorizado
    # mat_a shape: (m, bits), mat_b shape: (n, bits)
    # Expandir dimensiones para broadcasting: (m, 1, bits) XOR (1, n, bits) → (m, n, bits)
    xor = mat_a[:, np.newaxis, :] ^ mat_b[np.newaxis, :, :]
    return xor.sum(axis=2)  # (m, n)


def _greedy_match_fast(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    threshold: int,
) -> Tuple[int, float]:
    """
    Para cada hash de A, encuentra el hash más cercano en B (vectorizado).
    Mucho más rápido que el loop puro de Python.

    Returns
    -------
    (matches, avg_hamming)
    """
    if mat_a.shape[0] == 0 or mat_b.shape[0] == 0:
        return 0, 999.0

    # Si la matriz sería muy grande (>50M), procesar en bloques
    max_elements = 50_000_000
    len_a, len_b = mat_a.shape[0], mat_b.shape[0]

    if len_a * len_b > max_elements:
        # Procesar A en bloques
        block_size = max(1, max_elements // len_b)
        min_dists = []
        for start in range(0, len_a, block_size):
            end = min(start + block_size, len_a)
            dist_block = _hamming_matrix(mat_a[start:end], mat_b)
            min_dists.append(dist_block.min(axis=1))
        min_distances = np.concatenate(min_dists)
    else:
        dist_matrix = _hamming_matrix(mat_a, mat_b)
        min_distances = dist_matrix.min(axis=1)  # mejor match para cada hash de A

    mask = min_distances <= threshold
    matches = int(mask.sum())
    avg_hamming = float(min_distances[mask].mean()) if matches > 0 else 999.0
    return matches, avg_hamming


# ─── Modo GPU: Similitud Coseno ──────────────────────────────────────────────

def _cosine_similarity_matrix(mat_a: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de similitud coseno entre embeddings.
    mat_a: (m, dim), mat_b: (n, dim) → resultado: (m, n)
    Los embeddings ya están normalizados L2, así que es un simple dot product.
    """
    return mat_a @ mat_b.T  # (m, n)


def _greedy_match_cosine(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    threshold: float,
) -> Tuple[int, float]:
    """
    Para cada embedding de A, encuentra el más similar en B por coseno.

    Returns
    -------
    (matches, avg_similarity)  — avg_similarity está en 0..1
    """
    if mat_a.shape[0] == 0 or mat_b.shape[0] == 0:
        return 0, 0.0

    max_elements = 50_000_000
    len_a, len_b = mat_a.shape[0], mat_b.shape[0]

    if len_a * len_b > max_elements:
        block_size = max(1, max_elements // len_b)
        max_sims = []
        for start in range(0, len_a, block_size):
            end = min(start + block_size, len_a)
            sim_block = _cosine_similarity_matrix(mat_a[start:end], mat_b)
            max_sims.append(sim_block.max(axis=1))
        max_similarities = np.concatenate(max_sims)
    else:
        sim_matrix = _cosine_similarity_matrix(mat_a, mat_b)
        max_similarities = sim_matrix.max(axis=1)  # mejor match por cada frame de A

    mask = max_similarities >= threshold
    matches = int(mask.sum())
    avg_sim = float(max_similarities[mask].mean()) if matches > 0 else 0.0
    return matches, avg_sim


def compare_pair(
    fp_a: VideoFingerprint,
    fp_b: VideoFingerprint,
) -> MatchResult:
    """
    Compara dos fingerprints usando distancia Hamming vectorizada.
    """
    # Asegurar que A es el más corto (menos hashes)
    if len(fp_a.hashes) > len(fp_b.hashes):
        fp_a, fp_b = fp_b, fp_a

    total_a = len(fp_a.hashes)
    if total_a == 0:
        return MatchResult(
            video_a=fp_a, video_b=fp_b,
            matched_frames=0, total_frames_a=0,
            best_offset=0, match_ratio=0.0, avg_hamming=999.0,
        )

    mat_a = _build_hash_matrix(fp_a.hashes)
    mat_b = _build_hash_matrix(fp_b.hashes)

    if config.USE_GPU:
        # Modo GPU: similitud coseno sobre embeddings float32
        matches, avg_sim = _greedy_match_cosine(
            mat_a.astype(np.float32), mat_b.astype(np.float32),
            config.COSINE_THRESHOLD,
        )
        # Convertir similitud coseno a un "score" compatible con el reporte
        # avg_hamming aquí representa (1 - avg_similarity) para consistencia
        avg_metric = (1.0 - avg_sim) * 100 if avg_sim > 0 else 999.0
    else:
        matches, avg_metric = _greedy_match_fast(mat_a, mat_b, config.HAMMING_THRESHOLD)

    return MatchResult(
        video_a=fp_a,
        video_b=fp_b,
        matched_frames=matches,
        total_frames_a=total_a,
        best_offset=-1,
        match_ratio=matches / total_a,
        avg_hamming=avg_metric,
    )


def compare_all(
    fingerprints: List[VideoFingerprint],
    show_progress: bool = True,
) -> List[MatchResult]:
    """
    Compara todos los pares posibles y devuelve solo los resultados
    que superan el umbral de similitud (duplicados).
    """
    results: List[MatchResult] = []
    valid = [fp for fp in fingerprints if fp.hashes and not fp.error]

    pairs = list(combinations(valid, 2))

    iterator = tqdm(pairs, unit="par", ncols=90, desc="   Comparando") if show_progress else pairs
    for fp_a, fp_b in iterator:
        result = compare_pair(fp_a, fp_b)
        if result.is_duplicate:
            results.append(result)
            if show_progress:
                iterator.set_postfix_str(
                    f"✅ {len(results)} dup | {fp_a.path.name[:15]}↔{fp_b.path.name[:15]}"
                )

    # Ordenar por similitud descendente
    results.sort(key=lambda r: r.match_ratio, reverse=True)
    return results

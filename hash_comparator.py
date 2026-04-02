"""
Compara las huellas digitales (fingerprints) de los videos para encontrar
duplicados o videos que contengan fragmentos de otros.

Optimizaciones:
  - Hashes como numpy arrays → distancias Hamming vectorizadas.
  - Matriz de distancias precalculada por par de videos.
  - Descarte temprano de pares que no pueden alcanzar el umbral.
  - Barra de progreso en la comparación.
"""

from bisect import bisect_left
from dataclasses import dataclass
from itertools import combinations
from typing import List, Tuple

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            f"{'[DUPLICADO]' if self.is_duplicate else '[SIN COINCIDENCIA]'}  "
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


def _lis_length(seq: np.ndarray) -> int:
    """
    Longitud de la subsecuencia creciente más larga (LIS) en O(n log n).
    Se usa para verificar que los frames coincidentes mantienen
    un orden temporal coherente entre los dos videos.
    """
    if len(seq) == 0:
        return 0
    tails: list = []
    for val in seq:
        pos = bisect_left(tails, val)
        if pos == len(tails):
            tails.append(val)
        else:
            tails[pos] = val
    return len(tails)


def _greedy_match_fast(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    threshold: int,
) -> Tuple[int, float, np.ndarray]:
    """
    Para cada hash de A, encuentra el hash más cercano en B (vectorizado).
    Mucho más rápido que el loop puro de Python.

    Returns
    -------
    (matches, avg_hamming, matched_b_indices)
        matched_b_indices: índices en B de los frames de A que coincidieron,
                           en el orden original de A.
    """
    if mat_a.shape[0] == 0 or mat_b.shape[0] == 0:
        return 0, 999.0, np.array([], dtype=np.intp)

    # Si la matriz sería muy grande (>50M), procesar en bloques
    max_elements = 50_000_000
    len_a, len_b = mat_a.shape[0], mat_b.shape[0]

    if len_a * len_b > max_elements:
        # Procesar A en bloques
        block_size = max(1, max_elements // len_b)
        min_dists = []
        best_indices = []
        for start in range(0, len_a, block_size):
            end = min(start + block_size, len_a)
            dist_block = _hamming_matrix(mat_a[start:end], mat_b)
            min_dists.append(dist_block.min(axis=1))
            best_indices.append(dist_block.argmin(axis=1))
        min_distances = np.concatenate(min_dists)
        best_b_indices = np.concatenate(best_indices)
    else:
        dist_matrix = _hamming_matrix(mat_a, mat_b)
        min_distances = dist_matrix.min(axis=1)  # mejor match para cada hash de A
        best_b_indices = dist_matrix.argmin(axis=1)

    mask = min_distances <= threshold
    matches = int(mask.sum())
    avg_hamming = float(min_distances[mask].mean()) if matches > 0 else 999.0
    return matches, avg_hamming, best_b_indices[mask]


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
) -> Tuple[int, float, np.ndarray]:
    """
    Para cada embedding de A, encuentra el más similar en B por coseno.

    Returns
    -------
    (matches, avg_similarity, matched_b_indices)
        matched_b_indices: índices en B de los frames de A que coincidieron,
                           en el orden original de A.
    """
    if mat_a.shape[0] == 0 or mat_b.shape[0] == 0:
        return 0, 0.0, np.array([], dtype=np.intp)

    max_elements = 50_000_000
    len_a, len_b = mat_a.shape[0], mat_b.shape[0]

    if len_a * len_b > max_elements:
        block_size = max(1, max_elements // len_b)
        max_sims = []
        best_indices = []
        for start in range(0, len_a, block_size):
            end = min(start + block_size, len_a)
            sim_block = _cosine_similarity_matrix(mat_a[start:end], mat_b)
            max_sims.append(sim_block.max(axis=1))
            best_indices.append(sim_block.argmax(axis=1))
        max_similarities = np.concatenate(max_sims)
        best_b_indices = np.concatenate(best_indices)
    else:
        sim_matrix = _cosine_similarity_matrix(mat_a, mat_b)
        max_similarities = sim_matrix.max(axis=1)  # mejor match por cada frame de A
        best_b_indices = sim_matrix.argmax(axis=1)

    mask = max_similarities >= threshold
    matches = int(mask.sum())
    avg_sim = float(max_similarities[mask].mean()) if matches > 0 else 0.0
    return matches, avg_sim, best_b_indices[mask]


def _compare_with_matrices(
    fp_a: VideoFingerprint,
    fp_b: VideoFingerprint,
    mat_a: np.ndarray,
    mat_b: np.ndarray,
) -> MatchResult:
    """
    Compara dos fingerprints usando matrices de hashes pre-construidas.
    Evita reconstruir la matriz cada vez.
    """
    # Asegurar que A es el más corto (menos hashes)
    if mat_a.shape[0] > mat_b.shape[0]:
        fp_a, fp_b = fp_b, fp_a
        mat_a, mat_b = mat_b, mat_a

    total_a = mat_a.shape[0]
    if total_a == 0:
        return MatchResult(
            video_a=fp_a, video_b=fp_b,
            matched_frames=0, total_frames_a=0,
            best_offset=0, match_ratio=0.0, avg_hamming=999.0,
        )

    if config.USE_GPU:
        raw_matches, avg_sim, matched_b = _greedy_match_cosine(
            mat_a.astype(np.float32), mat_b.astype(np.float32),
            config.COSINE_THRESHOLD,
        )
        avg_metric = (1.0 - avg_sim) * 100 if avg_sim > 0 else 999.0
    else:
        raw_matches, avg_metric, matched_b = _greedy_match_fast(
            mat_a, mat_b, config.HAMMING_THRESHOLD,
        )

    # Verificación temporal: solo contar frames que mantienen
    # una secuencia creciente coherente en B (LIS).
    # Esto descarta coincidencias espurias entre videos no relacionados
    # donde los frames coincidentes aparecen en orden aleatorio.
    matches = _lis_length(matched_b) if len(matched_b) > 0 else 0

    return MatchResult(
        video_a=fp_a,
        video_b=fp_b,
        matched_frames=matches,
        total_frames_a=total_a,
        best_offset=-1,
        match_ratio=matches / total_a,
        avg_hamming=avg_metric,
    )


def compare_pair(
    fp_a: VideoFingerprint,
    fp_b: VideoFingerprint,
) -> MatchResult:
    """
    Compara dos fingerprints (construye matrices internamente).
    Útil para comparaciones individuales.
    """
    mat_a = _build_hash_matrix(fp_a.hashes)
    mat_b = _build_hash_matrix(fp_b.hashes)
    return _compare_with_matrices(fp_a, fp_b, mat_a, mat_b)


def compare_all(
    fingerprints: List[VideoFingerprint],
    show_progress: bool = True,
) -> List[MatchResult]:
    """
    Compara todos los pares posibles y devuelve solo los resultados
    que superan el umbral de similitud (duplicados).

    Optimizaciones:
      - Pre-construye matrices de hashes UNA vez.
      - Pre-filtra por duración para descartar pares imposibles.
      - Agrupa videos por "firma rápida" (hash del primer embedding) para
        reducir combinaciones cuando hay muchos videos.
      - Comparaciones en paralelo con ThreadPoolExecutor (numpy libera el GIL).
    """
    results: List[MatchResult] = []
    valid = [fp for fp in fingerprints if fp.hashes and not fp.error]

    if len(valid) < 2:
        return results

    # Pre-construir matriz de hashes/embeddings para cada video (una sola vez)
    matrices = {}
    for fp in valid:
        matrices[id(fp)] = _build_hash_matrix(fp.hashes)

    # Pre-filtrar pares por duración para descartar combinaciones imposibles
    threshold_ratio = config.MATCH_RATIO_THRESHOLD * 0.5
    pairs = []
    for fp_a, fp_b in combinations(valid, 2):
        dur_a = fp_a.duration_seconds or 0
        dur_b = fp_b.duration_seconds or 0
        if dur_a > 0 and dur_b > 0:
            ratio = min(dur_a, dur_b) / max(dur_a, dur_b)
            if ratio < threshold_ratio:
                continue
        pairs.append((fp_a, fp_b))

    n_total = len(valid) * (len(valid) - 1) // 2
    n_pairs = len(pairs)

    if show_progress:
        skipped = n_total - n_pairs
        if skipped > 0:
            print(f"   Pares descartados por duración: {skipped:,} "
                  f"(quedan {n_pairs:,})")

    if not pairs:
        return results

    # Función de trabajo para un batch de pares
    def _compare_batch(batch):
        batch_results = []
        for fp_a, fp_b in batch:
            r = _compare_with_matrices(
                fp_a, fp_b, matrices[id(fp_a)], matrices[id(fp_b)]
            )
            if r.is_duplicate:
                batch_results.append(r)
        return batch_results

    # Dividir en bloques para paralelizar (numpy libera el GIL en ops vectorizadas)
    import os
    n_workers = min(os.cpu_count() or 4, 8)
    batch_size = max(500, n_pairs // (n_workers * 4))
    batches = [pairs[i:i + batch_size] for i in range(0, n_pairs, batch_size)]

    if show_progress:
        pbar = tqdm(total=n_pairs, unit="par", ncols=90, desc="   Comparando")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_compare_batch, b): len(b) for b in batches}
        for future in as_completed(futures):
            batch_results = future.result()
            results.extend(batch_results)
            if show_progress:
                pbar.update(futures[future])
                if batch_results:
                    pbar.set_postfix_str(f"OK: {len(results)} dup")

    if show_progress:
        pbar.close()

    # Liberar matrices
    del matrices, pairs

    results.sort(key=lambda r: r.match_ratio, reverse=True)
    return results

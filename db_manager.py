"""
Gestiona una base de datos SQLite con los resultados del análisis
de videos duplicados: videos analizados, coincidencias y thumbnails.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple


DB_FILENAME = "duplicates.db"


def _get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(output_dir: str) -> str:
    """Crea la BD y las tablas si no existen. Devuelve la ruta de la BD."""
    db_path = str(Path(output_dir) / DB_FILENAME)
    conn = _get_connection(db_path)
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                path        TEXT UNIQUE NOT NULL,
                filename    TEXT NOT NULL,
                duration_s  REAL DEFAULT 0,
                fps         REAL DEFAULT 0,
                width       INTEGER DEFAULT 0,
                height      INTEGER DEFAULT 0,
                filesize_mb REAL DEFAULT 0,
                num_hashes  INTEGER DEFAULT 0,
                error       TEXT,
                deleted     INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS matches (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                video_a_id      INTEGER NOT NULL REFERENCES videos(id),
                video_b_id      INTEGER NOT NULL REFERENCES videos(id),
                match_ratio     REAL NOT NULL,
                matched_frames  INTEGER NOT NULL,
                total_frames_a  INTEGER NOT NULL,
                avg_hamming     REAL NOT NULL,
                UNIQUE(video_a_id, video_b_id)
            );

            CREATE TABLE IF NOT EXISTS thumbnails (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id    INTEGER NOT NULL REFERENCES videos(id),
                thumb_path  TEXT NOT NULL,
                timestamp_s REAL DEFAULT 0
            );
        """)
        conn.commit()
    finally:
        conn.close()
    return db_path


def populate_videos(db_path: str, fingerprints) -> None:
    """Inserta o actualiza todos los videos analizados."""
    conn = _get_connection(db_path)
    try:
        for fp in fingerprints:
            conn.execute("""
                INSERT INTO videos (path, filename, duration_s, fps, width, height,
                                    filesize_mb, num_hashes, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    duration_s=excluded.duration_s,
                    fps=excluded.fps,
                    width=excluded.width,
                    height=excluded.height,
                    filesize_mb=excluded.filesize_mb,
                    num_hashes=excluded.num_hashes,
                    error=excluded.error
            """, (
                str(fp.path), fp.path.name,
                fp.duration_seconds, fp.fps,
                fp.width, fp.height,
                fp.filesize_mb, len(fp.hashes),
                fp.error,
            ))
        conn.commit()
    finally:
        conn.close()


def populate_matches(db_path: str, matches) -> None:
    """Inserta las coincidencias encontradas."""
    conn = _get_connection(db_path)
    try:
        for m in matches:
            # Obtener IDs de los videos
            row_a = conn.execute(
                "SELECT id FROM videos WHERE path=?", (str(m.video_a.path),)
            ).fetchone()
            row_b = conn.execute(
                "SELECT id FROM videos WHERE path=?", (str(m.video_b.path),)
            ).fetchone()
            if row_a and row_b:
                conn.execute("""
                    INSERT INTO matches (video_a_id, video_b_id, match_ratio,
                                         matched_frames, total_frames_a, avg_hamming)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(video_a_id, video_b_id) DO UPDATE SET
                        match_ratio=excluded.match_ratio,
                        matched_frames=excluded.matched_frames,
                        total_frames_a=excluded.total_frames_a,
                        avg_hamming=excluded.avg_hamming
                """, (
                    row_a["id"], row_b["id"],
                    m.match_ratio, m.matched_frames,
                    m.total_frames_a, m.avg_hamming,
                ))
        conn.commit()
    finally:
        conn.close()


def populate_thumbnails(db_path: str, thumbs_map: dict, thumbs_dir: str) -> None:
    """Inserta las rutas de los thumbnails ya generados."""
    conn = _get_connection(db_path)
    try:
        for video_path_str, thumbs in thumbs_map.items():
            row = conn.execute(
                "SELECT id FROM videos WHERE path=?", (video_path_str,)
            ).fetchone()
            if not row:
                continue
            vid_id = row["id"]
            # Limpiar thumbs previos de este video
            conn.execute("DELETE FROM thumbnails WHERE video_id=?", (vid_id,))
            for rel_path, ts in thumbs:
                full_thumb = str(Path(thumbs_dir).parent / rel_path)
                conn.execute("""
                    INSERT INTO thumbnails (video_id, thumb_path, timestamp_s)
                    VALUES (?, ?, ?)
                """, (vid_id, full_thumb, ts))
        conn.commit()
    finally:
        conn.close()


# ─── Consultas para la GUI ──────────────────────────────────────────────────

def get_all_matches(db_path: str) -> List[dict]:
    """Devuelve todos los matches con info de ambos videos."""
    conn = _get_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT
                m.id            AS match_id,
                m.match_ratio,
                m.matched_frames,
                m.total_frames_a,
                m.avg_hamming,
                va.id AS va_id, va.path AS va_path, va.filename AS va_name,
                va.duration_s AS va_dur, va.filesize_mb AS va_size,
                va.width AS va_w, va.height AS va_h, va.deleted AS va_del,
                vb.id AS vb_id, vb.path AS vb_path, vb.filename AS vb_name,
                vb.duration_s AS vb_dur, vb.filesize_mb AS vb_size,
                vb.width AS vb_w, vb.height AS vb_h, vb.deleted AS vb_del
            FROM matches m
            JOIN videos va ON m.video_a_id = va.id
            JOIN videos vb ON m.video_b_id = vb.id
            ORDER BY m.match_ratio DESC
        """).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_thumbnails_for_video(db_path: str, video_id: int) -> List[dict]:
    """Devuelve los thumbnails de un video."""
    conn = _get_connection(db_path)
    try:
        rows = conn.execute("""
            SELECT thumb_path, timestamp_s
            FROM thumbnails WHERE video_id=?
            ORDER BY timestamp_s
        """, (video_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def mark_deleted(db_path: str, video_id: int) -> None:
    """Marca un video como eliminado en la BD."""
    conn = _get_connection(db_path)
    try:
        conn.execute("UPDATE videos SET deleted=1 WHERE id=?", (video_id,))
        conn.commit()
    finally:
        conn.close()


def get_db_path_from_dir(output_dir: str) -> str:
    return str(Path(output_dir) / DB_FILENAME)

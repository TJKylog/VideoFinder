"""
Ventana gráfica (tkinter) para consultar la BD de videos duplicados,
comparar thumbnails lado a lado y eliminar archivos.
"""

import os
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
from typing import List

from PIL import Image, ImageTk

import db_manager


# ─── Constantes ──────────────────────────────────────────────────────────────
_THUMB_DISPLAY_H = 120
_BG = "#0f0f1a"
_BG2 = "#1c1c30"
_FG = "#e8e8f0"
_ACCENT = "#e94560"
_DIM = "#8888aa"
_BORDER = "#2a2a42"


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class DuplicateViewerApp:
    """Ventana principal del visor de duplicados."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.matches: List[dict] = []
        self._photo_refs: list = []  # evita garbage collection de imágenes

        self.root = tk.Tk()
        self.root.title("FindDuplicatedVideos — Visor de Duplicados")
        self.root.configure(bg=_BG)
        self.root.geometry("1100x700")
        self.root.minsize(800, 500)

        self._build_ui()
        self._load_matches()

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Estilo ttk
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview",
                         background=_BG2, foreground=_FG,
                         fieldbackground=_BG2, borderwidth=0,
                         font=("Segoe UI", 11))
        style.configure("Treeview.Heading",
                         background=_BORDER, foreground=_ACCENT,
                         font=("Segoe UI", 10, "bold"))
        style.map("Treeview",
                   background=[("selected", _ACCENT)],
                   foreground=[("selected", "#fff")])

        # ── Panel superior: lista de matches ──
        top = tk.Frame(self.root, bg=_BG)
        top.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        lbl = tk.Label(top, text="Pares duplicados encontrados",
                        bg=_BG, fg=_ACCENT, font=("Segoe UI", 13, "bold"),
                        anchor="w")
        lbl.pack(fill=tk.X, pady=(0, 4))

        cols = ("num", "video_a", "dur_a", "size_a",
                "video_b", "dur_b", "size_b", "similitud", "estado")
        self.tree = ttk.Treeview(top, columns=cols, show="headings",
                                  selectmode="browse", height=10)
        self.tree.heading("num", text="#")
        self.tree.heading("video_a", text="Video A")
        self.tree.heading("dur_a", text="Dur. A")
        self.tree.heading("size_a", text="Tam. A")
        self.tree.heading("video_b", text="Video B")
        self.tree.heading("dur_b", text="Dur. B")
        self.tree.heading("size_b", text="Tam. B")
        self.tree.heading("similitud", text="Similitud")
        self.tree.heading("estado", text="Estado")

        self.tree.column("num", width=40, anchor="center")
        self.tree.column("video_a", width=180)
        self.tree.column("dur_a", width=65, anchor="center")
        self.tree.column("size_a", width=70, anchor="e")
        self.tree.column("video_b", width=180)
        self.tree.column("dur_b", width=65, anchor="center")
        self.tree.column("size_b", width=70, anchor="e")
        self.tree.column("similitud", width=80, anchor="center")
        self.tree.column("estado", width=100, anchor="center")

        scrollbar = ttk.Scrollbar(top, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # ── Panel inferior: comparación de thumbnails ──
        bottom = tk.Frame(self.root, bg=_BG)
        bottom.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Info del par seleccionado
        self.info_frame = tk.Frame(bottom, bg=_BG)
        self.info_frame.pack(fill=tk.X, pady=(0, 4))
        self.lbl_info = tk.Label(self.info_frame, text="Selecciona un par para comparar",
                                  bg=_BG, fg=_DIM, font=("Segoe UI", 10))
        self.lbl_info.pack(side=tk.LEFT)

        # Contenedor de thumbnails A y B
        thumb_container = tk.Frame(bottom, bg=_BG)
        thumb_container.pack(fill=tk.BOTH, expand=True)

        # Video A
        frame_a = tk.LabelFrame(thumb_container, text=" Video A ",
                                 bg=_BG2, fg="#3498db",
                                 font=("Segoe UI", 10, "bold"))
        frame_a.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.lbl_path_a = tk.Label(frame_a, text="", bg=_BG2, fg=_DIM,
                                    font=("Segoe UI", 8), anchor="w",
                                    wraplength=500)
        self.lbl_path_a.pack(fill=tk.X, padx=4, pady=2)
        self.canvas_a = tk.Canvas(frame_a, bg=_BG2, highlightthickness=0, height=140)
        self.canvas_a.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Video B
        frame_b = tk.LabelFrame(thumb_container, text=" Video B ",
                                 bg=_BG2, fg="#9b59b6",
                                 font=("Segoe UI", 10, "bold"))
        frame_b.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.lbl_path_b = tk.Label(frame_b, text="", bg=_BG2, fg=_DIM,
                                    font=("Segoe UI", 8), anchor="w",
                                    wraplength=500)
        self.lbl_path_b.pack(fill=tk.X, padx=4, pady=2)
        self.canvas_b = tk.Canvas(frame_b, bg=_BG2, highlightthickness=0, height=140)
        self.canvas_b.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # ── Barra de botones ──
        btn_bar = tk.Frame(self.root, bg=_BG)
        btn_bar.pack(fill=tk.X, padx=8, pady=(0, 8))

        self.btn_del_a = tk.Button(
            btn_bar, text="Eliminar Video A", bg="#c0392b", fg="#fff",
            font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=12, pady=4,
            command=lambda: self._delete_video("a"), state=tk.DISABLED)
        self.btn_del_a.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_del_b = tk.Button(
            btn_bar, text="Eliminar Video B", bg="#8e44ad", fg="#fff",
            font=("Segoe UI", 10, "bold"), relief=tk.FLAT, padx=12, pady=4,
            command=lambda: self._delete_video("b"), state=tk.DISABLED)
        self.btn_del_b.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_refresh = tk.Button(
            btn_bar, text="Actualizar", bg=_BORDER, fg=_FG,
            font=("Segoe UI", 10), relief=tk.FLAT, padx=12, pady=4,
            command=self._load_matches)
        self.btn_refresh.pack(side=tk.RIGHT)

    # ── Datos ────────────────────────────────────────────────────────────

    def _load_matches(self):
        self.matches = db_manager.get_all_matches(self.db_path)
        # Limpiar tree
        for item in self.tree.get_children():
            self.tree.delete(item)

        for i, m in enumerate(self.matches, 1):
            pct = f"{min(m['match_ratio'] * 100, 100):.1f}%"
            estado = ""
            if m["va_del"]:
                estado += "A eliminado"
            if m["vb_del"]:
                estado += (" / " if estado else "") + "B eliminado"
            if not estado:
                estado = "Pendiente"

            self.tree.insert("", tk.END, iid=str(i - 1), values=(
                i,
                m["va_name"], _format_duration(m["va_dur"]),
                f"{m['va_size']:.1f} MB",
                m["vb_name"], _format_duration(m["vb_dur"]),
                f"{m['vb_size']:.1f} MB",
                pct, estado,
            ))

    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        m = self.matches[idx]

        self.lbl_info.config(
            text=f"Similitud: {m['match_ratio']*100:.1f}%  |  "
                 f"Frames: {m['matched_frames']}/{m['total_frames_a']}  |  "
                 f"Hamming avg: {m['avg_hamming']:.1f}",
            fg=_ACCENT,
        )

        # Rutas
        self.lbl_path_a.config(text=m["va_path"])
        self.lbl_path_b.config(text=m["vb_path"])

        # Thumbnails
        self._show_thumbs(self.canvas_a, m["va_id"])
        self._show_thumbs(self.canvas_b, m["vb_id"])

        # Botones
        self.btn_del_a.config(
            state=tk.NORMAL if not m["va_del"] and Path(m["va_path"]).exists() else tk.DISABLED)
        self.btn_del_b.config(
            state=tk.NORMAL if not m["vb_del"] and Path(m["vb_path"]).exists() else tk.DISABLED)

        # Guardar selección actual
        self._selected_match = m

    def _show_thumbs(self, canvas: tk.Canvas, video_id: int):
        canvas.delete("all")
        self._photo_refs.clear()
        thumbs = db_manager.get_thumbnails_for_video(self.db_path, video_id)
        x_offset = 5
        for t in thumbs:
            tp = t["thumb_path"]
            if not Path(tp).exists():
                continue
            try:
                img = Image.open(tp)
                ratio = _THUMB_DISPLAY_H / img.height
                new_w = int(img.width * ratio)
                img = img.resize((new_w, _THUMB_DISPLAY_H), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._photo_refs.append(photo)
                canvas.create_image(x_offset, 5, anchor=tk.NW, image=photo)
                ts_txt = _format_duration(t["timestamp_s"])
                canvas.create_text(x_offset + new_w // 2, _THUMB_DISPLAY_H + 12,
                                    text=ts_txt, fill=_DIM,
                                    font=("Segoe UI", 8))
                x_offset += new_w + 8
            except Exception:
                continue
        canvas.config(scrollregion=(0, 0, x_offset, _THUMB_DISPLAY_H + 25))

    # ── Eliminar ─────────────────────────────────────────────────────────

    def _delete_video(self, which: str):
        m = getattr(self, "_selected_match", None)
        if not m:
            return
        vid_id = m["va_id"] if which == "a" else m["vb_id"]
        vid_path = m["va_path"] if which == "a" else m["vb_path"]
        vid_name = m["va_name"] if which == "a" else m["vb_name"]

        confirm = messagebox.askyesno(
            "Confirmar eliminacion",
            f"¿Eliminar permanentemente el archivo?\n\n{vid_name}\n{vid_path}",
            icon="warning",
        )
        if not confirm:
            return

        try:
            p = Path(vid_path)
            if p.exists():
                p.unlink()
            db_manager.mark_deleted(self.db_path, vid_id)
            messagebox.showinfo("Eliminado", f"{vid_name} eliminado correctamente.")
            self._load_matches()
        except OSError as e:
            messagebox.showerror("Error", f"No se pudo eliminar:\n{e}")

    # ── Run ──────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


def launch_viewer(db_path: str) -> None:
    """Punto de entrada para abrir la ventana del visor."""
    if not Path(db_path).exists():
        print(f"[ERROR] No se encontro la BD: {db_path}")
        return
    app = DuplicateViewerApp(db_path)
    app.run()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python viewer_gui.py <ruta_a_duplicates.db>")
        sys.exit(1)
    launch_viewer(sys.argv[1])

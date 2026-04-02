"""
Visor web local para consultar la BD de videos duplicados,
comparar thumbnails lado a lado y eliminar archivos.

Lanza un servidor HTTP local y abre el navegador automaticamente.
"""

import base64
import json
import os
import shutil
import socket
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import db_manager

_HOST = "127.0.0.1"
_HTML_FILE = Path(__file__).parent / "viewer.html"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _thumb_to_base64(path: str):
    try:
        p = Path(path)
        if not p.exists():
            return None
        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class ViewerHandler(BaseHTTPRequestHandler):
    db_path = ""

    def log_message(self, format, *args):
        pass

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, content):
        body = content.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._html(_HTML_FILE.read_text(encoding="utf-8"))
        elif path == "/api/matches":
            self._api_matches(params)
        elif path == "/api/thumbnails":
            self._api_thumbnails(params)
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/delete":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._api_delete(body)
        else:
            self.send_error(404)

    def _api_matches(self, params):
        page = int(params.get("page", ["0"])[0])
        limit = min(int(params.get("limit", ["50"])[0]), 200)
        offset = page * limit
        total = db_manager.get_match_count(self.db_path)
        matches = db_manager.get_all_matches(self.db_path, limit=limit, offset=offset)
        self._json({"total": total, "page": page, "limit": limit, "matches": matches})

    def _api_thumbnails(self, params):
        vid_id = int(params.get("video_id", ["0"])[0])
        if vid_id <= 0:
            self._json({"thumbs": []})
            return
        thumbs = db_manager.get_thumbnails_for_video(self.db_path, vid_id)
        result = []
        for t in thumbs:
            b64 = _thumb_to_base64(t["thumb_path"])
            if b64:
                result.append({"src": b64, "ts": _format_duration(t["timestamp_s"])})
        self._json({"thumbs": result})

    def _api_delete(self, body):
        vid_id = body.get("video_id", 0)
        vid_path = body.get("video_path", "")
        if not vid_id or not vid_path:
            self._json({"ok": False, "error": "Faltan parametros"}, 400)
            return
        try:
            p = Path(vid_path)
            if p.exists():
                trash_dir = p.parent / "_duplicados_papelera"
                trash_dir.mkdir(exist_ok=True)
                dest = trash_dir / p.name
                # Evitar sobreescribir si ya existe un archivo con el mismo nombre
                counter = 1
                while dest.exists():
                    dest = trash_dir / f"{p.stem}_{counter}{p.suffix}"
                    counter += 1
                shutil.move(str(p), str(dest))
            db_manager.mark_deleted(self.db_path, vid_id)
            self._json({"ok": True})
        except OSError as e:
            self._json({"ok": False, "error": str(e)}, 500)


def launch_viewer(db_path):
    if not Path(db_path).exists():
        print(f"[ERROR] No se encontro la BD: {db_path}")
        return
    if not _HTML_FILE.exists():
        print(f"[ERROR] No se encontro viewer.html: {_HTML_FILE}")
        return

    ViewerHandler.db_path = db_path
    port = _find_free_port()
    server = HTTPServer((_HOST, port), ViewerHandler)
    url = f"http://{_HOST}:{port}"

    print(f"[VISOR] Servidor iniciado en {url}")
    print(f"[VISOR] BD: {db_path}")
    print("[VISOR] Presiona Ctrl+C para detener\n")

    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[VISOR] Servidor detenido.")
    finally:
        server.server_close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python viewer_gui.py <ruta_a_duplicates.db>")
        sys.exit(1)
    launch_viewer(sys.argv[1])

"""Loopback-only, single-process hackathon server. No cloud services or build step."""
import argparse
import base64
import binascii
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from acoustic.audio import ExperimentError
from acoustic.experiment import Experiment

ROOT = Path(__file__).resolve().parent
MAX_BODY = 17 * 1024 * 1024


def make_handler(experiment):
    lock = threading.RLock()

    class Handler(BaseHTTPRequestHandler):
        def end_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob: data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            super().end_headers()

        def send(self, code, payload, mime="application/json; charset=utf-8"):
            raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8") if not isinstance(payload, bytes) else payload
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def local_request(self):
            expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in expected:
                self.send(403, {"error": "Разрешён только локальный адрес."})
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + host for host in expected}:
                self.send(403, {"error": "Запрос из другого сайта запрещён."})
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            path = urlsplit(self.path).path
            if path == "/api/state":
                with lock:
                    self.send(200, experiment.state())
            elif path == "/api/report":
                with lock:
                    self.send(200, experiment.state()["report"])
            elif path in {"/", "/index.html", "/styles.css", "/app.js", "/favicon.svg"}:
                name = "index.html" if path == "/" else path[1:]
                mime = {"html": "text/html; charset=utf-8", "css": "text/css; charset=utf-8", "js": "text/javascript; charset=utf-8", "svg": "image/svg+xml"}[name.rsplit(".", 1)[1]]
                self.send(200, (ROOT / "web" / name).read_bytes(), mime)
            else:
                self.send(404, {"error": "Не найдено."})

        def do_POST(self):
            if not self.local_request():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    self.send(413, {"error": "Размер запроса должен быть не больше 17 МБ."})
                    return
                if self.headers.get_content_type() != "application/json":
                    raise ExperimentError("Ожидается JSON.")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ExperimentError("Ожидается объект JSON.")
                path = urlsplit(self.path).path
                with lock:
                    if path in {"/api/analyze", "/api/data"}:
                        try:
                            raw = base64.b64decode(payload["audio"], validate=True)
                        except (KeyError, TypeError, binascii.Error) as exc:
                            raise ExperimentError("Некорректные аудиоданные.") from exc
                        filename = payload.get("filename", "")
                        if not isinstance(filename, str):
                            raise ExperimentError("Некорректное имя файла.")
                        result = (experiment.analyze(raw, filename, payload.get("mode", "cascade")) if path == "/api/analyze" else
                                  experiment.add(raw, filename, payload.get("split"), payload.get("label"), payload.get("session")))
                    elif path == "/api/train":
                        result = experiment.train(payload.get("stage", "both"))
                    elif path == "/api/evaluate":
                        result = experiment.evaluate(payload.get("mode", "cascade"))
                    else:
                        self.send(404, {"error": "Не найдено."})
                        return
                self.send(200, result)
            except (ExperimentError, ValueError, TypeError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception:
                traceback.print_exc()
                self.send(500, {"error": "Ошибка обработки. Подробности в локальной консоли сервера."})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Acoustic Detective — local experiment")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Experiment(args.data_dir, args.artifacts_dir)))
    print(f"Acoustic Detective: http://127.0.0.1:{args.port}  (PID {os.getpid()}, Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

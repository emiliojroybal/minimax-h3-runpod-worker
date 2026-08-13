from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from handler import capabilities, handler


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "H3Mock/0.1"

    def do_GET(self) -> None:
        if self.path in {"/health", "/capabilities"}:
            self._json(200, {"status": "ready", "capabilities": capabilities()})
        else:
            self._json(404, {"error": {"code": "NOT_FOUND", "message": "Route not found."}})

    def do_POST(self) -> None:
        if self.path != "/run":
            self._json(404, {"error": {"code": "NOT_FOUND", "message": "Route not found."}})
            return
        try:
            length = min(int(self.headers.get("content-length", "0")), 1_000_000)
            payload = json.loads(self.rfile.read(length) or b"{}")
            job_input: dict[str, Any] = payload.get("input", payload)
            result = handler({"id": "local", "input": job_input})
            self._json(200, result)
        except Exception as exc:
            self._json(400, {"error": {"code": "BAD_REQUEST", "message": str(exc)}})

    def log_message(self, format: str, *args: object) -> None:
        print(f"local-server: {format % args}", flush=True)

    def _json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print("H3 mock contract server listening on 0.0.0.0:8000", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8000), RequestHandler).serve_forever()

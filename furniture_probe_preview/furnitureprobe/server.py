"""Local HTTP backend for the furniture probe GUI.

The GUI is a browser page served by this stdlib-only server.  Every numeric
result shown in the page (probe weights, leak cells, material shades) is
computed here by :mod:`furnitureprobe.lighting` and :mod:`furnitureprobe.render`
— the page only draws them.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .geometry import linear_to_srgb
from .lighting import LightingEngine
from .render import Renderer, aces_tone_map, EXPOSURE
from .scene import Scene, PRESETS, Probe

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _srgb_swatch(linear: list[float]) -> list[int]:
    return [
        int(round(255 * linear_to_srgb(aces_tone_map(v * EXPOSURE))))
        for v in linear
    ]


class Session:
    """Holds the active scene and its baked engine behind a lock."""

    def __init__(self, scene: Scene):
        self.lock = threading.Lock()
        self.scene = scene
        self.engine = LightingEngine(scene)
        self.engine.bake_probes()

    def rebuild(self) -> None:
        engine = LightingEngine(self.scene)
        engine.bake_probes()
        self.engine = engine


def scene_payload(scene: Scene) -> dict:
    d = scene.to_dict()
    for p in d["probes"]:
        p["component"] = next(
            q.component for q in scene.probes if q.id == p["id"]
        )
    return d


class Handler(BaseHTTPRequestHandler):
    session: Session  # injected on the server class

    def log_message(self, *args):  # quiet console
        pass

    # -- helpers -------------------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    # -- routing -------------------------------------------------------------

    def do_GET(self):  # noqa: N802
        try:
            self._handle_get()
        except Exception as exc:  # never drop the connection silently
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _handle_get(self):
        route = urlparse(self.path).path
        sess = self.server.session  # type: ignore[attr-defined]
        if route in ("/", "/index.html"):
            return self._send(200, (WEB_DIR / "index.html").read_bytes(),
                              "text/html; charset=utf-8")
        if route == "/app.js":
            return self._send(200, (WEB_DIR / "app.js").read_bytes(),
                              "application/javascript")
        if route == "/style.css":
            return self._send(200, (WEB_DIR / "style.css").read_bytes(),
                              "text/css")
        if route == "/api/presets":
            return self._json({"presets": list(PRESETS)})
        if route == "/api/scene":
            with sess.lock:
                return self._json(scene_payload(sess.scene))
        if route == "/api/field":
            with sess.lock:
                return self._json(self._field_payload())
        if route.startswith("/api/leak-report"):
            with sess.lock:
                return self._json(self._leak_payload())
        if route.startswith("/api/material/"):
            name = route.rsplit("/", 1)[-1]
            with sess.lock:
                return self._json(self._material_payload(name))
        self._json({"error": "not found"}, 404)

    def do_POST(self):  # noqa: N802
        try:
            self._handle_post()
        except (KeyError, ValueError, FileNotFoundError) as exc:
            self._json({"error": str(exc)}, 400)
        except Exception as exc:  # defensive: surface, don't drop
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def _handle_post(self):
        route = urlparse(self.path).path
        sess = self.server.session  # type: ignore[attr-defined]
        data = self._read_json()
        with sess.lock:
            if route == "/api/load-preset":
                preset = data.get("preset", "uniform")
                if preset not in PRESETS:
                    return self._json({"error": "unknown preset"}, 400)
                sess.scene = PRESETS[preset]()
                sess.rebuild()
                return self._json(scene_payload(sess.scene))
            if route == "/api/move-probe":
                self._move_probe(data)
                sess.rebuild()
                return self._json(scene_payload(sess.scene))
            if route == "/api/add-probe":
                self._add_probe(data)
                sess.rebuild()
                return self._json(scene_payload(sess.scene))
            if route == "/api/move-furniture":
                self._move_furniture(data)
                # Sampling changes; the static probe network does NOT
                # rebake and is never contaminated by furniture colours.
                return self._json(scene_payload(sess.scene))
            if route == "/api/save":
                sess.scene.save(data["path"])
                return self._json({"saved": data["path"]})
            if route == "/api/load":
                sess.scene = Scene.load(data["path"])
                sess.rebuild()
                return self._json(scene_payload(sess.scene))
            if route == "/api/import-probes":
                warnings = sess.engine.import_probe_values(
                    data.get("values", {})
                )
                return self._json({"warnings": warnings})
            if route == "/api/set-fallback":
                policy = data.get("fallback", "nearest")
                if policy not in ("nearest", "dark", "error"):
                    return self._json({"error": "bad policy"}, 400)
                sess.scene.fallback = policy
                return self._json(scene_payload(sess.scene))
        self._json({"error": "not found"}, 404)

    # -- mutations -----------------------------------------------------------

    def _move_probe(self, data: dict) -> None:
        sess = self.server.session  # type: ignore[attr-defined]
        probe = next((p for p in sess.scene.probes if p.id == data["id"]),
                     None)
        if probe is None:
            raise KeyError(f"unknown probe id: {data['id']}")
        probe.x = float(data["x"])
        probe.y = float(data["y"])
        probe.baked = False  # position changed: rebake on next rebuild

    def _add_probe(self, data: dict) -> None:
        sess = self.server.session  # type: ignore[attr-defined]
        existing = {p.id for p in sess.scene.probes}
        pid = data.get("id") or f"P{len(sess.scene.probes) + 1}"
        if pid in existing:
            raise ValueError(f"probe id {pid} already exists")
        sess.scene.probes.append(
            Probe(pid, float(data["x"]), float(data["y"]),
                  float(data.get("coverage", 3.0)))
        )

    def _move_furniture(self, data: dict) -> None:
        sess = self.server.session  # type: ignore[attr-defined]
        f = next((x for x in sess.scene.furniture if x.name == data["id"]),
                 None)
        if f is None:
            raise KeyError(f"unknown furniture id: {data['id']}")
        f.x = float(data["x"])
        f.y = float(data["y"])
        if "rotation" in data:
            f.rotation = float(data["rotation"])

    # -- computed payloads ---------------------------------------------------

    def _field_payload(self) -> dict:
        sess = self.server.session  # type: ignore[attr-defined]
        engine, scene = sess.engine, sess.scene
        g = engine.grid
        field = engine.evaluate_field()
        ref = engine.reference_field()
        cells = []
        for j in range(g.ny):
            for i in range(g.nx):
                s = field[j][i]
                cells.append({
                    "i": i, "j": j,
                    "x": g.cell_center(i, j).x,
                    "y": g.cell_center(i, j).y,
                    "rgb": _srgb_swatch(s.irradiance),
                    "refRgb": _srgb_swatch(ref[j][i]),
                    "status": s.status,
                    "leakRisk": s.leak_risk,
                    "weights": s.weights,
                })
        return {
            "grid": {"nx": g.nx, "ny": g.ny, "dx": g.dx, "dy": g.dy},
            "cells": cells,
            "probes": [
                {
                    "id": p.id, "x": p.x, "y": p.y, "coverage": p.coverage,
                    "component": p.component,
                    "irradiance": p.irradiance,
                    "swatch": _srgb_swatch(p.irradiance),
                }
                for p in scene.probes
            ],
        }

    def _leak_payload(self) -> dict:
        sess = self.server.session  # type: ignore[attr-defined]
        report = sess.engine.leak_report()
        g = sess.engine.grid
        return {
            "grid": {"nx": g.nx, "ny": g.ny, "dx": g.dx, "dy": g.dy},
            "risk": [[i, j] for i, j in report["risk_cells"]],
            "approximation": [
                {"i": i, "j": j, "error": round(e, 4)}
                for i, j, e in report["approximation_cells"]
            ],
            "uncovered": [[i, j] for i, j in report["uncovered_cells"]],
            "maxAbsoluteError": round(report["max_absolute_error"], 4),
            "absTolerance": report["abs_tolerance"],
            "relTolerance": report["rel_tolerance"],
            "fallback": sess.scene.fallback,
        }

    def _material_payload(self, name: str) -> dict:
        sess = self.server.session  # type: ignore[attr-defined]
        f = next((x for x in sess.scene.furniture if x.name == name), None)
        if f is None:
            raise KeyError(f"unknown furniture id: {name}")
        preview = Renderer(sess.engine).material_preview(f)
        preview["shaded_srgb"] = _srgb_swatch(preview["shaded_linear"])
        preview["reference_shaded_srgb"] = _srgb_swatch(
            preview["reference_shaded_linear"]
        )
        return preview


def make_server(session: Session, host: str = "127.0.0.1",
                port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), Handler)
    server.session = session  # type: ignore[attr-defined]
    return server

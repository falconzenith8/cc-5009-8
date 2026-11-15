"""Command-line helper for the furniture probe previewer.

The CLI is auxiliary: it bakes probes, runs headless renders and the test
suites, and serves the GUI.  The documented furniture-probe entry point is
the graphical window (``serve`` then open the printed URL).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .geometry import Point
from .lighting import LightingEngine
from .render import Renderer
from .scene import Scene, PRESETS


def _load(source: str) -> Scene:
    if source in PRESETS:
        return PRESETS[source]()
    return Scene.load(source)


def cmd_bake(args) -> int:
    scene = _load(args.scene)
    engine = LightingEngine(scene, cell=args.cell)
    engine.bake_probes(force=True)
    rows = [
        {"id": p.id, "x": p.x, "y": p.y, "coverage": p.coverage,
         "component": p.component, "irradiance_linear": p.irradiance}
        for p in scene.probes
    ]
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.output}")
    else:
        print(json.dumps(rows, indent=2))
    if args.save_scene:
        scene.save(args.save_scene)
        print(f"scene with baked probes saved to {args.save_scene}")
    return 0


def cmd_render(args) -> int:
    scene = _load(args.scene)
    engine = LightingEngine(scene, cell=args.cell)
    engine.bake_probes()
    renderer = Renderer(engine, scale=args.scale)
    highlight = Point(args.at[0], args.at[1]) if args.at else None
    renderer.render(use_reference=False, highlight=highlight).save_ppm(
        args.output
    )
    ref_path = args.output.replace(".ppm", ".reference.ppm")
    renderer.render(use_reference=True, highlight=highlight).save_ppm(ref_path)
    print(f"probe render: {args.output}")
    print(f"reference:    {ref_path}")
    return 0


def cmd_leak(args) -> int:
    scene = _load(args.scene)
    engine = LightingEngine(scene, cell=args.cell)
    engine.bake_probes()
    report = engine.leak_report()
    summary = {
        "max_absolute_error": round(report["max_absolute_error"], 4),
        "abs_tolerance": report["abs_tolerance"],
        "rel_tolerance": report["rel_tolerance"],
        "structural_leak_risk_cells": len(report["risk_cells"]),
        "probe_approximation_cells": len(report["approximation_cells"]),
        "uncovered_fallback_cells": len(report["uncovered_cells"]),
        "fallback_policy": scene.fallback,
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_sample(args) -> int:
    scene = _load(args.scene)
    engine = LightingEngine(scene, cell=args.cell)
    engine.bake_probes()
    s = engine.sample(Point(args.at[0], args.at[1]))
    print(json.dumps({
        "point": [s.point.x, s.point.y],
        "component": s.component,
        "status": s.status,
        "reason": s.reason,
        "irradiance_linear": s.irradiance,
        "weights": s.weights,
        "leak_risk": s.leak_risk,
        "rejected_cross_wall_probes": s.cross_component_probes,
    }, indent=2))
    return 0


def cmd_serve(args) -> int:
    from .server import Session, make_server

    scene = _load(args.scene)
    session = Session(scene)
    server = make_server(session, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}/"
    print(f"Furniture probe previewer GUI: {url}")
    print("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_acceptance(args) -> int:
    """Automated entry point for the interpolation-correctness acceptance."""
    import unittest

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName("tests.test_interpolation_acceptance")
    result = unittest.TextTestRunner(verbosity=args.verbose and 2 or 1).run(
        suite
    )
    return 0 if result.wasSuccessful() else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="furniture-probe",
        description="Furniture display lighting probe previewer (CLI helper; "
                    "the GUI is the primary entry point).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("scene", help="preset name (uniform|two-room|sparse) "
                                     "or path to a saved .json scene")
        p.add_argument("--cell", type=float, default=0.2,
                       help="analysis grid cell size in metres")

    p = sub.add_parser("bake", help="bake probes from static lights")
    common(p)
    p.add_argument("-o", "--output", help="write probe JSON here")
    p.add_argument("--save-scene", help="save scene carrying baked values")
    p.set_defaults(func=cmd_bake)

    p = sub.add_parser("render", help="headless PPM render + reference")
    common(p)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--scale", type=int, default=60)
    p.add_argument("--at", nargs=2, type=float, metavar=("X", "Y"),
                   help="point to mark in the render")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("leak", help="report leak risk and coverage gaps")
    common(p)
    p.set_defaults(func=cmd_leak)

    p = sub.add_parser("sample", help="inspect one interpolated point")
    common(p)
    p.add_argument("at", nargs=2, type=float, metavar=("X", "Y"))
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("serve", help="launch the graphical preview window")
    common(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser(
        "acceptance",
        help="run the interpolation-correctness acceptance test suite",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_acceptance)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

"""Kernel-driven furniture-probe regression checks.

The same evaluation runs headless (acceptance tests, CLI) and inside the GUI
material preview, so a failure seen on screen is the same verdict the test
suite produces.  Each check reports the dominant probe id, the directly read
interpolation weights and an explicit lighting error against the analytic
direct-light solution — never a pass derived from default/zero values.
"""

from __future__ import annotations

import math

from .geometry import Point
from .lighting import LightingEngine, direct_irradiance
from .scene import Scene

# Lighting-error budget shared by the acceptance tests and the GUI.  Kept
# tight on purpose: the validation furniture sits exactly on a probe, so the
# correct network has essentially zero error, while the controlled corrupt
# value (60 per channel) fails by construction.
LIGHTING_ERROR_BUDGET = 5.0
_MIN_EVIDENCE_WEIGHT = 0.5


def _l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[k] - b[k]) ** 2 for k in range(3)))


def evaluate_furniture(engine: LightingEngine, name: str) -> dict:
    """Run one furniture regression sample through the interpolation kernel.

    The expected irradiance is the direct-light reference evaluated at the
    furniture position; the observed value is the actual probe sample the
    renderer would shade with.
    """
    scene = engine.scene
    furniture = next((f for f in scene.furniture if f.name == name), None)
    if furniture is None:
        raise KeyError(f"unknown furniture: {name}")
    point = Point(furniture.x, furniture.y)
    sample = engine.sample(point)
    reference = direct_irradiance(point, scene)
    error = _l2(sample.irradiance, reference)

    # A verdict needs real evidence: an actual participating probe carrying
    # most of the weight, finite values and a covered status.  An empty or
    # default-looking sample can never masquerade as success.
    has_evidence = (
        sample.status == "covered"
        and bool(sample.weights)
        and all(math.isfinite(v) for v in sample.irradiance)
        and max(sample.weights.values()) >= _MIN_EVIDENCE_WEIGHT
    )
    dominant = max(sample.weights, key=sample.weights.get) if sample.weights \
        else None

    failures: list[str] = []
    if not has_evidence:
        failures.append(
            f"insufficient interpolation evidence "
            f"(status={sample.status}, weights={sample.weights})"
        )
    if error > LIGHTING_ERROR_BUDGET:
        failures.append(
            f"lighting error {error:.4f} exceeds budget "
            f"{LIGHTING_ERROR_BUDGET:.4f}"
        )

    return {
        "furniture": name,
        "position": [furniture.x, furniture.y],
        "status": sample.status,
        "reason": sample.reason,
        "weights": dict(sample.weights),
        "dominant_probe": dominant,
        "probe_irradiance": list(sample.irradiance),
        "reference_irradiance": list(reference),
        "lighting_error": error,
        "error_budget": LIGHTING_ERROR_BUDGET,
        "rejected_probes": list(sample.cross_component_probes),
        "passed": not failures,
        "failures": failures,
    }


def run_regression(scene: Scene) -> dict:
    """Evaluate every furniture probe sample; overall pass needs all of them."""
    engine = LightingEngine(scene)
    engine.bake_probes()
    checks = [evaluate_furniture(engine, f.name) for f in scene.furniture]
    failed_probes = sorted({
        c["dominant_probe"] for c in checks
        if not c["passed"] and c["dominant_probe"] is not None
    })
    return {
        "scene": scene.name,
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "failed_probe_ids": failed_probes,
    }

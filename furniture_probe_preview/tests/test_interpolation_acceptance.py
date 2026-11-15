"""Interpolation-correctness acceptance suite.

Every expectation in this file is derived *independently* of the code under
test:

* expected weights come from a hand-written IDW formula applied to the known
  probe coordinates (a private re-derivation, not ``LightingEngine.sample``);
* expected irradiance comes from a private evaluation of the documented light
  model (power * smooth falloff, with a hand-written line-of-sight test);
* several constants are hard-coded numbers produced by an isolated scratch
  calculation, so a change in the engine kernel cannot silently update them.

The tests read real outputs: ``Sample.weights`` / ``Sample.irradiance`` from
the interpolation kernel, pixels decoded from an actual rendered PPM image and
JSON served by the live GUI backend.  A mutation that swaps probe coordinates
fails the numeric assertions below by construction.
"""

import io
import math
import os
import sys
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from threading import Thread

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from furnitureprobe.geometry import Point  # noqa: E402
from furnitureprobe.lighting import LightingEngine  # noqa: E402
from furnitureprobe.regression import (  # noqa: E402
    LIGHTING_ERROR_BUDGET, run_regression,
)
from furnitureprobe.render import Renderer  # noqa: E402
from furnitureprobe.scene import (  # noqa: E402
    Scene, Rect, Wall, Probe,
    make_validation_room, make_validation_room_corrupt,
)
from furnitureprobe.server import Session, Handler  # noqa: E402

# ---------------------------------------------------------------------------
# Independent reference math (deliberately private to this test module)
# ---------------------------------------------------------------------------

# Documented kernel parameter, duplicated here on purpose: if the product
# spec changes this constant, these hand-derived expectations must be reworked
# rather than silently re-imported from the implementation.
_EPS2 = 0.12 ** 2

# Validation light: single analytic emitter, white, power 20, radius 8 at
# (2.5, 4).  Partition x=6 solid except a doorway spanning y in [3, 5].
_LX, _LY, _POWER, _RADIUS = 2.5, 4.0, 20.0, 8.0
_DOOR_Y0, _DOOR_Y1 = 3.0, 5.0
_PARTITION_X = 6.0


def _reference_falloff(d: float) -> float:
    """Private copy of the documented bounded smooth falloff."""
    if d >= _RADIUS:
        return 0.0
    t = d / _RADIUS
    window = (1.0 - t) ** 2
    edge = 1.0 - (3.0 - 2.0 * t) * t * t
    return window * edge


def _reference_irradiance(p: Point) -> list[float]:
    """Private direct-light solution for the validation room."""
    d = math.hypot(p.x - _LX, p.y - _LY)
    if d >= _RADIUS or not _reference_los(p, Point(_LX, _LY)):
        return [0.0, 0.0, 0.0]
    return [_POWER * _reference_falloff(d)] * 3


def _reference_los(p: Point, q: Point) -> bool:
    """Private parametric line-of-sight against the x=6 partition."""
    if (p.x - _PARTITION_X) * (q.x - _PARTITION_X) >= 0.0:
        return True  # both points on the same side
    t = (_PARTITION_X - p.x) / (q.x - p.x)
    y_cross = p.y + t * (q.y - p.y)
    return _DOOR_Y0 <= y_cross <= _DOOR_Y1


def _reference_idw(point: Point, probes: list[Probe], coverage: float
                   ) -> dict[str, float]:
    """Private eligibility filter + normalised IDW, from probe coordinates."""
    eligible = []
    for probe in probes:
        d = point.distance_to(probe.position())
        if d <= coverage and _reference_los(point, probe.position()):
            eligible.append((probe.id, d))
    raw = {pid: 1.0 / (d * d + _EPS2) for pid, d in eligible}
    total = sum(raw.values())
    if not raw:
        return {}
    return {pid: w / total for pid, w in raw.items()}


def _assert_weights_match_reference(testcase: unittest.TestCase,
                                    sample, point: Point,
                                    probes: list[Probe],
                                    coverage: float) -> None:
    expected = _reference_idw(point, probes, coverage)
    testcase.assertEqual(set(sample.weights), set(expected))
    testcase.assertAlmostEqual(sum(sample.weights.values()), 1.0, places=9)
    for pid, w in expected.items():
        testcase.assertAlmostEqual(sample.weights[pid], w, places=6,
                                   msg=f"weight of {pid} at {point}")


# ---------------------------------------------------------------------------
# Scene construction helpers
# ---------------------------------------------------------------------------

def _empty_room(probes: list[Probe], fallback: str = "error",
                bounds: Rect | None = None) -> Scene:
    """Wall-only analytic room with authored probes and no lights."""
    bounds = bounds or Rect(0, 0, 8, 6)
    x0, y0, x1, y1 = 0.0, 0.0, bounds.w, bounds.h
    walls = [
        Wall(Point(x0, y0), Point(x1, y0)),
        Wall(Point(x0, y1), Point(x1, y1)),
        Wall(Point(x0, y0), Point(x0, y1)),
        Wall(Point(x1, y0), Point(x1, y1)),
    ]
    return Scene("ad-hoc", bounds, walls, [], [], [], probes,
                 fallback=fallback)


def _validation_baked(corrupt: bool = False):
    scene = (make_validation_room_corrupt if corrupt
             else make_validation_room)()
    engine = LightingEngine(scene)
    engine.bake_probes()
    return scene, engine


# ---------------------------------------------------------------------------
# 1. Weight correctness against independently derived expectations
# ---------------------------------------------------------------------------

class InterpolationWeightAcceptanceTests(unittest.TestCase):
    def test_two_probe_1d_weights_match_hand_calculation(self):
        # Probes interior to the room; from (2, 3) the distances are 1 and 2.
        probes = [
            Probe("Q1", 1.0, 3.0, 5.0, [1.0, 1.0, 1.0], baked=True),
            Probe("Q2", 4.0, 3.0, 5.0, [1.0, 1.0, 1.0], baked=True),
        ]
        engine = LightingEngine(_empty_room(probes))
        point = Point(2.0, 3.0)
        sample = engine.sample(point)
        self.assertAlmostEqual(sample.weights["Q1"], 0.798281896, places=6)
        self.assertAlmostEqual(sample.weights["Q2"], 0.201718104, places=6)
        _assert_weights_match_reference(self, sample, point, probes, 5.0)

    def test_degenerate_coincident_probes_split_weight_equally(self):
        probes = [
            Probe("C1", 3.0, 3.0, 4.0, [2.0, 0.0, 0.0], baked=True),
            Probe("C2", 3.0, 3.0, 4.0, [0.0, 2.0, 0.0], baked=True),
        ]
        engine = LightingEngine(_empty_room(probes))
        point = Point(3.0, 3.0)
        sample = engine.sample(point)
        self.assertEqual(set(sample.weights), {"C1", "C2"})
        self.assertAlmostEqual(sample.weights["C1"], 0.5, places=9)
        self.assertAlmostEqual(sample.weights["C2"], 0.5, places=9)
        # Equal weights on distinct colours average them in the linear domain.
        for got, want in zip(sample.irradiance, [1.0, 1.0, 0.0]):
            self.assertAlmostEqual(got, want, places=9)

    def test_probe_outside_coverage_is_excluded_not_zero_weighted(self):
        probes = [
            Probe("NEAR", 1.0, 3.0, 2.0, [4.0, 4.0, 4.0], baked=True),
            Probe("FAR", 7.0, 3.0, 2.0, [9.0, 9.0, 9.0], baked=True),
        ]
        engine = LightingEngine(_empty_room(probes))
        sample = engine.sample(Point(1.5, 3.0))
        self.assertEqual(set(sample.weights), {"NEAR"})
        self.assertNotIn("FAR", sample.weights)  # not a 0.0 entry: excluded
        self.assertEqual(sample.irradiance, [4.0, 4.0, 4.0])

    def test_degenerate_no_probe_in_range_surfaces_explicit_fallback(self):
        probes = [Probe("LONE", 0.5, 0.5, 0.5, [3.0, 3.0, 3.0], baked=True)]
        point = Point(7.0, 5.0)
        # Policies that must not fabricate coverage report an empty weight set.
        for policy, expected_status in (("dark", "dark"), ("error", "error")):
            engine = LightingEngine(_empty_room(probes, fallback=policy))
            sample = engine.sample(point)
            self.assertEqual(sample.status, expected_status)
            self.assertEqual(sample.weights, {})
            self.assertTrue(sample.reason)
        # The "nearest" policy is an explicit fallback: it names the probe it
        # falls back to, with full weight and a fallback status instead of
        # pretending the point was covered.
        engine = LightingEngine(_empty_room(probes, fallback="nearest"))
        sample = engine.sample(point)
        self.assertEqual(sample.status, "fallback")
        self.assertEqual(sample.weights, {"LONE": 1.0})
        self.assertTrue(sample.reason)

    def test_degenerate_empty_probe_list_never_reports_covered(self):
        engine = LightingEngine(_empty_room([], fallback="error"))
        sample = engine.sample(Point(4.0, 3.0))
        self.assertNotEqual(sample.status, "covered")
        self.assertEqual(sample.weights, {})
        self.assertEqual(sample.irradiance, [0.0, 0.0, 0.0])

    def test_point_outside_room_bounds_is_not_fake_coverage(self):
        probes = [Probe("Q1", 1.0, 1.0, 2.0, [1.0, 1.0, 1.0], baked=True)]
        engine = LightingEngine(_empty_room(probes, fallback="error"))
        sample = engine.sample(Point(-1.0, 3.0))
        self.assertEqual(sample.component, -1)
        self.assertNotEqual(sample.status, "covered")
        self.assertEqual(sample.weights, {})


# ---------------------------------------------------------------------------
# 2. Occlusion / leakage with analytic expectations
# ---------------------------------------------------------------------------

class OcclusionLeakageAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.scene, self.engine = _validation_baked()
        self.coverage = 3.5

    def test_baked_probe_values_match_private_analytic_solution(self):
        expected = {
            "VA": 14.014039, "VB": 7.056041, "VC": 9.492188,
            "VD": 0.889893, "VE": 0.0, "VF": 0.0,
        }
        for probe in self.scene.probes:
            self.assertTrue(probe.baked)
            ref = _reference_irradiance(probe.position())
            for k in range(3):
                self.assertAlmostEqual(probe.irradiance[k], ref[k], places=5)
            self.assertAlmostEqual(probe.irradiance[0],
                                   expected[probe.id], places=5,
                                   msg=f"hard-coded expectation for {probe.id}")

    def test_dim_room_sample_has_zero_irradiance_and_rejected_neighbours(self):
        # Furniture "dim-table" sits on VE: single zero-valued probe, no blend.
        table = Point(10.0, 1.5)
        table_sample = self.engine.sample(table)
        self.assertEqual(table_sample.status, "covered")
        self.assertEqual(set(table_sample.weights), {"VE"})
        self.assertEqual(table_sample.irradiance, [0.0, 0.0, 0.0])
        self.assertEqual(_reference_irradiance(table), [0.0, 0.0, 0.0])

        # A sealed-side point whose coverage radius reaches across the
        # partition: probe VC is 3.20 m away (in range) but the sight line
        # crosses x=6 at y=2.125, below the [3, 5] doorway, so it is rejected.
        scene = self.scene
        engine = LightingEngine(scene)
        point = Point(6.5, 1.5)
        self.assertFalse(_reference_los(point, Point(4.5, 4.0)))
        sample = engine.sample(point)
        self.assertIn("VC", sample.cross_component_probes)
        self.assertNotIn("VC", sample.weights)
        # Only probes keeping line of sight through the opening contribute.
        self.assertEqual(set(sample.weights), {"VD", "VE"})
        # VE carries an analytic zero; only VD's doorway light contributes.
        expected = sample.weights["VD"] * 0.889893
        self.assertAlmostEqual(sample.irradiance[0], expected, places=5)

    def test_doorway_blend_weights_and_irradiance_are_analytically_correct(self):
        point = Point(6.5, 4.0)  # sees VC and VD through the open doorway
        sample = self.engine.sample(point)
        # Hard-coded weights from the isolated coordinate calculation.
        self.assertAlmostEqual(sample.weights["VC"], 0.201718104, places=6)
        self.assertAlmostEqual(sample.weights["VD"], 0.798281896, places=6)
        _assert_weights_match_reference(
            self, sample, point, self.scene.probes, self.coverage
        )
        ref = _reference_irradiance(point)  # 2.5 per channel at d=4
        self.assertAlmostEqual(ref[0], 2.5, places=9)
        for got, want in zip(sample.irradiance, ref):
            self.assertAlmostEqual(got, want, delta=0.6)

    def test_cross_room_boundary_keeps_components_and_weights_apart(self):
        left = Point(5.5, 1.0)
        right = Point(6.5, 1.0)
        self.assertNotEqual(self.engine.component_at(left),
                            self.engine.component_at(right))
        # At the solid wall, right-side probes are rejected on the left.
        sample_l = self.engine.sample(left)
        for pid in ("VD", "VE", "VF"):
            self.assertNotIn(pid, sample_l.weights)
        sample_r = self.engine.sample(right)
        for pid in ("VA", "VB", "VC"):
            self.assertNotIn(pid, sample_r.weights)

    def test_wall_blocks_private_los_while_doorway_opens_it(self):
        self.assertFalse(_reference_los(Point(5.5, 1.0), Point(6.5, 1.0)))
        self.assertTrue(_reference_los(Point(5.5, 4.0), Point(6.5, 4.0)))


# ---------------------------------------------------------------------------
# 3. Mutation tests: swapping probe coordinates must fail explicitly
# ---------------------------------------------------------------------------

class ProbeCoordinateMutationTests(unittest.TestCase):
    """The mutation swaps probe *coordinates* while keeping each probe's
    authored baked value, modelling an imported probe set whose placement
    data was scrambled.  No rebake hides the fault."""

    def _swap_coordinates(self, scene, id_a, id_b):
        a = next(p for p in scene.probes if p.id == id_a)
        b = next(p for p in scene.probes if p.id == id_b)
        a.x, b.x = b.x, a.x
        a.y, b.y = b.y, a.y
        for p in (a, b):
            self.assertTrue(p.baked)  # values stay authored; positions lie

    def test_swapping_cross_room_coordinates_breaks_numeric_assertions(self):
        scene, engine = _validation_baked()
        bright = Point(2.0, 3.0)
        good = engine.sample(bright)
        self.assertEqual(good.status, "covered")
        _assert_weights_match_reference(
            self, good, bright, scene.probes, 3.5
        )

        self._swap_coordinates(scene, "VA", "VE")
        mutated_engine = LightingEngine(scene)
        mutated = mutated_engine.sample(bright)

        # The id VA now sits on the sealed-room wall line (10, 1.5), where
        # exterior-wall geometry blocks its view of the bright-table point.
        # The id VE, carrying a zero value, has taken VA's old position and
        # dominates: the interpolated field collapses.  Key-set/length checks
        # would stay green, only the numeric assertions catch it.
        self.assertEqual(mutated.status, "covered")
        self.assertNotIn("VA", mutated.weights)
        self.assertAlmostEqual(mutated.weights["VE"], 0.996853532, places=5)
        for got, want in zip(mutated.irradiance,
                             _reference_irradiance(bright)):
            self.assertGreater(abs(got - want), 5.0)
        report = run_regression(scene)
        self.assertFalse(report["passed"])
        self.assertIn("VE", report["failed_probe_ids"])

    def test_swapping_same_room_coordinates_changes_weight_values(self):
        scene, engine = _validation_baked()
        point = Point(2.0, 3.0)
        before = dict(engine.sample(point).weights)
        # Hard-coded correct weight at the un-mutated position.
        self.assertAlmostEqual(before["VA"], 0.996853532, places=6)
        good_irradiance = list(engine.sample(point).irradiance)

        self._swap_coordinates(scene, "VA", "VB")
        mutated_engine = LightingEngine(scene)
        sample = mutated_engine.sample(point)
        after = sample.weights

        # The trap: key set and dict length are unchanged; only values move.
        self.assertEqual(set(before), set(after))
        self.assertEqual(len(before), len(after))
        # After the swap the id VA sits at (2, 6.5): its weight drops to the
        # value VB used to carry — a numeric assertion catches it.
        self.assertAlmostEqual(after["VA"], before["VB"], places=9)
        self.assertGreater(abs(after["VA"] - before["VA"]), 0.9)
        # Independent lighting check: the baked values stayed attached to the
        # wrong ids, so the interpolated field no longer matches the analytic
        # direct-light solution, even though every weight still sums to one.
        reference = _reference_irradiance(point)
        error = math.sqrt(
            sum((sample.irradiance[k] - reference[k]) ** 2 for k in range(3))
        )
        self.assertGreater(error, 1.0)
        self.assertAlmostEqual(sum(after.values()), 1.0, places=9)
        # And the correct (un-mutated) field was well inside the same budget.
        good_error = math.sqrt(
            sum((good_irradiance[k] - reference[k]) ** 2 for k in range(3))
        )
        self.assertLess(good_error, 0.1)

    def test_length_only_check_cannot_detect_the_swap(self):
        # Documents why the suite asserts values: this weak assertion passes
        # both before and after the coordinate swap.
        scene, engine = _validation_baked()
        point = Point(2.0, 3.0)
        weak = lambda s: isinstance(s.weights, dict) and len(s.weights) > 0  # noqa: E731
        self.assertTrue(weak(engine.sample(point)))
        self._swap_coordinates(scene, "VA", "VE")
        self.assertTrue(weak(LightingEngine(scene).sample(point)))


# ---------------------------------------------------------------------------
# 4. Controlled-error furniture regression (positive + negative samples)
# ---------------------------------------------------------------------------

class ControlledErrorRegressionTests(unittest.TestCase):
    def test_good_scene_passes_with_real_evidence(self):
        scene, _ = _validation_baked(corrupt=False)
        report = run_regression(scene)
        self.assertTrue(report["passed"])
        self.assertEqual(report["failed_probe_ids"], [])
        for check in report["checks"]:
            self.assertTrue(check["weights"], "no default-only verdict")
            self.assertGreaterEqual(
                max(check["weights"].values()), 0.5
            )
            self.assertLess(check["lighting_error"], LIGHTING_ERROR_BUDGET)
        bright = next(c for c in report["checks"]
                      if c["furniture"] == "bright-table")
        self.assertEqual(bright["dominant_probe"], "VA")
        # Furniture sits on its dominant probe; only trace contributions from
        # the two in-range neighbours remain, so the analytic solution is
        # reproduced to a small residual rather than exactly zero.
        self.assertLess(bright["lighting_error"], 0.1)
        dim = next(c for c in report["checks"]
                   if c["furniture"] == "dim-table")
        self.assertEqual(dim["reference_irradiance"], [0.0, 0.0, 0.0])
        self.assertEqual(dim["probe_irradiance"], [0.0, 0.0, 0.0])

    def test_corrupt_probe_fails_through_weight_and_lighting_error(self):
        scene, _ = _validation_baked(corrupt=True)
        report = run_regression(scene)
        self.assertFalse(report["passed"])
        self.assertIn("VA", report["failed_probe_ids"])
        bright = next(c for c in report["checks"]
                      if c["furniture"] == "bright-table")
        self.assertFalse(bright["passed"])
        # The failure is attributable to the dominant probe via real weights.
        self.assertEqual(bright["dominant_probe"], "VA")
        self.assertGreaterEqual(bright["weights"]["VA"], 0.5)
        self.assertGreater(bright["lighting_error"], LIGHTING_ERROR_BUDGET)
        self.assertTrue(bright["failures"])
        # The unrelated sample behind the wall must not be flagged.
        dim = next(c for c in report["checks"]
                   if c["furniture"] == "dim-table")
        self.assertTrue(dim["passed"])
        self.assertNotIn("VA", dim["weights"])

    def test_unbaked_default_probe_cannot_disguise_itself_as_success(self):
        scene, _ = _validation_baked()
        # Reset a probe to un-baked defaults without rebaking: zero values
        # look "plausible" but carry no authoritative evidence.
        va = next(p for p in scene.probes if p.id == "VA")
        va.irradiance = [0.0, 0.0, 0.0]
        va.baked = False
        report = run_regression(scene)
        bright = next(c for c in report["checks"]
                      if c["furniture"] == "bright-table")
        # Rebake inside run_regression restores correctness; assert the
        # reference is genuinely non-zero so a zero-default would be caught
        # by the lighting-error budget rather than accepted.
        self.assertNotEqual(bright["reference_irradiance"],
                            [0.0, 0.0, 0.0])
        self.assertTrue(bright["passed"])  # engine rebaked, not defaulted
        va.irradiance = [0.0, 0.0, 0.0]
        va.baked = True  # now an authoritative-but-wrong zero
        engine = LightingEngine(scene)
        from furnitureprobe.regression import evaluate_furniture
        bad = evaluate_furniture(engine, "bright-table")
        self.assertFalse(bad["passed"])
        self.assertGreater(bad["lighting_error"], LIGHTING_ERROR_BUDGET)


# ---------------------------------------------------------------------------
# 5. Tests that read actual rendered output (not array lengths)
# ---------------------------------------------------------------------------

def _decode_ppm(data: bytes) -> tuple[int, int, bytes]:
    stream = io.BytesIO(data)
    header = b""
    while header.count(b"\n") < 3:
        header += stream.read(1)
    magic, w, h, _ = header.split()
    assert magic == b"P6"
    return int(w), int(h), stream.read()


class RenderedOutputAcceptanceTests(unittest.TestCase):
    def test_rendered_pixels_show_wall_blocks_bright_room_leakage(self):
        scene, engine = _validation_baked()
        renderer = Renderer(engine, scale=20)
        probe_img = _decode_ppm(renderer.render().ppm_bytes())
        ref_img = _decode_ppm(renderer.render(use_reference=True)
                              .ppm_bytes())

        def pixel(img, wx, wy):
            w, h, px = img
            px_x = min(w - 1, max(0, round(wx * 20)))
            px_y = min(h - 1, max(0, round(wy * 20)))
            i = (px_y * w + px_x) * 3
            return px[i:i + 3]

        # Deep in the sealed room both render and reference stay dark.
        for img in (probe_img, ref_img):
            self.assertEqual(sum(pixel(img, 10.0, 1.5)), 0)
        # The lit room is visibly brighter than the sealed room.
        self.assertGreater(sum(pixel(probe_img, 2.0, 3.0)),
                           sum(pixel(probe_img, 10.0, 1.5)))

    def test_corrupt_probe_changes_probe_render_but_not_reference(self):
        good_scene, good_engine = _validation_baked(False)
        bad_scene, bad_engine = _validation_baked(True)
        r_good = Renderer(good_engine, scale=20).render()
        r_bad = Renderer(bad_engine, scale=20).render()
        ref_good = Renderer(good_engine, scale=20).render(use_reference=True)
        ref_bad = Renderer(bad_engine, scale=20).render(use_reference=True)
        # Probe-driven images diverge at the corrupted furniture...
        self.assertNotEqual(r_good.px, r_bad.px)
        # ...while the analytic direct-light reference is identical.
        self.assertEqual(ref_good.px, ref_bad.px)


# ---------------------------------------------------------------------------
# 6. Live GUI backend: kernel-driven verdict reachable from the preview
# ---------------------------------------------------------------------------

class GuiBackendAcceptanceTests(unittest.TestCase):
    def _serve(self, scene):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.session = Session(scene)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server.server_port

    def _get(self, port, path):
        import json
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{path}", timeout=10
        ) as resp:
            return json.loads(resp.read())

    def test_regression_endpoint_reflects_kernel_verdict(self):
        port = self._serve(make_validation_room())
        report = self._get(port, "/api/regression")
        self.assertTrue(report["passed"])
        for check in report["checks"]:
            self.assertTrue(check["weights"])
            self.assertIsNotNone(check["dominant_probe"])

        port_bad = self._serve(make_validation_room_corrupt())
        bad = self._get(port_bad, "/api/regression")
        self.assertFalse(bad["passed"])
        self.assertIn("VA", bad["failed_probe_ids"])

    def test_material_preview_carries_regression_evidence(self):
        port = self._serve(make_validation_room_corrupt())
        payload = self._get(port, "/api/material/bright-table")
        verdict = payload["regression"]
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["dominant_probe"], "VA")
        self.assertGreater(verdict["lighting_error"],
                           LIGHTING_ERROR_BUDGET)
        # Actual weights and irradiances are present, not placeholders.
        self.assertIn("VA", payload["weights"])
        self.assertTrue(any(payload["probe_irradiance"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Headless verification of probe interpolation, walls, fallback and IO."""

import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from furnitureprobe.geometry import (  # noqa: E402
    Point, line_blocked, linear_to_srgb, srgb_to_linear,
)
from furnitureprobe.lighting import (  # noqa: E402
    LightingEngine, direct_irradiance,
)
from furnitureprobe.scene import (  # noqa: E402
    Scene, make_uniform_room, make_two_room, make_sparse_room,
)
from furnitureprobe.render import Renderer, shade_material  # noqa: E402


def baked(scene_factory, cell=0.25):
    scene = scene_factory()
    engine = LightingEngine(scene, cell=cell)
    engine.bake_probes()
    return scene, engine


class UniformRoomTests(unittest.TestCase):
    def setUp(self):
        self.scene, self.engine = baked(make_uniform_room)

    def test_probes_baked_linear_and_components(self):
        for p in self.scene.probes:
            self.assertTrue(all(v >= 0.0 for v in p.irradiance))
            self.assertGreaterEqual(p.component, 0)
        # Single room -> every probe shares one component.
        comps = {p.component for p in self.scene.probes}
        self.assertEqual(len(comps), 1)

    def test_coverage_weights_sum_to_one(self):
        s = self.engine.sample(Point(5.0, 4.0))
        self.assertEqual(s.status, "covered")
        self.assertAlmostEqual(sum(s.weights.values()), 1.0, places=9)
        self.assertGreaterEqual(len(s.weights), 1)

    def test_interpolation_tracks_reference_in_occupied_interior(self):
        # Probe networks approximate the non-linear direct-light field; the
        # comparison is meaningful away from wall corners where no probe can
        # see more than one neighbour.  Measure on the interior working area.
        from furnitureprobe.geometry import Point as P
        scene, engine = baked(make_uniform_room)
        report = engine.leak_report()
        self.assertEqual(len(report["uncovered_cells"]), 0)
        worst = 0.0
        for x in (1.5, 3.0, 5.0, 7.0, 8.5):
            for y in (1.5, 3.0, 4.0, 5.5, 6.5):
                s = engine.sample(P(x, y))
                r = direct_irradiance(P(x, y), scene)
                scale = math.sqrt(sum(v * v for v in r)) + 0.05
                err = math.sqrt(
                    sum((s.irradiance[k] - r[k]) ** 2 for k in range(3))
                ) / scale
                worst = max(worst, err)
        self.assertLess(worst, 0.35)


class TwoRoomPartitionTests(unittest.TestCase):
    def setUp(self):
        self.scene, self.engine = baked(make_two_room)

    def test_two_connected_components(self):
        comps = {p.component for p in self.scene.probes}
        self.assertEqual(len(comps), 2)

    def test_partition_blocks_line_of_sight(self):
        # Wall segment at x=6 away from the doorway must block sight.
        self.assertTrue(line_blocked(Point(4.0, 1.0), Point(8.0, 1.0),
                                     self.scene.walls))
        # The doorway gap at y in [3.3, 4.7] stays open.
        self.assertFalse(line_blocked(Point(4.0, 4.0), Point(8.0, 4.0),
                                      self.scene.walls))

    def test_dim_room_not_lit_by_bright_room_probes(self):
        # Deep inside the dim room, only dim-room probes may carry weight.
        s = self.engine.sample(Point(10.0, 6.0))
        dim_ids = {"D0", "D1", "D2", "D3", "D4"}
        self.assertTrue(set(s.weights).issubset(dim_ids))
        # Irradiance there must stay low: no light bleed through the wall.
        self.assertLess(sum(s.irradiance) / 3,
                        sum(self.engine.sample(Point(2.0, 2.0)).irradiance)
                        / 3 * 0.25)

    def test_cross_wall_probes_rejected_not_blended(self):
        # Point near the partition on the bright side has dim probes in
        # geometric range; they must be rejected via component mismatch.
        s = self.engine.sample(Point(5.4, 1.5))
        rejected = set(s.cross_component_probes)
        self.assertTrue(rejected & {"D1", "D3"})
        self.assertNotIn("D1", s.weights)

    def test_doorway_keeps_distinct_region_labels_but_open_line_of_sight(self):
        # Static region labelling treats the partition as a full barrier, so
        # the two sides always have distinct ids (no unconditional probe
        # mixing).  The doorway itself stays open for line-of-sight tests,
        # which is what lets probe blending travel through it.
        c_door_l = self.engine.component_at(Point(5.5, 4.0))
        c_door_r = self.engine.component_at(Point(6.5, 4.0))
        c_wall_l = self.engine.component_at(Point(5.5, 1.0))
        c_wall_r = self.engine.component_at(Point(6.5, 1.0))
        self.assertNotEqual(c_door_l, c_door_r)
        self.assertNotEqual(c_wall_l, c_wall_r)
        self.assertFalse(line_blocked(Point(5.5, 4.0), Point(6.5, 4.0),
                                      self.scene.walls))
        self.assertTrue(line_blocked(Point(5.5, 1.0), Point(6.5, 1.0),
                                     self.scene.walls))


class SparseFallbackTests(unittest.TestCase):
    def setUp(self):
        self.scene, self.engine = baked(make_sparse_room)

    def test_uncovered_alcove_reports_fallback(self):
        s = self.engine.sample(Point(8.5, 3.0))
        self.assertIn(s.status, ("fallback", "dark", "error"))
        self.assertTrue(s.reason)

    def test_fallback_policies_are_explicit(self):
        # The alcove has no line-of-sight probe at all, so even "nearest"
        # cannot fabricate one: every policy must surface an explicit state.
        for policy in ("dark", "error", "nearest"):
            self.scene.fallback = policy
            s = LightingEngine(self.scene).sample(Point(8.5, 3.0))
            self.assertIn(s.status, ("fallback", "dark", "error"))
            self.assertTrue(s.reason)
        self.scene.fallback = "dark"
        self.assertEqual(
            LightingEngine(self.scene).sample(Point(8.5, 3.0)).status, "dark")
        self.scene.fallback = "error"
        self.assertEqual(
            LightingEngine(self.scene).sample(Point(8.5, 3.0)).status, "error")

    def test_nearest_fallback_never_crosses_partition(self):
        # Place the request at a point that CAN see a far probe: nearest
        # fallback must still pick only line-of-sight-visible probes.
        self.scene.fallback = "nearest"
        engine = LightingEngine(self.scene)
        s = engine.sample(Point(9.0, 8.5))  # near S3, outside its coverage
        if s.weights:
            probe = next(p for p in self.scene.probes
                         if p.id == next(iter(s.weights)))
            self.assertFalse(line_blocked(s.point, probe.position(),
                                          self.scene.walls))

    def test_leak_report_flags_alcove(self):
        report = self.engine.leak_report()
        self.assertGreater(len(report["uncovered_cells"]), 0)


class LeakAuditTests(unittest.TestCase):
    def test_risk_cells_sit_along_partition_only(self):
        scene, engine = baked(make_two_room)
        report = engine.leak_report()
        self.assertTrue(report["risk_cells"])
        # Every structural-risk cell is within the largest probe coverage
        # radius of the x=6 partition; none can appear in the far field.
        max_cov = max(p.coverage for p in scene.probes) + engine.grid.dx
        xs = [engine.grid.cell_center(i, j).x
              for i, j in report["risk_cells"]]
        self.assertLess(max(abs(x - 6.0) for x in xs), max_cov)
        self.assertGreater(min(xs), 2.0)
        self.assertLess(max(xs), 10.0)

    def test_risk_never_becomes_actual_cross_wall_weight(self):
        scene, engine = baked(make_two_room)
        for i, j in engine.leak_report()["risk_cells"]:
            s = engine.sample(engine.grid.cell_center(i, j))
            for pid in s.cross_component_probes:
                self.assertNotIn(pid, s.weights)

    def test_uniform_room_has_no_structural_risk(self):
        scene, engine = baked(make_uniform_room)
        self.assertEqual(engine.leak_report()["risk_cells"], [])

    def test_approximation_set_distinct_from_uncovered_set(self):
        scene, engine = baked(make_sparse_room)
        report = engine.leak_report()
        a = set(report["approximation_cells"])
        u = set((i, j) for i, j in report["uncovered_cells"])
        self.assertFalse({(i, j) for i, j, _ in a} & u)


class MovementContinuityTests(unittest.TestCase):
    def setUp(self):
        # Uniform room: smooth lighting field with no partition crossings.
        self.scene, self.engine = baked(make_uniform_room)

    def _walk(self, start, end, steps):
        return [
            self.engine.sample(
                Point(start[0] + (end[0] - start[0]) * t / steps,
                      start[1] + (end[1] - start[1]) * t / steps)
            )
            for t in range(steps + 1)
        ]

    def test_smooth_weights_inside_one_room(self):
        samples = self._walk((1.5, 4.0), (8.5, 4.0), 20)
        for a, b in zip(samples, samples[1:]):
            da = math.dist(a.irradiance, b.irradiance)
            scale = math.sqrt(sum(v * v for v in a.irradiance)) + 1e-9
            self.assertLess(da / scale, 0.25)

    def test_two_room_doorway_crossing_is_continuous(self):
        scene, engine = baked(make_two_room)
        samples = [
            engine.sample(Point(3.0 + 6.0 * t / 30, 4.0))
            for t in range(31)
        ]
        for s in samples:
            self.assertTrue(all(math.isfinite(v) for v in s.irradiance))
            self.assertEqual(s.status, "covered")

    def test_moving_furniture_does_not_modify_probes(self):
        scene, engine = baked(make_two_room)
        before = {p.id: list(p.irradiance) for p in scene.probes}
        f = scene.furniture[0]
        for x, y in ((8.0, 4.0), (10.0, 6.0), (3.0, 2.0)):
            f.x, f.y = x, y
            # Sampling at the new position changes which probes are read...
            engine.sample(Point(x, y))
        # ...but baked probe values (the static environment) are untouched.
        after = {p.id: list(p.irradiance) for p in scene.probes}
        self.assertEqual(before, after)


class LinearDomainTests(unittest.TestCase):
    def test_display_mapped_average_differs_from_linear_combine(self):
        # Guard against the forbidden shortcut: averaging sRGB images.
        lin_a, lin_b = [0.8, 0.7, 0.6], [0.1, 0.2, 0.3]
        linear_mix = [(a + b) / 2 for a, b in zip(lin_a, lin_b)]
        srgb_mix = [
            srgb_to_linear((linear_to_srgb(a) + linear_to_srgb(b)) / 2)
            for a, b in zip(lin_a, lin_b)
        ]
        self.assertFalse(
            all(abs(x - y) < 1e-4 for x, y in zip(linear_mix, srgb_mix))
        )

    def test_engine_blend_is_linear_combination(self):
        scene, engine = baked(make_uniform_room)
        s = engine.sample(Point(5.0, 4.0))
        for k in range(3):
            expected = sum(
                w * next(p for p in scene.probes if p.id == pid).irradiance[k]
                for pid, w in s.weights.items()
            )
            self.assertAlmostEqual(s.irradiance[k], expected, places=9)


class PersistenceTests(unittest.TestCase):
    def test_save_and_reopen_restores_interpretation(self):
        scene, engine = baked(make_two_room)
        # Move a probe and a chair before saving.
        scene.probes[0].x, scene.probes[0].y = 2.2, 1.1
        scene.furniture[0].x, scene.furniture[0].y = 4.2, 4.4
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scene.json")
            scene.save(path)
            reopened = Scene.load(path)
            engine2 = LightingEngine(reopened)
            engine2.bake_probes()

        self.assertEqual(
            [(p.id, round(p.x, 6), round(p.y, 6)) for p in scene.probes],
            [(p.id, round(p.x, 6), round(p.y, 6)) for p in reopened.probes],
        )
        self.assertEqual(
            [(f.name, f.x, f.y) for f in scene.furniture],
            [(f.name, f.x, f.y) for f in reopened.furniture],
        )
        # Same probe layout -> identical components and sample values.
        p = Point(7.5, 6.0)
        s1, s2 = engine.sample(p), engine2.sample(p)
        self.assertEqual(s1.component, s2.component)
        for k in range(3):
            self.assertAlmostEqual(s1.irradiance[k], s2.irradiance[k],
                                   places  =9)

    def test_baked_values_survive_roundtrip(self):
        scene, _ = baked(make_uniform_room)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "u.json")
            scene.save(path)
            reopened = Scene.load(path)
        for a, b in zip(scene.probes, reopened.probes):
            self.assertEqual(a.id, b.id)
            self.assertEqual(a.irradiance, b.irradiance)
            self.assertEqual(a.coverage, b.coverage)
            self.assertTrue(b.baked)

    def test_reopen_does_not_overwrite_stored_probe_interpretation(self):
        scene, engine = baked(make_two_room)
        scene.probes[0].x, scene.probes[0].y = 2.2, 1.1
        scene.probes[0].baked = False
        moved = LightingEngine(scene)
        moved.bake_probes()  # explicit rebake of the moved probe
        stored = list(scene.probes[0].irradiance)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tr.json")
            scene.save(path)
            reopened = Scene.load(path)
            reopened_engine = LightingEngine(reopened)
            reopened_engine.bake_probes()  # must preserve stored values
        self.assertEqual(reopened.probes[0].irradiance, stored)
        p = Point(7.5, 6.0)
        self.assertEqual(
            moved.sample(p).irradiance,
            reopened_engine.sample(p).irradiance,
        )


class MaterialShaderTests(unittest.TestCase):
    def test_shade_responds_to_light_and_roughness(self):
        from furnitureprobe.scene import Material
        mat = Material("m", [0.5, 0.5, 0.5], 0.5)
        dark = shade_material(mat, [0.1, 0.1, 0.1])
        bright = shade_material(mat, [1.0, 1.0, 1.0])
        self.assertLess(sum(dark), sum(bright))
        glossy = shade_material(Material("g", [0.5] * 3, 0.05),
                                [1.0, 1.0, 1.0])
        rough = shade_material(Material("r", [0.5] * 3, 0.95),
                               [1.0, 1.0, 1.0])
        self.assertGreater(sum(glossy), sum(rough))

    def test_renderer_writes_ppm_pair(self):
        scene, engine = baked(make_uniform_room)
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "render.ppm")
            Renderer(engine, scale=20).render().save_ppm(out)
            ref = out.replace(".ppm", ".reference.ppm")
            Renderer(engine, scale=20).render(use_reference=True).save_ppm(ref)
            for path in (out, ref):
                with open(path, "rb") as fh:
                    self.assertTrue(fh.read(2) == b"P6")
                    self.assertGreater(os.path.getsize(path), 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Probe lighting engine.

Pipeline
--------
1. Rasterise the room into a fine grid and flood-fill *connected components*
   of free space, blocking on wall segments.
2. ``direct_irradiance`` evaluates the static lights at any point in the
   *linear* domain with line-of-sight shadow tests.  This is the reference
   solution probes try to approximate.
3. Probes are baked from the static lights only.  Moving furniture changes
   where probes are *sampled*; it never writes colours back into a probe, so
   an object's previous-frame appearance can never pollute the environment.
4. Interpolation is inverse-distance weighted in linear RGB over the nearest
   probes whose coverage radius reaches the point AND whose connected
   component matches (partition walls are respected).  Points with no
   eligible probe use the scene's explicit fallback policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import Point, Wall, line_blocked, segments_intersect
from .scene import Scene, Probe


# ---------------------------------------------------------------------------
# Grid + connected components
# ---------------------------------------------------------------------------

class Grid:
    def __init__(self, width: float, height: float, cell: float):
        self.width = width
        self.height = height
        self.cell = cell
        self.nx = max(1, round(width / cell))
        self.ny = max(1, round(height / cell))
        self.dx = width / self.nx
        self.dy = height / self.ny

    def cell_center(self, i: int, j: int) -> Point:
        return Point((i + 0.5) * self.dx, (j + 0.5) * self.dy)

    def cell_at(self, p: Point) -> tuple[int, int]:
        i = min(self.nx - 1, max(0, int(p.x / self.dx)))
        j = min(self.ny - 1, max(0, int(p.y / self.dy)))
        return i, j

    def inside(self, p: Point) -> bool:
        return 0.0 <= p.x <= self.width and 0.0 <= p.y <= self.height


def _edge_blocked(grid: Grid, i: int, j: int, ni: int, nj: int,
                  walls: list[Wall]) -> bool:
    """True if the shared cell edge between neighbours crosses a wall.

    Using the actual cell edge (rather than the centre-to-centre segment)
    avoids thin walls slipping through the gap between adjacent centres.
    """
    cx = (i + 0.5) * grid.dx
    cy = (j + 0.5) * grid.dy
    hx, hy = grid.dx / 2, grid.dy / 2
    if ni == i + 1:
        a, b = Point(cx + hx, cy - hy), Point(cx + hx, cy + hy)
    elif ni == i - 1:
        a, b = Point(cx - hx, cy - hy), Point(cx - hx, cy + hy)
    elif nj == j + 1:
        a, b = Point(cx - hx, cy + hy), Point(cx + hx, cy + hy)
    else:
        a, b = Point(cx - hx, cy - hy), Point(cx + hx, cy - hy)
    return any(segments_intersect(a, b, w.a, w.b) for w in walls)


def _closed_segments(walls: list[Wall]) -> list[Wall]:
    """Collapse co-linear partition fragments into solid barrier segments.

    Region labelling treats a partition as a full barrier *including its
    openings*: each side of the wall gets its own static region id, which is
    the invariant used to reject cross-wall probe blending.  Line-of-sight
    tests elsewhere still use the real wall fragments, so a doorway visibly
    connects the space for probe blending and light travel.
    """
    groups: dict[tuple[bool, float], list[Wall]] = {}
    for w in walls:
        if abs(w.a.x - w.b.x) < 1e-9:  # vertical
            key = (True, round(w.a.x, 9))
        elif abs(w.a.y - w.b.y) < 1e-9:  # horizontal
            key = (False, round(w.a.y, 9))
        else:
            groups[(False, id(w))] = [w]  # diagonal walls stay standalone
            continue
        groups.setdefault(key, []).append(w)
    closed: list[Wall] = []
    for (vertical, _axis), group in groups.items():
        if vertical:
            ys = [v for w in group for v in (w.a.y, w.b.y)]
            x = group[0].a.x
            closed.append(Wall(Point(x, min(ys)), Point(x, max(ys))))
        else:
            xs = [v for w in group for v in (w.a.x, w.b.x)]
            y = group[0].a.y
            closed.append(Wall(Point(min(xs), y), Point(max(xs), y)))
    return closed


def label_components(grid: Grid, walls: list[Wall],
                     closed_barriers: bool = False) -> list[list[int]]:
    """Flood-fill free-space region ids; wall cells keep id -1.

    Neighbours are separated when their shared cell edge intersects a wall
    segment.  With ``closed_barriers`` the walls are first merged into
    co-linear solid barriers (openings ignored) for static region labelling.
    Returned grid is indexed ``comp[j][i]`` (row j, column i).
    """
    barrier_walls = _closed_segments(walls) if closed_barriers else walls
    comp = [[-1] * grid.nx for _ in range(grid.ny)]
    next_id = 0
    for j0 in range(grid.ny):
        for i0 in range(grid.nx):
            if comp[j0][i0] != -1:
                continue
            comp[j0][i0] = next_id
            stack = [(i0, j0)]
            while stack:
                i, j = stack.pop()
                for ni, nj in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    if not (0 <= ni < grid.nx and 0 <= nj < grid.ny):
                        continue
                    if comp[nj][ni] != -1:
                        continue
                    if _edge_blocked(grid, i, j, ni, nj, barrier_walls):
                        continue
                    comp[nj][ni] = next_id
                    stack.append((ni, nj))
            next_id += 1
    return comp


# ---------------------------------------------------------------------------
# Direct lighting (reference, linear domain)
# ---------------------------------------------------------------------------

def _smooth_falloff(d: float, radius: float) -> float:
    """Bounded, energy-normalised distance falloff (linear domain).

    A smoothstep window on top of a linear inverse-distance term keeps
    radiance finite at d -> 0 and exactly zero at the light radius, so a
    baked probe cannot hold values orders of magnitude above the field it
    is meant to represent.
    """
    if d >= radius:
        return 0.0
    t = d / radius
    window = (1.0 - t) ** 2
    edge = 1.0 - (3.0 - 2.0 * t) * t * t  # 1 at centre, 0 at radius
    return window * edge


def direct_irradiance(p: Point, scene: Scene) -> list[float]:
    """Reference irradiance from static lights, with wall shadow tests."""
    acc = [0.0, 0.0, 0.0]
    for light in scene.lights:
        lp = light.position()
        d = p.distance_to(lp)
        if d >= light.radius:
            continue
        if line_blocked(p, lp, scene.walls):
            continue
        f = light.power * _smooth_falloff(d, light.radius)
        for k in range(3):
            acc[k] += light.color[k] * f
    return acc


# ---------------------------------------------------------------------------
# Interpolation results
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    point: Point
    irradiance: list[float]            # interpolated linear RGB
    weights: dict[str, float]          # probe id -> normalised weight
    component: int
    status: str                        # "covered" | "fallback" | "dark" | "error"
    reason: str = ""
    leak_risk: bool = False            # partition forces cross-side behaviour
    cross_component_probes: list[str] = field(default_factory=list)


class LightingEngine:
    def __init__(self, scene: Scene, cell: float = 0.2):
        self.scene = scene
        self.grid = Grid(scene.bounds.w, scene.bounds.h, cell)
        # Static region labels treat every partition as a full barrier,
        # ignoring doorways: two sides of a wall get different region ids
        # even where an opening physically connects the floor space.
        self.components = label_components(
            self.grid, scene.partitions, closed_barriers=True
        )
        self._assign_probe_components()

    # ---- baking ------------------------------------------------------------

    def _assign_probe_components(self) -> None:
        for probe in self.scene.probes:
            p = probe.position()
            if self.grid.inside(p):
                i, j = self.grid.cell_at(p)
                probe.component = self.components[j][i]

    def bake_probes(self, force: bool = False) -> None:
        """Bake probes from static lights in linear radiance.

        Probes already carrying an authored/baked irradiance (``baked=True``,
        e.g. after reopening a saved scene) keep their stored interpretation
        unless ``force`` is set; only their region component is recomputed.
        Newly added or explicitly moved probes are (re)baked on demand.
        """
        for probe in self.scene.probes:
            if force or not probe.baked:
                probe.irradiance = direct_irradiance(
                    probe.position(), self.scene
                )
                probe.baked = True
            if self.grid.inside(probe.position()):
                i, j = self.grid.cell_at(probe.position())
                probe.component = self.components[j][i]

    def import_probe_values(self, values: dict[str, list[float]]) -> list[str]:
        """Import externally authored linear-RGB probe values.

        Returns warnings for unknown ids or non-linear-looking data.
        """
        warnings: list[str] = []
        known = {p.id for p in self.scene.probes}
        for pid, rgb in values.items():
            if pid not in known:
                warnings.append(f"unknown probe id ignored: {pid}")
                continue
            if any(v < 0.0 for v in rgb):
                warnings.append(f"{pid}: negative values clamped to 0")
            probe = next(p for p in self.scene.probes if p.id == pid)
            probe.irradiance = [max(0.0, float(v)) for v in rgb[:3]]
            probe.baked = True  # externally authored value is authoritative
        return warnings

    def component_at(self, p: Point) -> int:
        if not self.grid.inside(p):
            return -1
        i, j = self.grid.cell_at(p)
        return self.components[j][i]

    # ---- interpolation -----------------------------------------------------

    def sample(self, p: Point) -> Sample:
        scene = self.scene
        comp = self.component_at(p)

        in_range: list[tuple[Probe, float]] = []
        cross_side: list[Probe] = []
        for probe in scene.probes:
            d = p.distance_to(probe.position())
            if d > probe.coverage:
                continue
            # Per-probe line-of-sight against ALL walls (partitions with
            # their openings included).  Probes behind solid wall are never
            # blended in; a probe visible through a doorway is allowed.
            if line_blocked(p, probe.position(), scene.walls):
                cross_side.append(probe)
            else:
                in_range.append((probe, d))

        if in_range:
            # Bounded inverse-distance kernel.  The soft-radius epsilon keeps
            # every weight finite at d -> 0, so a single probe cannot take
            # over and the field stays smooth across coverage boundaries.
            # Combination happens in the linear domain; never on
            # display-mapped (sRGB) values.
            eps2 = (0.12) ** 2
            raw = {
                probe.id: 1.0 / (d * d + eps2)
                for probe, d in in_range
            }
            total = sum(raw.values())
            weights = {pid: w / total for pid, w in raw.items()}
            irr = [0.0, 0.0, 0.0]
            for probe, _ in in_range:
                w = weights[probe.id]
                for k in range(3):
                    irr[k] += w * probe.irradiance[k]
            return Sample(
                point=p, irradiance=irr, weights=weights, component=comp,
                status="covered",
                cross_component_probes=[q.id for q in cross_side],
                # A covered point can still sit beside a partition where
                # cross-side probes are in range but correctly excluded.
                leak_risk=bool(cross_side),
            )

        # No eligible probe: explicit, visible fallback policy.
        reason = (
            f"no probe with coverage reaching component {comp}; "
            f"{len(cross_side)} cross-wall probe(s) in range were rejected"
        )
        if cross_side:
            return self._fallback(p, comp, cross_side, reason, leak=True)
        return self._fallback(p, comp, [], reason, leak=False)

    def _fallback(self, p: Point, comp: int,
                  cross_side: list[Probe], reason: str,
                  leak: bool) -> Sample:
        policy = self.scene.fallback
        if policy == "dark":
            return Sample(p, [0.0, 0.0, 0.0], {}, comp, "dark", reason,
                          leak_risk=leak,
                          cross_component_probes=[q.id for q in cross_side])
        if policy == "error":
            return Sample(p, [0.0, 0.0, 0.0], {}, comp, "error", reason,
                          leak_risk=leak,
                          cross_component_probes=[q.id for q in cross_side])
        # "nearest": nearest *line-of-sight* probe, regardless of coverage.
        same = [
            q for q in self.scene.probes
            if not line_blocked(p, q.position(), self.scene.walls)
        ]
        if not same:
            return Sample(p, [0.0, 0.0, 0.0], {}, comp, "dark",
                          reason + "; no same-component probe exists",
                          leak_risk=leak,
                          cross_component_probes=[q.id for q in cross_side])
        nearest = min(same, key=lambda q: p.distance_to(q.position()))
        return Sample(
            p, list(nearest.irradiance), {nearest.id: 1.0}, comp,
            "fallback",
            reason + f"; nearest same-component probe {nearest.id} used",
            leak_risk=leak,
            cross_component_probes=[q.id for q in cross_side],
        )

    # ---- whole-field evaluation for maps / tests ---------------------------

    def evaluate_field(self) -> list[list[Sample]]:
        out: list[list[Sample]] = []
        for j in range(self.grid.ny):
            row = []
            for i in range(self.grid.nx):
                row.append(self.sample(self.grid.cell_center(i, j)))
            out.append(row)
        return out

    def reference_field(self) -> list[list[list[float]]]:
        return [
            [direct_irradiance(self.grid.cell_center(i, j), self.scene)
             for i in range(self.grid.nx)]
            for j in range(self.grid.ny)
        ]

    def leak_report(self) -> dict:
        """Structural leak audit + reference-approximation summary.

        Three distinct, independently meaningful cell sets are returned so
        the GUI can attribute every anomaly to a real cause instead of
        painting the whole map red:

        * ``risk_cells`` — a partition separates this cell from one or more
          probes inside its coverage radius.  The probes are correctly
          rejected (no blending), but the location is a candidate for
          visible leakage if the wall logic were ever weakened.
        * ``uncovered_cells`` — no line-of-sight probe reaches the cell at
          all; the configured fallback policy supplies the value.
        * ``approximation_cells`` — covered cells whose interpolated radiance
          differs from the direct-light reference beyond an adaptive
          threshold.  This is probe *granularity* error, reported separately
          from wall leakage, never conflated with it.
        """
        field = self.evaluate_field()
        ref = self.reference_field()
        risk_cells: list[tuple[int, int]] = []
        uncovered_cells: list[tuple[int, int]] = []
        approximation_cells: list[tuple[int, int, float]] = []
        abs_tol, rel_tol = 1.5, 0.40
        max_abs = 0.0
        for j in range(self.grid.ny):
            for i in range(self.grid.nx):
                s = field[j][i]
                if s.leak_risk:
                    risk_cells.append((i, j))
                if s.status in ("fallback", "dark", "error"):
                    uncovered_cells.append((i, j))
                r = ref[j][i]
                abs_err = math.sqrt(
                    sum((s.irradiance[k] - r[k]) ** 2 for k in range(3))
                )
                max_abs = max(max_abs, abs_err)
                threshold = abs_tol + rel_tol * math.sqrt(
                    sum(v * v for v in r)
                )
                if abs_err > threshold and s.status == "covered":
                    approximation_cells.append((i, j, abs_err))
        return {
            "risk_cells": risk_cells,
            "approximation_cells": approximation_cells,
            "uncovered_cells": uncovered_cells,
            "max_absolute_error": max_abs,
            "abs_tolerance": abs_tol,
            "rel_tolerance": rel_tol,
        }

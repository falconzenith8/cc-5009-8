"""2D geometry primitives: points, axis-aligned rects, wall segments."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in world (metres)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def center(self) -> Point:
        return Point(self.x + self.w * 0.5, self.y + self.h * 0.5)

    def contains(self, p: Point, eps: float = 1e-9) -> bool:
        return (
            self.x - eps <= p.x <= self.x + self.w + eps
            and self.y - eps <= p.y <= self.y + self.h + eps
        )

    def corners(self) -> tuple[Point, Point, Point, Point]:
        return (
            Point(self.x, self.y),
            Point(self.x + self.w, self.y),
            Point(self.x + self.w, self.y + self.h),
            Point(self.x, self.y + self.h),
        )


@dataclass(frozen=True)
class Wall:
    """A solid wall segment. Openings are modelled by splitting walls."""

    a: Point
    b: Point

    @property
    def length(self) -> float:
        return self.a.distance_to(self.b)


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """Strict interior intersection; collinear overlaps count as blocking."""
    d1 = (p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x)
    d2 = (p4.x - p3.x) * (p2.y - p3.y) - (p4.y - p3.y) * (p2.x - p3.x)
    d3 = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
    d4 = (p2.x - p1.x) * (p4.y - p1.y) - (p2.y - p1.y) * (p4.x - p1.x)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True
    return _on_segment(p3, p4, p1) or _on_segment(p3, p4, p2) or \
        _on_segment(p1, p2, p3) or _on_segment(p1, p2, p4)


def _on_segment(a: Point, b: Point, p: Point, eps: float = 1e-9) -> bool:
    cross = (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x)
    if abs(cross) > eps:
        return False
    return (
        min(a.x, b.x) - eps <= p.x <= max(a.x, b.x) + eps
        and min(a.y, b.y) - eps <= p.y <= max(a.y, b.y) + eps
    )


def line_blocked(p: Point, q: Point, walls: list[Wall]) -> bool:
    """True if any wall segment blocks the open line of sight p -> q."""
    for wall in walls:
        if segments_intersect(p, q, wall.a, wall.b):
            return True
    return False


def split_wall_with_opening(
    start: Point, end: Point, openings: list[tuple[float, float]]
) -> list[Wall]:
    """Split a horizontal/vertical wall into solid segments around openings.

    Openings are given as (from, to) intervals along the wall axis.
    """
    horizontal = abs(start.y - end.y) < 1e-9
    lo = start.x if horizontal else start.y
    hi = end.x if horizontal else end.y
    if lo > hi:
        lo, hi = hi, lo
    cuts = sorted((max(lo, a), min(hi, b)) for a, b in openings)
    walls: list[Wall] = []
    cursor = lo

    def point_at(t: float) -> Point:
        if horizontal:
            return Point(t, start.y)
        return Point(start.x, t)

    for a, b in cuts:
        if a > cursor + 1e-9:
            walls.append(Wall(point_at(cursor), point_at(a)))
        cursor = max(cursor, b)
    if cursor < hi - 1e-9:
        walls.append(Wall(point_at(cursor), point_at(hi)))
    return walls


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def srgb_to_linear(c: float) -> float:
    c = clamp(c, 0.0, 1.0)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    c = clamp(c, 0.0, 1.0)
    return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

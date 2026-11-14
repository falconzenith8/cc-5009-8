"""Scene description: room bounds, walls, lights, zones, furniture, probes.

All light/colour values stored in the scene are *linear* radiance-style
triples.  Display conversion happens only at output time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .geometry import Point, Rect, Wall, split_wall_with_opening


@dataclass
class Light:
    """Point light in linear radiance units (top-down position)."""

    name: str
    x: float
    y: float
    color: list[float]  # linear RGB
    power: float
    radius: float  # falloff cutoff distance

    def position(self) -> Point:
        return Point(self.x, self.y)


@dataclass
class Zone:
    """A named brightness zone (informational region + expected radiance)."""

    name: str
    rect: Rect
    target_irradiance: list[float]


@dataclass
class Material:
    """Linear-space PBR-ish material used by the preview shader."""

    name: str
    albedo: list[float]  # linear RGB
    roughness: float
    metalness: float = 0.0


@dataclass
class Furniture:
    """Movable furniture (static environment is never modified by it)."""

    name: str
    kind: str  # "table" | "chair"
    x: float
    y: float
    w: float
    h: float
    rotation: float  # degrees
    material: Material

    def rect(self) -> Rect:
        return Rect(self.x - self.w / 2, self.y - self.h / 2, self.w, self.h)


@dataclass
class Probe:
    """Spatial lighting probe.

    Representation: irradiance triple (linear RGB) baked from the *static*
environment only, plus coverage radius and visibility bookkeeping.
    """

    id: str
    x: float
    y: float
    coverage: float  # metres, influence radius
    irradiance: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    component: int = -1  # connected-space component id, filled at bake time
    baked: bool = False  # whether irradiance is an authored/baked value

    def position(self) -> Point:
        return Point(self.x, self.y)


@dataclass
class Scene:
    name: str
    bounds: Rect
    walls: list[Wall]
    lights: list[Light]
    zones: list[Zone]
    furniture: list[Furniture]
    probes: list[Probe]
    # Internal partition walls (subset of walls). Used for static region
    # labelling; line-of-sight tests always use the full wall list so an
    # opening in a partition still lets probes blend through it.
    partitions: list[Wall] = field(default_factory=list)
    # Interpolation policy for points with no visible in-range probe.
    fallback: str = "nearest"  # "nearest" | "dark" | "error"

    # ---- (de)serialisation -------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "name": self.name,
            "bounds": asdict(self.bounds),
            "fallback": self.fallback,
            "walls": [
                {"ax": w.a.x, "ay": w.a.y, "bx": w.b.x, "by": w.b.y}
                for w in self.walls
            ],
            "partitions": [
                {"ax": w.a.x, "ay": w.a.y, "bx": w.b.x, "by": w.b.y}
                for w in self.partitions
            ],
            "lights": [asdict(l) for l in self.lights],
            "zones": [
                {"name": z.name, "rect": asdict(z.rect),
                 "target_irradiance": z.target_irradiance}
                for z in self.zones
            ],
            "furniture": [
                {
                    "name": f.name, "kind": f.kind, "x": f.x, "y": f.y,
                    "w": f.w, "h": f.h, "rotation": f.rotation,
                    "material": asdict(f.material),
                }
                for f in self.furniture
            ],
            "probes": [asdict(p) for p in self.probes],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        b = data["bounds"]
        walls = [
            Wall(Point(w["ax"], w["ay"]), Point(w["bx"], w["by"]))
            for w in data["walls"]
        ]
        partitions = [
            Wall(Point(w["ax"], w["ay"]), Point(w["bx"], w["by"]))
            for w in data.get("partitions", [])
        ]
        lights = [Light(**l) for l in data["lights"]]
        zones = [
            Zone(z["name"], Rect(**z["rect"]), z["target_irradiance"])
            for z in data["zones"]
        ]
        furniture = [
            Furniture(
                name=f["name"], kind=f["kind"], x=f["x"], y=f["y"],
                w=f["w"], h=f["h"], rotation=f["rotation"],
                material=Material(**f["material"]),
            )
            for f in data["furniture"]
        ]
        probes = [Probe(**p) for p in data["probes"]]
        return cls(
            name=data["name"],
            bounds=Rect(b["x"], b["y"], b["w"], b["h"]),
            walls=walls,
            lights=lights,
            zones=zones,
            furniture=furniture,
            probes=probes,
            partitions=partitions,
            fallback=data.get("fallback", "nearest"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Scene":
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Preset construction
# ---------------------------------------------------------------------------

WOOD = Material("oak", [0.36, 0.22, 0.11], roughness=0.55)
FABRIC = Material("blue fabric", [0.05, 0.10, 0.32], roughness=0.85)
DARK_METAL = Material("dark metal", [0.18, 0.18, 0.20], roughness=0.3,
                      metalness=0.8)


def _outer_walls(bounds: Rect, openings: list[tuple[str, float, float]]
                 ) -> list[Wall]:
    """Build perimeter walls; openings reference a side and axis interval."""
    x0, y0, x1, y1 = bounds.x, bounds.y, bounds.x + bounds.w, bounds.y + bounds.h
    sides = {
        "top": (Point(x0, y0), Point(x1, y0)),
        "bottom": (Point(x0, y1), Point(x1, y1)),
        "left": (Point(x0, y0), Point(x0, y1)),
        "right": (Point(x1, y0), Point(x1, y1)),
    }
    by_side: dict[str, list[tuple[float, float]]] = {}
    for side, a, b in openings:
        by_side.setdefault(side, []).append((a, b))
    walls: list[Wall] = []
    for side, (a, b) in sides.items():
        walls.extend(split_wall_with_opening(a, b, by_side.get(side, [])))
    return walls


def _partition(x: float, y0: float, y1: float,
               openings: list[tuple[float, float]]) -> list[Wall]:
    return split_wall_with_opening(Point(x, y0), Point(x, y1), openings)


def make_uniform_room() -> Scene:
    """Evenly lit single room with a regular probe grid."""
    bounds = Rect(0, 0, 10, 8)
    walls = _outer_walls(bounds, [])
    lights = [
        Light("key1", 2.5, 2.0, [1.0, 0.97, 0.9], 26.0, 7.0),
        Light("key2", 7.5, 2.0, [1.0, 0.97, 0.9], 26.0, 7.0),
        Light("key3", 2.5, 6.0, [1.0, 0.97, 0.9], 26.0, 7.0),
        Light("key4", 7.5, 6.0, [1.0, 0.97, 0.9], 26.0, 7.0),
    ]
    spots = [
        (1.0, 1.0), (3.0, 1.0), (5.0, 1.0), (7.0, 1.0), (9.0, 1.0),
        (1.0, 2.75), (3.0, 2.75), (5.0, 2.75), (7.0, 2.75), (9.0, 2.75),
        (1.0, 4.0), (3.0, 4.0), (5.0, 4.0), (7.0, 4.0), (9.0, 4.0),
        (1.0, 5.25), (3.0, 5.25), (5.0, 5.25), (7.0, 5.25), (9.0, 5.25),
        (1.0, 7.0), (3.0, 7.0), (5.0, 7.0), (7.0, 7.0), (9.0, 7.0),
    ]
    probes = [
        Probe(f"P{i}", x, y, coverage=2.0)
        for i, (x, y) in enumerate(spots)
    ]
    furniture = [
        Furniture("table-1", "table", 5.0, 4.0, 1.8, 1.0, 0.0, WOOD),
        Furniture("chair-a", "chair", 5.0, 2.9, 0.5, 0.5, 0.0, FABRIC),
        Furniture("chair-b", "chair", 5.0, 5.1, 0.5, 0.5, 180.0, FABRIC),
    ]
    zones = [Zone("whole room", bounds, [0.9, 0.88, 0.8])]
    return Scene("uniform-room", bounds, walls, lights, zones, furniture,
                 probes)


def make_two_room() -> Scene:
    """Bright room + dim room separated by a partition with one doorway."""
    bounds = Rect(0, 0, 12, 8)
    walls = _outer_walls(bounds, [])
    # Partition at x=6 with a 1.4 m doorway centred on y=4.
    partitions = _partition(6.0, 0.0, 8.0, [(3.3, 4.7)])
    walls += partitions
    lights = [
        Light("bright-a", 1.8, 1.6, [1.0, 0.96, 0.88], 28.0, 5.5),
        Light("bright-b", 1.8, 6.4, [1.0, 0.96, 0.88], 28.0, 5.5),
        Light("bright-c", 4.0, 4.0, [1.0, 0.96, 0.88], 16.0, 3.5),
        Light("dim-a", 9.5, 4.0, [0.25, 0.3, 0.55], 6.0, 5.5),
    ]
    probes = [
        Probe("B1", 1.5, 1.5, 3.0),
        Probe("B2", 4.5, 1.5, 3.0),
        Probe("B3", 1.5, 6.5, 3.0),
        Probe("B4", 4.5, 6.5, 3.0),
        Probe("B5", 4.6, 4.0, 2.6),   # bright side of the doorway
        Probe("D0", 7.0, 4.0, 2.6),   # dim side, catches light through opening
        Probe("D1", 7.5, 1.5, 3.0),
        Probe("D2", 10.5, 1.5, 3.0),
        Probe("D3", 7.5, 6.5, 3.0),
        Probe("D4", 10.5, 6.5, 3.0),
    ]
    furniture = [
        Furniture("table-1", "table", 3.0, 4.0, 1.8, 1.0, 0.0, WOOD),
        Furniture("chair-a", "chair", 3.0, 2.9, 0.5, 0.5, 0.0, FABRIC),
        Furniture("chair-b", "chair", 3.0, 5.1, 0.5, 0.5, 180.0, FABRIC),
        Furniture("lamp-table", "table", 9.5, 4.0, 0.8, 0.8, 0.0,
                  DARK_METAL),
    ]
    zones = [
        Zone("bright room", Rect(0, 0, 6, 8), [1.0, 0.95, 0.85]),
        Zone("dim room", Rect(6, 0, 6, 8), [0.12, 0.15, 0.3]),
    ]
    return Scene("two-room", bounds, walls, lights, zones, furniture,
                 probes, partitions=partitions)


def make_sparse_room() -> Scene:
    """L-shaped room with deliberately sparse probes (uncovered corner)."""
    bounds = Rect(0, 0, 10, 10)
    walls = _outer_walls(bounds, [])
    # Internal partition forming an L: blocks off the top-right quadrant
    # except for a gap, leaving an alcove with no probe.
    partitions = _partition(6.0, 0.0, 6.0, [(0.0, 1.2)])
    partitions += split_wall_with_opening(
        Point(6.0, 6.0), Point(10.0, 6.0), [(8.6, 10.0)]
    )
    walls += partitions
    lights = [
        Light("main-a", 2.5, 3.0, [1.0, 0.95, 0.85], 26.0, 7.0),
        Light("main-b", 2.5, 8.0, [1.0, 0.95, 0.85], 22.0, 6.0),
    ]
    probes = [
        Probe("S1", 2.0, 2.5, 3.0),
        Probe("S2", 2.0, 7.5, 3.0),
        Probe("S3", 8.0, 8.0, 3.2),
        # Alcove (x in 6..10, y in 0..6) intentionally has no probe.
    ]
    furniture = [
        Furniture("table-1", "table", 3.0, 5.0, 1.6, 1.0, 0.0, WOOD),
        Furniture("chair-a", "chair", 3.0, 3.9, 0.5, 0.5, 0.0, FABRIC),
    ]
    zones = [
        Zone("main floor", Rect(0, 0, 6, 10), [0.8, 0.76, 0.68]),
        Zone("alcove (sparse)", Rect(6, 0, 4, 6), [0.05, 0.05, 0.06]),
    ]
    return Scene("sparse-room", bounds, walls, lights, zones, furniture,
                 probes, partitions=partitions, fallback="nearest")


PRESETS = {
    "uniform": make_uniform_room,
    "two-room": make_two_room,
    "sparse": make_sparse_room,
}

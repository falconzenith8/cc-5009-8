"""Linear-domain material shading and headless software rendering.

The rasteriser is deliberately small (top-down polygons) so the same math
that drives the web GUI can be verified off-screen via the CLI without a
display server.  All shading happens on linear radiance; sRGB encoding and
tone mapping are applied once, at the very end.
"""

from __future__ import annotations

import math
from pathlib import Path

from .geometry import Point, linear_to_srgb, clamp
from .lighting import LightingEngine, direct_irradiance
from .scene import Furniture, Material


EXPOSURE = 0.18  # maps typical baked irradiance to a mid-tone


def aces_tone_map(x: float) -> float:
    # Simple ACES approximation, applied per linear channel.
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def shade_material(mat: Material, irradiance: list[float],
                   specular_alignment: float = 0.85) -> list[float]:
    """Lambert diffuse + roughness-controlled Blinn-Phong specular.

    ``irradiance`` is linear RGB arriving at the surface (probe sample or
    direct reference).  ``specular_alignment`` is the clamped dot product of
    the surface normal with the half vector for the preview's fixed view and
    key-light direction.  Specular energy is derived from the same linear
    irradiance and scaled by (1 - roughness); metals tint it with albedo.
    """
    linear = [0.0, 0.0, 0.0]
    diffuse_energy = 1.0 - mat.metalness
    for k in range(3):
        linear[k] = irradiance[k] * mat.albedo[k] * diffuse_energy / math.pi
    # Specular strength grows monotonically as roughness shrinks.  The
    # alignment factor models the fixed preview view/light direction.
    spec = (1.0 - mat.roughness) ** 2 * specular_alignment
    lum = sum(irradiance) / 3.0
    for k in range(3):
        tint = mat.albedo[k] if mat.metalness else 1.0
        linear[k] += lum * spec * tint
    return linear


def _rotated_rect_corners(f: Furniture) -> list[Point]:
    a = math.radians(f.rotation)
    ca, sa = math.cos(a), math.sin(a)
    corners = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx, ly = sx * f.w / 2, sy * f.h / 2
        corners.append(
            Point(f.x + lx * ca - ly * sa, f.y + lx * sa + ly * ca)
        )
    return corners


class Image:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.px = bytearray(w * h * 3)

    def set(self, x: int, y: int, linear: list[float]) -> None:
        if not (0 <= x < self.w and 0 <= y < self.h):
            return
        idx = (y * self.w + x) * 3
        for k in range(3):
            self.px[idx + k] = int(
                round(255 * linear_to_srgb(aces_tone_map(linear[k] * EXPOSURE)))
            )

    def save_ppm(self, path: str | Path) -> None:
        header = f"P6\n{self.w} {self.h}\n255\n".encode()
        Path(path).write_bytes(header + bytes(self.px))


class Renderer:
    def __init__(self, engine: LightingEngine, scale: int = 60):
        self.engine = engine
        self.scene = engine.scene
        self.scale = scale  # pixels per metre
        self.w = round(self.scene.bounds.w * scale)
        self.h = round(self.scene.bounds.h * scale)

    def _world_to_pixel(self, p: Point) -> tuple[int, int]:
        return (round(p.x * self.scale), round(p.y * self.scale))

    def render(self, use_reference: bool = False,
               highlight: Point | None = None) -> Image:
        """Render the room; furniture is shaded from probe samples.

        With ``use_reference`` the background uses direct light evaluation
        instead of probe interpolation, giving the reference comparison.
        """
        img = Image(self.w, self.h)
        scene = self.scene
        for py in range(self.h):
            wy = (py + 0.5) / self.scale
            for px in range(self.w):
                wx = (px + 0.5) / self.scale
                p = Point(wx, wy)
                if use_reference:
                    irr = direct_irradiance(p, scene)
                else:
                    irr = self.engine.sample(p).irradiance
                # Floor material: neutral 0.5 albedo Lambertian.
                img.set(px, py, [v * 0.5 / math.pi for v in irr])

        for f in scene.furniture:
            self._draw_furniture(img, f, use_reference)
        if highlight is not None:
            self._draw_marker(img, highlight)
        return img

    def _draw_furniture(self, img: Image, f: Furniture,
                        use_reference: bool) -> None:
        if use_reference:
            irr = direct_irradiance(Point(f.x, f.y), self.scene)
        else:
            irr = self.engine.sample(Point(f.x, f.y)).irradiance
        color = shade_material(f.material, irr)
        edge = [min(1.0, c + 0.02) for c in color]
        self._fill_polygon(img, _rotated_rect_corners(f), color, edge)

    def _fill_polygon(self, img: Image, corners: list[Point],
                      fill: list[float], edge: list[float]) -> None:
        ys = [p.y for p in corners]
        ymin, ymax = max(0.0, min(ys)), min(img.h / self.scale, max(ys))
        for py in range(int(ymin * self.scale), int(math.ceil(ymax * self.scale))):
            wy = (py + 0.5) / self.scale
            xs: list[float] = []
            for k in range(len(corners)):
                a, b = corners[k], corners[(k + 1) % len(corners)]
                if (a.y <= wy < b.y) or (b.y <= wy < a.y):
                    t = (wy - a.y) / (b.y - a.y)
                    xs.append(a.x + t * (b.x - a.x))
            xs.sort()
            for lo, hi in zip(xs[0::2], xs[1::2]):
                x0 = max(0, int(lo * self.scale))
                x1 = min(img.w, int(math.ceil(hi * self.scale)))
                for px in range(x0, x1):
                    img.set(px, py, fill)
        for k in range(len(corners)):
            self._draw_line(img, corners[k], corners[(k + 1) % len(corners)],
                            edge)

    def _draw_line(self, img: Image, a: Point, b: Point,
                   color: list[float]) -> None:
        x0, y0 = self._world_to_pixel(a)
        x1, y1 = self._world_to_pixel(b)
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for t in range(steps + 1):
            u = t / steps
            img.set(round(x0 + (x1 - x0) * u), round(y0 + (y1 - y0) * u),
                    color)

    def _draw_marker(self, img: Image, p: Point) -> None:
        cx, cy = self._world_to_pixel(p)
        marker = [2.0, 2.0, 0.4]
        for r in range(6):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    img.set(cx + dx, cy + dy, marker)
                    img.set(cx + dy, cy + dx, marker)

    # ---- material preview window frame -------------------------------------

    def material_preview(self, f: Furniture, use_reference: bool = False
                         ) -> dict:
        """Data for the material preview window (GUI renders the swatch).

        Returns the probe sample, weights, shaded colour and, for comparison,
        the direct-light reference shade at the furniture position.
        """
        p = Point(f.x, f.y)
        sample = self.engine.sample(p)
        ref_irr = direct_irradiance(p, self.scene)
        return {
            "furniture": f.name,
            "material": f.material.name,
            "position": [f.x, f.y],
            "sample_status": sample.status,
            "sample_reason": sample.reason,
            "weights": sample.weights,
            "probe_irradiance": sample.irradiance,
            "reference_irradiance": ref_irr,
            "shaded_linear": shade_material(f.material, sample.irradiance),
            "reference_shaded_linear": shade_material(f.material, ref_irr),
            "leak_risk": sample.leak_risk,
            "rejected_probes": sample.cross_component_probes,
        }

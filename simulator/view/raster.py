"""A software renderer: flat-shaded polygons, painter's algorithm, Pillow.

There is no GPU library available and none is being installed, so the 3D view is
drawn with ``ImageDraw.polygon``. That sounds slow and is not: Pillow fills
polygons in C, so what costs time is the *Python* call per face. Two things keep
the number of calls down. Faces are convex n-gons rather than triangles, so a
cube costs six calls and a cylinder cap costs one (:mod:`mesh`), and everything
before the draw loop -- transform, back-face cull, near clip, projection, area
cull, shading, depth sort -- is done with numpy over the whole mesh at once.

The trade for that speed is no depth buffer. Faces are sorted back to front and
drawn over each other, so geometry must be mostly convex for the result to be
correct -- which a stylised engine cutaway is. Back faces are culled by the sign
of the face normal against the view direction, using the winding convention from
:mod:`mesh`.

Two details of the draw loop are worth knowing before changing it. Pillow will
accept a numpy array as a coordinate list and then silently draw almost nothing,
so the coordinates are flattened to a plain Python list once per frame. And it
accepts a packed ``int`` colour for RGB images in little-endian byte order, which
avoids building a tuple per face.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .mesh import Mesh


def _rot_x(a: float) -> np.ndarray:
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = float(np.cos(a)), float(np.sin(a))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@dataclass
class Camera:
    """Orbits a target point. Angles are radians, distance is in metres."""

    target: np.ndarray
    distance: float = 1.1
    azimuth: float = 0.9
    elevation: float = 0.35
    focal: float = 900.0

    def rotation(self) -> np.ndarray:
        return _rot_y(self.azimuth) @ _rot_x(self.elevation)

    def eye(self) -> np.ndarray:
        """Camera position in world space."""
        return self.target + self.rotation().T @ np.array([0.0, 0.0, self.distance])

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        self.azimuth += d_azimuth
        # stop just short of the poles: at exactly +-90 degrees the up vector and
        # the view direction become parallel and the frame flips
        self.elevation = float(np.clip(self.elevation + d_elevation, -1.45, 1.45))

    def zoom(self, factor: float, min_distance: float = 0.25, max_distance: float = 8.0) -> None:
        self.distance = float(np.clip(self.distance * factor, min_distance, max_distance))


def frame_camera(
    lo: np.ndarray,
    hi: np.ndarray,
    width: int,
    height: int,
    azimuth: float = 0.62,
    elevation: float = 0.40,
    fill: float = 0.82,
    focal_scale: float = 1.35,
    target: np.ndarray | None = None,
) -> Camera:
    """A camera that fits a bounding box in the viewport.

    The distance is derived from the pinhole relation ``size * focal / distance``
    rather than guessed, so the model fills the same fraction of the frame at any
    window size.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    centre = 0.5 * (lo + hi) if target is None else np.asarray(target, dtype=np.float64)
    extent = float(np.max(hi - lo))
    focal = float(focal_scale) * float(width)
    smallest = float(min(width, height))
    distance = extent * focal / max(fill * smallest, 1.0)
    return Camera(
        target=centre, distance=distance, azimuth=azimuth, elevation=elevation, focal=focal
    )


class SoftwareRenderer:
    """Renders :class:`Mesh` objects into a Pillow image."""

    def __init__(
        self,
        width: int,
        height: int,
        background_top=(24, 28, 35),
        background_bottom=(9, 11, 14),
        light=(0.42, 0.80, 0.43),
        ambient: float = 0.30,
        shininess: float = 26.0,
        specular: float = 0.32,
        shadow_color=(12, 13, 16),
        shadow_strength: float = 1.0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.shadow_color = tuple(int(v) for v in shadow_color)
        self.shadow_strength = float(max(shadow_strength, 0.0))
        self.image = Image.new("RGB", (self.width, self.height))
        self.draw = ImageDraw.Draw(self.image)
        self._background = self._make_background(background_top, background_bottom)

        light = np.asarray(light, dtype=np.float64)
        self.light = light / float(np.linalg.norm(light))
        self.ambient = float(ambient)
        self.shininess = float(max(shininess, 1.0))
        self.specular = float(specular)
        self.near = 0.05
        self.min_area = 0.6

        self.stats: dict[str, float] = {}

    def _make_background(self, top, bottom) -> Image.Image:
        """A vertical gradient, built once at full size and reused every frame."""
        top = np.asarray(top, dtype=np.float64).reshape(1, 3)
        bottom = np.asarray(bottom, dtype=np.float64).reshape(1, 3)
        rows = np.linspace(0.0, 1.0, self.height, dtype=np.float64).reshape(self.height, 1)
        column = top * (1.0 - rows) + bottom * rows  # (height, 3)
        strip = np.repeat(column[:, None, :], self.width, axis=1)
        return Image.fromarray(np.clip(strip, 0, 255).astype(np.uint8), "RGB")

    # ------------------------------------------------------------------ #
    def render(self, mesh: Mesh, camera: Camera, shadow: bool = True) -> Image.Image:
        """Draw one frame. Returns the image (also kept in ``self.image``)."""
        self.image.paste(self._background)

        starts = mesh.starts
        faces = mesh.face_count
        if mesh.verts.shape[0] == 0 or faces == 0:
            self.stats = {"input": 0.0, "clipped": 0.0, "backfaced": 0.0, "tiny": 0.0, "drawn": 0.0}
            return self.image

        rot = camera.rotation()
        cam = (mesh.verts - camera.target) @ rot.T
        cam[:, 2] += camera.distance

        if shadow:
            self._draw_shadow(mesh, camera, rot)

        z = cam[:, 2]
        index = mesh.index
        first = starts[:-1]
        counts = np.diff(starts)

        # Every frame reports the same keys, whatever stage it stopped at: a
        # partial set made ``stats["drawn"]`` raise for callers that log it.
        stats = {"input": float(faces), "clipped": 0.0, "backfaced": 0.0, "tiny": 0.0, "drawn": 0.0}

        # A face survives if any part of it is in front of the near plane. It
        # used to be dropped whole as soon as *one* of its vertices crossed,
        # which punched holes in the assembly the moment the camera zoomed
        # inside it. Clamping the depth below keeps such a face visible and only
        # distorts the few that actually straddle the plane.
        keep = np.maximum.reduceat(z[index], first) > self.near
        stats["clipped"] = float(faces - int(keep.sum()))
        if not keep.any():
            self.stats = stats
            return self.image

        # camera-space vertex positions gathered per index slot, so the face
        # slices below are plain views instead of further gather operations
        cv = cam[index]
        p0, p1, p2 = cv[first], cv[first + 1], cv[first + 2]
        normals = np.cross(p1 - p0, p2 - p0)
        centroid = np.add.reduceat(cv, first, axis=0) / counts[:, None]
        facing = np.einsum("ij,ij->i", normals, centroid) < 0.0
        stats["backfaced"] = float(faces - int(facing.sum()))
        if not facing.any():
            self.stats = stats
            return self.image

        # projection. The depth is clamped to just in front of the near plane:
        # a vertex behind the camera would otherwise divide by a negative (or
        # vanishing) depth and throw its face to an enormous, meaningless place
        # on screen, which is what made the view flicker when zooming in.
        scale = camera.focal / np.maximum(z, self.near)
        points = np.stack(
            [
                cam[:, 0] * scale + self.width * 0.5,
                self.height * 0.5 - cam[:, 1] * scale,
            ],
            axis=1,
        )
        pv = points[index]
        px, py = pv[:, 0], pv[:, 1]
        x0 = np.minimum.reduceat(px, first)
        x1 = np.maximum.reduceat(px, first)
        y0 = np.minimum.reduceat(py, first)
        y1 = np.maximum.reduceat(py, first)

        # drop sub-pixel faces and anything fully off screen: they cost the same
        # as large ones, and Pillow clips faces that straddle the border anyway
        big = (
            ((x1 - x0) * (y1 - y0) > 2.0 * self.min_area)
            & (x1 > 0.0)
            & (x0 < self.width)
            & (y1 > 0.0)
            & (y0 < self.height)
        )
        stats["tiny"] = float(faces - int(big.sum()))

        keep &= facing & big
        if not keep.any():
            self.stats = stats
            return self.image
        first = first[keep]
        counts = counts[keep]
        normals = normals[keep]
        centroid = centroid[keep]
        colors = mesh.colors[keep]

        # shading, in camera space
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0.0] = 1.0
        unit = normals / lengths
        light = self.light @ rot.T
        lambert = np.abs(unit @ light)
        view = -centroid / np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-9)
        half = light[None, :] + view
        half /= np.maximum(np.linalg.norm(half, axis=1, keepdims=True), 1e-9)
        spec = np.clip(np.einsum("ij,ij->i", unit, half), 0.0, None) ** self.shininess

        shade = self.ambient + (1.0 - self.ambient) * lambert
        rgb = colors * shade[:, None] + self.specular * spec[:, None]
        rgb = np.clip(rgb, 0.0, 1.0) * 255.0
        pcm = rgb.astype(np.int32)

        # painter's algorithm: farthest first
        order = np.argsort(-centroid[:, 2], kind="stable")

        # one flattening pass, then only list slicing and a packed int per face
        flat = pv.ravel().tolist()
        cuts = (first * 2).tolist()
        spans = (counts * 2).tolist()
        fills = (pcm[:, 0] | (pcm[:, 1] << 8) | (pcm[:, 2] << 16)).tolist()

        polygon = self.draw.polygon
        for f in order.tolist():
            at = cuts[f]
            polygon(flat[at : at + spans[f]], fill=fills[f])

        stats["drawn"] = float(len(order))
        self.stats = stats
        return self.image

    # ------------------------------------------------------------------ #
    def _draw_shadow(self, mesh: Mesh, camera: Camera, rot: np.ndarray) -> None:
        if mesh.verts.shape[0] == 0:
            return
        lo, hi = mesh.bounds()
        centre_world = np.array([(lo[0] + hi[0]) * 0.5, lo[1], (lo[2] + hi[2]) * 0.5])
        radius = float(max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])) * 0.55
        cam = (centre_world - camera.target) @ rot.T
        cam[2] += camera.distance
        if cam[2] <= self.near:
            return
        scale = camera.focal / cam[2]
        cx = cam[0] * scale + self.width * 0.5
        cy = -cam[1] * scale + self.height * 0.5
        rx = radius * 1.35 * scale
        ry = rx * 0.30
        if rx < 2.0:
            return
        if self.shadow_strength >= 1.0:
            self.draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=self.shadow_color)
            return
        # A soft contact shadow without an alpha channel or a blur: a stack of
        # concentric ellipses whose colour is interpolated from the background
        # towards the shadow colour. On a light viewport a hard black ellipse
        # looks like a hole, so the strength is dialled down by the caller.
        px = int(min(max(cx, 0), self.width - 1))
        py = int(min(max(cy, 0), self.height - 1))
        bg = self._background.getpixel((px, py))
        steps = 14
        for i in range(steps, 0, -1):
            t = i / steps
            f = ((1.0 - t) ** 1.35) * self.shadow_strength
            fill = tuple(int(round(bg[c] + (self.shadow_color[c] - bg[c]) * f)) for c in range(3))
            self.draw.ellipse([cx - rx * t, cy - ry * t, cx + rx * t, cy + ry * t], fill=fill)

"""Meshes and the primitives the engine model is assembled from.

Everything is built procedurally from the engine specification rather than
loaded from a file, so a 4-cylinder inline and a 60-degree V8 come out of the
same code with the same look.

Winding convention: every face is generated so that
``cross(v1 - v0, v2 - v0)`` points *out of* the solid. The renderer relies on
that to cull back faces without needing a depth buffer, so any new primitive
must follow it.

Faces are convex polygons rather than triangles. Pillow fills a polygon in C,
and what the renderer actually pays for is the Python *call* -- which costs the
same whether the polygon has three vertices or thirty. A cube is therefore six
quads instead of twelve triangles, a cylinder cap is a single polygon instead
of a fan, and the scene reaches the screen in a third of the calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _box_faces() -> tuple[tuple[int, ...], ...]:
    """The six quads of a unit box, wound outwards.

    Corners are indexed by the sign triple (x, y, z) in the order
    ``(-,-,-) (+,-,-) (+,+,-) (-,+,-) (-,-,+) (+,-,+) (+,+,+) (-,+,+)``.
    """
    return (
        (0, 3, 2, 1),  # -z
        (4, 5, 6, 7),  # +z
        (0, 4, 7, 3),  # -x
        (1, 2, 6, 5),  # +x
        (0, 1, 5, 4),  # -y
        (3, 7, 6, 2),  # +y
    )


_BOX_FACES = _box_faces()
_BOX_CORNERS = np.array(
    [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ],
    dtype=np.float64,
)


@dataclass
class Mesh:
    """Convex polygons over one shared vertex pool.

    Face ``f`` uses the vertices ``index[starts[f]:starts[f + 1]]``, listed in
    outward winding order, and is flat-shaded with ``colors[f]``.
    """

    verts: np.ndarray  # (V, 3) float64
    index: np.ndarray  # (I,) int32
    starts: np.ndarray  # (F + 1,) int32
    colors: np.ndarray  # (F, 3) float64 in 0..1

    @staticmethod
    def empty() -> Mesh:
        return Mesh(
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.int32),
            np.zeros((1,), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float64),
        )

    @property
    def face_count(self) -> int:
        return max(int(self.starts.shape[0]) - 1, 0)

    @property
    def triangle_count(self) -> int:
        """Equivalent triangle count: the budget the old renderer worked in."""
        if self.face_count == 0:
            return 0
        return int(np.sum(np.diff(self.starts.astype(np.int64)) - 2))

    def __len__(self) -> int:
        return self.face_count

    def merge(self, other: Mesh) -> Mesh:
        if other is None or other.face_count == 0:
            return self
        if self.face_count == 0:
            return other
        offset = self.verts.shape[0]
        return Mesh(
            np.concatenate([self.verts, other.verts], axis=0),
            np.concatenate([self.index, other.index + offset], axis=0),
            np.concatenate([self.starts[:-1], other.starts + self.starts[-1]], axis=0),
            np.concatenate([self.colors, other.colors], axis=0),
        )

    def transformed(self, rotation: np.ndarray | None = None, offset: np.ndarray | None = None) -> Mesh:
        verts = self.verts
        if rotation is not None:
            verts = verts @ np.asarray(rotation, dtype=np.float64).T
        if offset is not None:
            verts = verts + np.asarray(offset, dtype=np.float64)
        return Mesh(verts, self.index, self.starts, self.colors)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.verts.shape[0] == 0:
            return np.zeros(3), np.zeros(3)
        return self.verts.min(axis=0), self.verts.max(axis=0)


def orthonormal_basis(axis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(u, v, w)`` with ``w`` along ``axis`` and ``cross(u, v) == w``."""
    w = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(w))
    if norm < 1e-12:
        w = np.array([0.0, 1.0, 0.0])
        norm = 1.0
    w = w / norm
    ref = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, w)
    u = u / float(np.linalg.norm(u))
    v = np.cross(w, u)
    return u, v, w


class MeshBuilder:
    """Accumulates primitives into a single :class:`Mesh`."""

    def __init__(self) -> None:
        self._verts: list[np.ndarray] = []
        self._index: list[int] = []
        self._starts: list[int] = [0]
        self._colors: list[tuple[float, float, float]] = []
        self._base = 0

    # ------------------------------------------------------------------ #
    def add_faces(self, verts: np.ndarray, faces, color) -> None:
        """Append vertices and a list of index tuples (local to ``verts``)."""
        verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        rgb = (float(color[0]), float(color[1]), float(color[2]))
        base = self._base
        self._verts.append(verts)
        self._base += verts.shape[0]
        index = self._index
        starts = self._starts
        colors = self._colors
        for face in faces:
            index.extend(int(i) + base for i in face)
            starts.append(len(index))
            colors.append(rgb)

    def add_mesh(self, mesh: Mesh) -> None:
        """Append an already-coloured mesh, keeping its per-face colours."""
        if mesh is None or mesh.face_count == 0:
            return
        base = self._base
        self._verts.append(np.asarray(mesh.verts, dtype=np.float64).reshape(-1, 3))
        self._base += mesh.verts.shape[0]
        self._index.extend((mesh.index.astype(np.int64) + base).tolist())
        shift = self._starts[-1]
        self._starts.extend((mesh.starts[1:].astype(np.int64) + shift).tolist())
        self._colors.extend(map(tuple, mesh.colors.tolist()))

    # ------------------------------------------------------------------ #
    def add_box(self, center, size, color) -> None:
        half = np.asarray(size, dtype=np.float64).reshape(3) * 0.5
        center = np.asarray(center, dtype=np.float64).reshape(3)
        self.add_faces(center + _BOX_CORNERS * half, _BOX_FACES, color)

    def add_prism(
        self,
        base,
        axis,
        length: float,
        radius: float,
        color,
        segments: int = 12,
        cap_start: bool = True,
        cap_end: bool = True,
    ) -> None:
        """A cylinder from ``base`` along ``axis``."""
        segments = max(3, int(segments))
        radii = np.full(segments, float(radius), dtype=np.float64)
        self.add_profile_prism(base, axis, length, radii, color, cap_start, cap_end)

    def add_profile_prism(
        self,
        base,
        axis,
        length: float,
        radii: np.ndarray,
        color,
        cap_start: bool = True,
        cap_end: bool = True,
    ) -> None:
        """A prism whose radius varies around the ring.

        ``radii`` holds one radius per segment, so the same call makes a round
        cylinder and a three-lobed Wankel rotor.
        """
        u, v, w = orthonormal_basis(axis)
        base = np.asarray(base, dtype=np.float64).reshape(3)
        radii = np.asarray(radii, dtype=np.float64).reshape(-1)
        segments = max(3, int(radii.shape[0]))
        angles = np.arange(segments, dtype=np.float64) * (2.0 * np.pi / segments)
        radial = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
        low = base + radial * radii[:, None]
        high = low + w * float(length)

        faces: list[tuple[int, ...]] = []
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((i, j, j + segments, i + segments))
        if cap_end:
            faces.append(tuple(range(segments, 2 * segments)))
        if cap_start:
            faces.append(tuple(range(segments - 1, -1, -1)))
        self.add_faces(np.concatenate([low, high]), faces, color)

    def add_cylinder_between(self, p0, p1, radius: float, color, segments: int = 12) -> None:
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        delta = p1 - p0
        length = float(np.linalg.norm(delta))
        if length < 1e-9:
            return
        self.add_prism(p0, delta / length, length, radius, color, segments=segments)

    def add_ring(self, center, normal, inner: float, outer: float, color, segments: int = 12) -> None:
        """A flat annulus, for example a cylinder deck around a bore."""
        u, v, _ = orthonormal_basis(normal)
        center = np.asarray(center, dtype=np.float64)
        segments = max(3, int(segments))
        angles = np.arange(segments, dtype=np.float64) * (2.0 * np.pi / segments)
        radial = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
        verts = np.concatenate([center + radial * inner, center + radial * outer])
        faces = []
        for i in range(segments):
            j = (i + 1) % segments
            faces.append((i, i + segments, j + segments, j))
        self.add_faces(verts, faces, color)

    def add_disc(self, center, normal, radius: float, color, segments: int = 12) -> None:
        u, v, _ = orthonormal_basis(normal)
        center = np.asarray(center, dtype=np.float64)
        segments = max(3, int(segments))
        angles = np.arange(segments, dtype=np.float64) * (2.0 * np.pi / segments)
        radial = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
        self.add_faces(center + radial * float(radius), (tuple(range(segments)),), color)

    # ------------------------------------------------------------------ #
    def build(self) -> Mesh:
        if not self._verts:
            return Mesh.empty()
        return Mesh(
            np.concatenate(self._verts, axis=0),
            np.asarray(self._index, dtype=np.int32),
            np.asarray(self._starts, dtype=np.int32),
            np.asarray(self._colors, dtype=np.float64).reshape(-1, 3),
        )

"""GPU renderer for the engine assembly.

The old path rasterised polygons in Python with Pillow. That was the right
answer when no GPU library could be installed, but it also set the frame budget:
the cost was one Python call per face, so the whole renderer was built around
keeping that count down (convex n-gons instead of triangles, painter's
algorithm instead of a depth buffer).

This module replaces it with real GPU rendering: the mesh is uploaded once and
drawn with a depth buffer, back-face culling and multisampling, which costs
nothing per face and gives correct occlusion for free. The geometry itself is
still built by :mod:`simulator.view.engine3d` -- that code is the project's
actual asset (it encodes what a flat-six, a Wankel and a gas turbine look like)
and none of it needs to change to be drawn here.

Design notes
------------
* **One static buffer, one dynamic buffer.** Everything in an ``EngineScene``
  is either static (block, head, manifolds) or a rigid mover (piston, rod,
  crank throw, valve). Splitting them means a frame only re-uploads the small
  moving set instead of the whole assembly.
* **Per-vertex normals, flat-shaded.** The source mesh is flat-shaded polygons,
  so every triangle's three corners share its face normal. That is exactly what
  a machined casting looks like, and it keeps the vertex format small.
* **No gamma guessing.** The scene is lit in linear space and converted once at
  the end, so the highlights do not clip into white as they did when the values
  were used as sRGB directly.
* **Convex n-gons are triangulated on upload**, not at draw time: a fan from the
  first vertex is correct for the convex polygons the mesh builder produces.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# shaders
# --------------------------------------------------------------------------- #
VERTEX_SHADER = """#version 330 core
in vec3 in_pos;
in vec3 in_nrm;
in vec3 in_col;

uniform mat4 u_view_proj;
uniform mat4 u_model;

out vec3 v_world;
out vec3 v_normal;
out vec3 v_color;
out vec3 v_local;

void main() {
    vec4 world = u_model * vec4(in_pos, 1.0);
    v_world = world.xyz;
    // the assembly only ever rotates rigidly, so the model matrix is enough
    // to carry the normal; no inverse-transpose is needed
    v_normal = mat3(u_model) * in_nrm;
    v_color = in_col;
    v_local = in_pos;
    gl_Position = u_view_proj * world;
}
"""

FRAGMENT_SHADER = """#version 330 core
in vec3 v_world;
in vec3 v_normal;
in vec3 v_color;
in vec3 v_local;

uniform vec3 u_eye;
uniform vec3 u_light_a;      // key light direction
uniform vec3 u_light_b;      // fill light direction
uniform float u_rim_power;
uniform float u_specular;
uniform float u_shininess;
uniform float u_ghost;       // 0 = solid part, 1 = translucent shell
uniform vec3 u_shell_tint;

out vec4 frag;

vec3 linear_to_srgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;
    vec3 v = normalize(u_eye - v_world);

    float key = max(dot(n, u_light_a), 0.0);
    float fill = max(dot(n, u_light_b), 0.0) * 0.35;
    // a wrapped diffuse term keeps the shadow side readable instead of black,
    // which matters on a dark background
    float wrap = max(dot(n, u_light_a) * 0.5 + 0.5, 0.0) * 0.18;

    vec3 h = normalize(u_light_a + v);
    float spec = pow(max(dot(n, h), 0.0), u_shininess) * u_specular;

    // rim light separates an edge from the background; without it a dark grey
    // casting simply disappears against a dark background
    float rim = pow(1.0 - max(dot(n, v), 0.0), u_rim_power);

    vec3 base = mix(v_color, u_shell_tint, u_ghost * 0.85);
    vec3 color = base * (0.22 + key * 0.92 + fill + wrap);
    color += vec3(1.0) * spec * 0.9;
    color += base * rim * 0.42;

    if (u_ghost > 0.001) {
        color = mix(color, u_shell_tint * (0.35 + key * 0.65), 0.35);
    }

    // a faint height gradient reads as a contact shadow under the assembly
    frag = vec4(linear_to_srgb(color), 1.0);
}
"""


# --------------------------------------------------------------------------- #
# linear algebra
# --------------------------------------------------------------------------- #
def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    norm = np.linalg.norm(forward)
    forward = forward / norm if norm > 1e-9 else np.array([0.0, 0.0, -1.0])
    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        # looking straight along the up axis: pick another reference
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
        rn = np.linalg.norm(right)
        if rn < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
            rn = 1.0
    right /= rn
    true_up = np.cross(right, forward)

    m = np.eye(4, dtype=np.float64)
    m[0, :3] = right
    m[1, :3] = true_up
    m[2, :3] = -forward
    m[0, 3] = -right @ eye
    m[1, 3] = -true_up @ eye
    m[2, 3] = forward @ eye
    return m


def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / np.tan(fov_y * 0.5)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, c, -s, 0.0], [0.0, s, c, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def fit_distance(lo, hi, fov_y: float, aspect: float, fill: float = 0.82) -> float:
    """Distance at which a bounding box fills ``fill`` of the shorter view axis.

    Framing by ``extent * constant`` is wrong across engines: a Wankel is a
    short fat barrel and a V12 is a long slender block, so the same multiplier
    leaves one tiny and the other overflowing. Working from the bounding sphere
    and the actual field of view gives every engine the same apparent size.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    radius = 0.5 * float(np.linalg.norm(hi - lo))
    if radius <= 1e-9:
        return 1.0
    # the vertical half-angle is fov/2; the horizontal one is wider for a wide
    # viewport, so the vertical axis is the binding constraint
    half_v = fov_y * 0.5
    half_h = half_v * max(aspect, 1e-6)
    limiting = min(half_v, half_h)
    return radius / max(np.sin(limiting) * max(fill, 1e-3), 1e-6)


def orbit_eye(centre, distance: float, azimuth: float, elevation: float) -> np.ndarray:
    """Camera position on a sphere around ``centre``."""
    ce = np.cos(elevation)
    return np.asarray(centre, dtype=np.float64) + np.array(
        [
            distance * ce * np.sin(azimuth),
            distance * np.sin(elevation),
            distance * ce * np.cos(azimuth),
        ]
    )


def transform_mesh(mesh, rotation: np.ndarray | None = None, offset=None) -> np.ndarray:
    """Apply a rigid transform to a mesh's vertices (returns a new array)."""
    verts = np.asarray(mesh.verts, dtype=np.float64)
    if rotation is not None:
        verts = verts @ np.asarray(rotation, dtype=np.float64)[:3, :3].T
    if offset is not None:
        verts = verts + np.asarray(offset, dtype=np.float64).reshape(1, 3)
    return verts


# --------------------------------------------------------------------------- #
# mesh packing
# --------------------------------------------------------------------------- #
def triangulate(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a :class:`Mesh` into ``(triangles, face_of_triangle)``.

    Faces are convex n-gons, so a fan from the first corner is a correct
    triangulation and needs no general polygon routine.
    """
    starts = mesh.starts
    counts = np.diff(starts)
    faces = mesh.face_count
    if faces == 0:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    total = int(np.sum(np.maximum(counts - 2, 0)))
    tri = np.empty((total, 3), dtype=np.int64)
    owner = np.empty((total,), dtype=np.int64)
    at = 0
    index = mesh.index
    for f in range(faces):
        count = int(counts[f])
        if count < 3:
            continue
        first = int(starts[f])
        base = int(index[first])
        for i in range(1, count - 1):
            tri[at, 0] = base
            tri[at, 1] = int(index[first + i])
            tri[at, 2] = int(index[first + i + 1])
            owner[at] = f
            at += 1
    return tri[:at], owner[:at]


def pack_triangles(mesh, triangles: np.ndarray, face_of: np.ndarray, colors=None) -> np.ndarray:
    """Build the interleaved ``pos(3) nrm(3) col(3)`` float32 array.

    Flat shading: the face normal is replicated to all three corners, which is
    exactly the look the original polygon renderer produced.
    """
    verts = np.asarray(mesh.verts, dtype=np.float64)
    if colors is None:
        colors = mesh.colors
    colors = np.asarray(colors, dtype=np.float64)

    p0 = verts[triangles[:, 0]]
    p1 = verts[triangles[:, 1]]
    p2 = verts[triangles[:, 2]]
    normal = np.cross(p1 - p0, p2 - p0)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    length[length < 1e-12] = 1.0
    normal = normal / length

    # The material colours in the mesh are authored as sRGB (they came from a
    # renderer that wrote them straight to an 8-bit image). Lighting them as if
    # they were linear and then converting the result to sRGB brightens every
    # surface by roughly a gamma step, which is why the assembly came out pale
    # and washed out. Decode to linear here, once, and the fragment shader can
    # then encode the result back for display.
    linear = np.power(np.clip(colors, 0.0, 1.0), 2.2)

    data = np.empty((len(triangles), 3, 9), dtype=np.float32)
    data[:, :, 0:3] = verts[triangles]
    data[:, :, 3:6] = normal[:, None, :]
    data[:, :, 6:9] = linear[face_of][:, None, :]
    return data.reshape(-1, 9)

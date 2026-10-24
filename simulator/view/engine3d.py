"""Procedural engine assemblies.

The 3D model is generated from the same :class:`~simulator.physics.EngineSpec`
driving the simulation, so a V8 is 90 degrees wide and has eight pistons because
the *physics* says so, not because someone modelled it that way. Change the bore
or the bank angle in a preset and the picture follows.

The assembly is deliberately a cutaway. A solid block would hide everything
interesting, and with a painter's-algorithm renderer it would also be the only
thing ever drawn. Instead the block is its deck plate with real bores in it, the
head is its rails, and the pistons, rods, crank, valves and cams are all in view.

Motion is rigid-body. Every moving part is a prototype mesh plus a single
rotation about the crankshaft axis (X, with Y up) and a translation, because
everything in a piston engine moves in a plane perpendicular to the crank. That
keeps the per-frame update to a couple of matrix multiplies per part, which
matters when the audio thread wants the CPU too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mesh import Mesh, MeshBuilder

TWO_PI = 2.0 * math.pi

# Paint: aluminium castings, dark steel, bronze valve springs, and the two
# manifold families so the intake and exhaust sides read apart at a glance.
ALUMINIUM = (0.60, 0.62, 0.66)
ALUMINIUM_DARK = (0.44, 0.46, 0.50)
CAST_IRON = (0.34, 0.35, 0.38)
STEEL = (0.50, 0.53, 0.58)
STEEL_DARK = (0.30, 0.32, 0.36)
PISTON = (0.74, 0.74, 0.76)
ROD = (0.56, 0.54, 0.50)
BRONZE = (0.58, 0.44, 0.20)
INTAKE = (0.26, 0.32, 0.42)
INTAKE_DARK = (0.19, 0.23, 0.30)
EXHAUST = (0.44, 0.29, 0.22)
FLAME = (0.85, 0.45, 0.16)


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


# Degrees within the four-stroke cycle (0 = firing TDC). These are typical road
# cam figures: the exhaust opens 50 degrees before BDC and shuts just after TDC,
# the intake opens just before TDC and shuts 60 degrees after BDC, so the two
# overlap while both are slightly open.
EXHAUST_OPEN = 130.0
EXHAUST_CLOSE = 370.0
INTAKE_OPEN = 350.0
INTAKE_CLOSE = 600.0


def _lift(angle_deg: float, start: float, end: float) -> float:
    if angle_deg < start or angle_deg > end:
        return 0.0
    x = (angle_deg - start) / max(end - start, 1e-6)
    return max(0.0, math.sin(math.pi * x)) ** 1.2


# --------------------------------------------------------------------------- #
# dimensions
# --------------------------------------------------------------------------- #
@dataclass
class Dims:
    """Everything the geometry needs, derived from the spec, in metres."""

    bore: float
    stroke: float
    crank_radius: float
    rod_length: float
    cylinders: int
    pitch: float
    bank_angle: float
    cycle_revs: float

    x_min: float
    x_max: float
    half_z: float
    wall: float
    deck_y: float
    head_top: float
    cam_y: float
    pan_y: float
    piston_height: float
    piston_top: float

    @property
    def length(self) -> float:
        return self.x_max - self.x_min


def dimensions(spec) -> Dims:
    bore = float(spec.bore)
    stroke = float(spec.stroke)
    r = 0.5 * stroke
    rod = spec.rod_ratio * r
    n = max(int(spec.cylinders), 1)
    pitch = bore * 1.32
    wall = max(bore * 0.16, 0.010)

    # For a V engine each pair of cylinders shares one crank pin, so the block is
    # only half as long as the cylinder count alone would suggest.
    paired = spec.bank_angle > 1e-6 and n > 1
    pins = n // 2 if paired else n
    span = (pins - 1) * pitch if pins > 1 else 0.0
    x_min = -0.5 * span - bore * 0.85
    x_max = 0.5 * span + bore * 0.85

    half_z = bore * 0.80
    piston_height = bore * 0.62
    piston_top = piston_height * 0.55
    deck_y = rod + r + piston_top + bore * 0.06
    head_top = deck_y + bore * 0.75
    cam_y = head_top + bore * 0.30
    pan_y = -(r + bore * 0.55)

    return Dims(
        bore=bore,
        stroke=stroke,
        crank_radius=r,
        rod_length=rod,
        cylinders=n,
        pitch=pitch,
        bank_angle=float(spec.bank_angle),
        cycle_revs=float(spec.cycle_revs),
        x_min=x_min,
        x_max=x_max,
        half_z=half_z,
        wall=wall,
        deck_y=deck_y,
        head_top=head_top,
        cam_y=cam_y,
        pan_y=pan_y,
        piston_height=piston_height,
        piston_top=piston_top,
    )


@dataclass
class BankPlan:
    tilt: float
    cylinders: list[int]
    x: dict[int, float]

    @property
    def inward(self) -> float:
        """Which local Z side of this bank faces into the vee."""
        return -1.0 if self.tilt > 0.0 else 1.0


# --------------------------------------------------------------------------- #
# moving parts
# --------------------------------------------------------------------------- #
class Mover:
    """A prototype mesh plus the rigid transform applied to it each frame."""

    __slots__ = ("mesh", "angle", "offset")

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh
        self.angle = 0.0
        self.offset = np.zeros(3, dtype=np.float64)


def _piston_mesh(dims: Dims, segments: int) -> Mesh:
    b = MeshBuilder()
    radius = dims.bore * 0.5 * 0.94
    height = dims.piston_height
    b.add_prism((0.0, -height * 0.5, 0.0), (0.0, 1.0, 0.0), height, radius, PISTON, segments=segments)
    b.add_prism(
        (0.0, height * 0.5, 0.0), (0.0, 1.0, 0.0), 0.010, radius * 1.02, STEEL, segments=segments
    )
    return b.build()


def _rod_mesh(dims: Dims, segments: int) -> Mesh:
    b = MeshBuilder()
    length = dims.rod_length
    width = dims.bore * 0.20
    thickness = dims.bore * 0.16
    b.add_box((0.0, length * 0.5, 0.0), (width, length, thickness), ROD)
    b.add_prism((0.0, -thickness * 0.9, 0.0), (1.0, 0.0, 0.0), thickness * 1.8, width * 0.9, ROD, segments=8)
    b.add_prism(
        (0.0, length - thickness * 0.7, 0.0), (1.0, 0.0, 0.0), thickness * 1.4, width * 0.62, ROD, segments=8
    )
    return b.build()


def _crank_throw_mesh(dims: Dims, segments: int) -> Mesh:
    """One crank throw: arm, journal and counterweight, turned as a unit."""
    b = MeshBuilder()
    r = dims.crank_radius
    pin_radius = dims.bore * 0.13
    web = dims.bore * 0.16
    b.add_box((0.0, r * 0.5, 0.0), (web, r + pin_radius, pin_radius * 2.2), STEEL_DARK)
    b.add_prism((-web * 0.5, r, 0.0), (1.0, 0.0, 0.0), web, pin_radius, STEEL, segments=10)
    b.add_box((0.0, -r * 0.75, 0.0), (web, r * 0.9, dims.bore * 0.52), STEEL_DARK)
    return b.build()


def _flywheel_mesh(dims: Dims, segments: int) -> Mesh:
    b = MeshBuilder()
    radius = dims.bore * 0.95
    thickness = dims.bore * 0.14
    b.add_prism(
        (-thickness * 0.5, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        thickness,
        radius,
        CAST_IRON,
        segments=max(16, segments),
    )
    # a few spokes so rotation is readable
    for i in range(4):
        a = i * TWO_PI / 4.0
        b.add_box(
            (0.0, radius * 0.60 * math.cos(a), radius * 0.60 * math.sin(a)),
            (thickness, radius * 0.62, radius * 0.16),
            STEEL,
        )
    return b.build()


def _valve_mesh(dims: Dims, segments: int) -> Mesh:
    """Stem, spring and retainer, standing on the local origin."""
    b = MeshBuilder()
    stem_r = dims.bore * 0.045
    length = dims.bore * 0.80
    b.add_prism((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), length, stem_r, STEEL, segments=8)
    spring_r = dims.bore * 0.115
    b.add_prism((0.0, length * 0.16, 0.0), (0.0, 1.0, 0.0), length * 0.44, spring_r, BRONZE, segments=8)
    b.add_prism(
        (0.0, length * 0.60, 0.0), (0.0, 1.0, 0.0), dims.bore * 0.05, spring_r * 1.5, STEEL_DARK, segments=8
    )
    return b.build()


def _cam_mesh(dims: Dims, segments: int) -> Mesh:
    b = MeshBuilder()
    length = dims.length * 0.96
    radius = dims.bore * 0.13
    b.add_prism((-length * 0.5, 0.0, 0.0), (1.0, 0.0, 0.0), length, radius, STEEL, segments=10)
    return b.build()


def _cam_lobe_mesh(dims: Dims, segments: int) -> Mesh:
    b = MeshBuilder()
    radius = dims.bore * 0.19
    thickness = dims.bore * 0.22
    b.add_prism(
        (-thickness * 0.5, radius * 0.34, 0.0), (1.0, 0.0, 0.0), thickness, radius, STEEL_DARK, segments=10
    )
    return b.build()


# --------------------------------------------------------------------------- #
# the scene
# --------------------------------------------------------------------------- #
class EngineScene:
    """Builds and animates the 3D assembly for one engine."""

    def __init__(self, spec, segments: int = 12) -> None:
        self.spec = spec
        self.dims = dimensions(spec)
        self.segments = int(segments)
        layout = str(getattr(spec, "layout", "inline"))
        if layout == "turbine" or spec.kind == "turbine":
            self.kind = "turbine"
        elif layout == "rotary":
            self.kind = "rotary"
        else:
            # "radial" has no cylinder-star geometry of its own: it falls back to
            # the inline arrangement. The sound and the physics handle a radial
            # correctly (uneven firing, two banks), so only the picture is an
            # approximation -- and no bundled preset uses it.
            self.kind = "piston"

        self.plans: list[BankPlan] = []
        self.static = Mesh.empty()
        self.movers: list[Mover] = []
        self._model = None

        self._piston_movers: list[tuple[Mover, int, BankPlan]] = []
        self._rod_movers: list[tuple[Mover, int, BankPlan]] = []
        self._crank_movers: list[tuple[Mover, int]] = []
        self._valve_movers: list[tuple[Mover, int, BankPlan, float, bool]] = []
        self._cam_movers: list[tuple[Mover, BankPlan]] = []
        self._lobe_movers: list[tuple[Mover, BankPlan, float]] = []
        self._rotor_movers: list[tuple[Mover, int]] = []
        self._spin_movers: list[tuple[Mover, float, bool]] = []

        if self.kind == "turbine":
            self._build_turbine()
        elif self.kind == "rotary":
            self._build_rotary()
        else:
            self._build_piston()

        self._allocate()

    # ------------------------------------------------------------------ #
    # piston engines
    # ------------------------------------------------------------------ #
    def _plan_banks(self) -> None:
        dims = self.dims
        n = dims.cylinders
        angle = dims.bank_angle
        if angle > 1e-6 and n > 1:
            half = angle * 0.5
            plans = [
                BankPlan(tilt=-half, cylinders=list(range(0, n, 2)), x={}),
                BankPlan(tilt=half, cylinders=list(range(1, n, 2)), x={}),
            ]
        else:
            plans = [BankPlan(tilt=0.0, cylinders=list(range(n)), x={})]

        for plan in plans:
            count = len(plan.cylinders)
            span = (count - 1) * dims.pitch if count > 1 else 0.0
            for k, cyl in enumerate(plan.cylinders):
                plan.x[cyl] = -0.5 * span + k * dims.pitch
        self.plans = plans

    @property
    def paired(self) -> bool:
        return len(self.plans) > 1

    def _build_piston(self) -> None:
        dims = self.dims
        seg = self.segments
        self._plan_banks()

        static = MeshBuilder()
        self._build_bedplate(static)
        self._build_crank_axis(static)

        for plan in self.plans:
            if dims.cycle_revs >= 1.5:
                self._build_head(static, plan)
                self._build_manifolds(static, plan)
            else:
                self._build_two_stroke(static, plan)

        # the crankshaft: one throw per pin, all turning together
        pin_indices = sorted({self._pin_index(c) for c in range(dims.cylinders)})
        throw = _crank_throw_mesh(dims, seg)
        for p in pin_indices:
            mover = Mover(throw)
            self.movers.append(mover)
            self._crank_movers.append((mover, p))

        fly = Mover(_flywheel_mesh(dims, seg))
        fly.offset = np.array([dims.x_max + dims.bore * 0.45, 0.0, 0.0])
        self.movers.append(fly)
        self._spin_movers.append((fly, 1.0, False))

        piston = _piston_mesh(dims, seg)
        rod = _rod_mesh(dims, seg)

        for plan in self.plans:
            for cyl in plan.cylinders:
                pm = Mover(piston)
                self.movers.append(pm)
                self._piston_movers.append((pm, cyl, plan))

                rm = Mover(rod)
                self.movers.append(rm)
                self._rod_movers.append((rm, cyl, plan))

            if dims.cycle_revs >= 1.5:
                valve = _valve_mesh(dims, seg)
                cam = _cam_mesh(dims, seg)
                lobe = _cam_lobe_mesh(dims, seg)

                cam_move = Mover(cam)
                self.movers.append(cam_move)
                self._cam_movers.append((cam_move, plan))

                for i, cyl in enumerate(plan.cylinders):
                    for intake in (True, False):
                        vm = Mover(valve)
                        self.movers.append(vm)
                        lateral = plan.inward * dims.half_z * 0.42 * (1.0 if intake else -1.0)
                        self._valve_movers.append((vm, cyl, plan, lateral, intake))
                    lm = Mover(lobe)
                    lm.offset = np.array([plan.x[cyl], 0.0, 0.0])
                    self.movers.append(lm)
                    self._lobe_movers.append((lm, plan, plan.x[cyl]))
                    _ = i

        self.static = static.build()

    def _build_bedplate(self, b: MeshBuilder) -> None:
        dims = self.dims
        x0, x1 = dims.x_min, dims.x_max
        length = x1 - x0
        cx = 0.5 * (x0 + x1)
        depth = dims.half_z * 2.0 + dims.wall * 2.0

        b.add_box(
            (cx, dims.pan_y + dims.bore * 0.16, 0.0),
            (length, dims.bore * 0.32, depth * 0.86),
            CAST_IRON,
        )
        for z in (-dims.half_z - dims.wall * 0.5, dims.half_z + dims.wall * 0.5):
            b.add_box((cx, dims.pan_y * 0.45, z), (length, dims.bore * 0.60, dims.wall), STEEL_DARK)
            # The crankcase walls stay below the crank centreline. Higher walls
            # would hide the crank, rods and pistons, which is the whole point of
            # drawing the engine open in the first place.
            wall_top = -dims.crank_radius * 0.35
            wall_bottom = dims.pan_y * 0.9
            b.add_box(
                (cx, 0.5 * (wall_top + wall_bottom), z),
                (length, wall_top - wall_bottom, dims.wall),
                ALUMINIUM_DARK,
            )

        pins = max(len({self._pin_index(c) for c in range(dims.cylinders)}), 1)
        span = (pins - 1) * dims.pitch if pins > 1 else 0.0
        for k in range(pins):
            x = -0.5 * span + k * dims.pitch
            b.add_box(
                (x, dims.pan_y * 0.30, 0.0),
                (dims.wall * 1.6, abs(dims.pan_y) * 0.9, dims.half_z * 1.7),
                STEEL_DARK,
            )
        for x in (x0, x1):
            b.add_box((x, dims.pan_y * 0.45, 0.0), (dims.wall, dims.bore * 0.60, depth), STEEL_DARK)

    def _build_crank_axis(self, b: MeshBuilder) -> None:
        dims = self.dims
        radius = dims.bore * 0.155
        x0 = dims.x_min - dims.bore * 0.30
        x1 = dims.x_max + dims.bore * 0.30
        b.add_prism((x0, 0.0, 0.0), (1.0, 0.0, 0.0), x1 - x0, radius, STEEL, segments=max(12, self.segments))

    def _build_head(self, b: MeshBuilder, plan: BankPlan, valvetrain: bool = True) -> None:
        """Deck, head rails and cam pedestals, built along +Y then tilted."""
        dims = self.dims
        local = MeshBuilder()
        x0 = min(plan.x.values()) - dims.bore * 0.75
        x1 = max(plan.x.values()) + dims.bore * 0.75
        length = x1 - x0
        cx = 0.5 * (x0 + x1)
        rail_z = dims.half_z + dims.wall * 0.5
        rail_h = dims.wall * 1.2

        for z in (-rail_z, rail_z):
            local.add_box((cx, dims.deck_y + rail_h * 0.5, z), (length, rail_h, dims.wall), ALUMINIUM)
            local.add_box(
                (cx, dims.head_top - dims.wall * 0.7, z),
                (length, dims.wall * 1.4, dims.wall * 1.1),
                ALUMINIUM_DARK,
            )
        for cyl in plan.cylinders:
            local.add_ring(
                (plan.x[cyl], dims.deck_y + rail_h, 0.0),
                (0.0, 1.0, 0.0),
                dims.bore * 0.5,
                dims.half_z,
                ALUMINIUM,
                segments=max(12, self.segments),
            )
        for a, c in zip(plan.cylinders, plan.cylinders[1:]):
            gap_x = 0.5 * (plan.x[a] + plan.x[c])
            local.add_box(
                (gap_x, dims.deck_y + rail_h * 0.5, 0.0),
                (max(abs(plan.x[c] - plan.x[a]) - dims.bore * 1.02, dims.wall), rail_h, dims.half_z * 1.7),
                ALUMINIUM_DARK,
            )
        if valvetrain:
            for cyl in plan.cylinders:
                local.add_box(
                    (plan.x[cyl], 0.5 * (dims.head_top + dims.cam_y), -dims.half_z * 0.30),
                    (dims.wall * 1.4, dims.cam_y - dims.head_top, dims.wall * 2.2),
                    ALUMINIUM_DARK,
                )

        mesh = local.build().transformed(rot_x(plan.tilt))
        b.add_mesh(mesh)

    def _build_manifolds(self, b: MeshBuilder, plan: BankPlan) -> None:
        """An intake log inside the vee and an exhaust log outside it."""
        dims = self.dims
        local = MeshBuilder()
        x0 = min(plan.x.values())
        x1 = max(plan.x.values())
        runner_y = dims.cam_y + dims.bore * 0.10
        inward = plan.inward
        for sign, colour in ((inward, INTAKE), (-inward, EXHAUST)):
            z = sign * (dims.half_z + dims.bore * 0.55)
            local.add_prism(
                (x0 - dims.bore * 0.6, runner_y, z),
                (1.0, 0.0, 0.0),
                (x1 - x0) + dims.bore * 1.2,
                dims.bore * 0.19,
                colour,
                segments=10,
            )
            for cyl in plan.cylinders:
                local.add_cylinder_between(
                    (plan.x[cyl], dims.cam_y - dims.bore * 0.35, 0.0),
                    (plan.x[cyl], runner_y, z),
                    dims.bore * 0.11,
                    colour,
                    segments=8,
                )
        mesh = local.build().transformed(rot_x(plan.tilt))
        b.add_mesh(mesh)

    def _build_two_stroke(self, b: MeshBuilder, plan: BankPlan) -> None:
        """An air-cooled finned barrel and a tuned expansion chamber."""
        dims = self.dims
        bore = dims.bore
        local = MeshBuilder()
        out = -plan.inward  # the exhaust leaves the side the inlet does not
        for cyl in plan.cylinders:
            x = plan.x[cyl]
            # head casting, sat straight on the deck
            local.add_box(
                (x, dims.deck_y + bore * 0.22, 0.0),
                (bore * 1.45, bore * 0.45, bore * 1.55),
                ALUMINIUM_DARK,
            )
            # A two-stroke single is a closed casting, unlike the cutaway
            # multi-cylinder blocks: without the barrel the piston just hangs in
            # space between the head and the crankcase.
            bottom = -dims.crank_radius * 0.35
            local.add_box(
                (x, 0.5 * (dims.deck_y + bottom), 0.0),
                (bore * 1.24, dims.deck_y - bottom, bore * 1.24),
                CAST_IRON,
            )
            # cooling fins, the signature of an air-cooled two-stroke. They are
            # drawn as short ribs against the casting rather than full rings:
            # wide rings read as loose plates from the camera's high angle.
            for k in range(5):
                local.add_box(
                    (x, dims.deck_y - bore * (0.30 + 0.28 * k), 0.0),
                    (bore * 1.42, bore * 0.10, bore * 1.42),
                    ALUMINIUM_DARK,
                )
            # spark plug
            local.add_cylinder_between(
                (x, dims.deck_y + bore * 0.44, 0.0),
                (x, dims.deck_y + bore * 0.78, -bore * 0.18),
                bore * 0.10,
                STEEL,
                segments=8,
            )

            y = dims.deck_y - bore * 0.85
            z0 = out * bore * 0.62
            z1 = out * bore * 1.55
            z2 = out * bore * 2.45
            z3 = out * bore * 3.25
            # exhaust port, header and the diverging cone that makes a
            # two-stroke behave like a tuned pipe
            local.add_cylinder_between(
                (x, y, z0), (x, y + bore * 0.06, z1), bore * 0.12, EXHAUST, segments=8
            )
            local.add_cylinder_between(
                (x, y + bore * 0.06, z1),
                (x, y + bore * 0.50, z2),
                bore * 0.30,
                EXHAUST,
                segments=max(12, self.segments),
            )
            # ... and the stinger that sets the back pressure
            local.add_cylinder_between(
                (x, y + bore * 0.50, z2), (x, y + bore * 0.68, z3), bore * 0.075, EXHAUST, segments=8
            )
            # reed block and carburettor on the other side
            local.add_box(
                (x, y, plan.inward * bore * 0.95),
                (bore * 0.46, bore * 0.44, bore * 0.40),
                INTAKE_DARK,
            )
        mesh = local.build().transformed(rot_x(plan.tilt))
        b.add_mesh(mesh)

    # ------------------------------------------------------------------ #
    # rotary (Wankel)
    # ------------------------------------------------------------------ #
    def _build_rotary(self) -> None:
        dims = self.dims
        seg = max(16, self.segments)
        rotors = max(int(self.spec.cylinders), 1)
        width = dims.bore * 0.85
        housing_r = dims.bore * 1.30
        gap = width * 0.18
        total = rotors * (width + gap)

        static = MeshBuilder()
        x0 = -0.5 * total
        # side housings, one between each rotor and one at each end
        for k in range(rotors + 1):
            x = x0 + k * (width + gap)
            static.add_prism((x - gap * 0.5, 0.0, 0.0), (1.0, 0.0, 0.0), gap, housing_r, CAST_IRON, segments=seg)
        # eccentric shaft
        static.add_prism(
            (x0 - gap, 0.0, 0.0), (1.0, 0.0, 0.0), total + 2 * gap, dims.bore * 0.12, STEEL, segments=10
        )
        # peripheral intake/exhaust ports on the top and bottom
        for sign, colour in ((1.0, INTAKE), (-1.0, EXHAUST)):
            static.add_prism(
                (x0 - gap, sign * housing_r * 0.95, 0.0),
                (1.0, 0.0, 0.0),
                total + 2 * gap,
                dims.bore * 0.12,
                colour,
                segments=8,
            )
        self.static = static.build()
        self._rotary_geometry = (width, gap, housing_r, x0, total)

        rotor_mesh = self._rotor_mesh(housing_r * 0.72, width * 0.70)
        eccentricity = housing_r * 0.16
        self._rotor_x = []
        for k in range(rotors):
            centre_x = x0 + k * (width + gap) + (gap + width) * 0.5
            self._rotor_x.append(centre_x)
            mover = Mover(rotor_mesh)
            self.movers.append(mover)
            self._rotor_movers.append((mover, k))

        pulley_mesh = _flywheel_mesh(dims, seg).transformed(np.diag([1.0, 0.55, 0.55]))
        pulley = Mover(pulley_mesh)
        pulley.offset = np.array([x0 + total + gap * 2.5, 0.0, 0.0])
        self.movers.append(pulley)
        self._spin_movers.append((pulley, 1.0, False))
        self._rotary_eccentricity = eccentricity
        self._rotary_x0 = x0

    def _rotor_mesh(self, radius: float, width: float) -> Mesh:
        """A Wankel rotor: a rounded triangle, as a three-lobed prism."""
        segments = 30
        angles = np.arange(segments, dtype=np.float64) * (TWO_PI / segments)
        # A Wankel rotor is close to a Reuleaux triangle: three lobes, but only
        # gently, or it stops reading as a rotor and starts reading as a star.
        radii = radius * (1.0 + 0.22 * np.cos(3.0 * angles))
        b = MeshBuilder()
        b.add_profile_prism((-width * 0.5, 0.0, 0.0), (1.0, 0.0, 0.0), width, radii, ALUMINIUM)
        return b.build()

    # ------------------------------------------------------------------ #
    # gas turbine
    # ------------------------------------------------------------------ #
    def _build_turbine(self) -> None:
        seg = max(18, self.segments)
        # The preset's 200 mm "bore" is a placeholder for a machine that has no
        # cylinders at all, so the casing is sized directly instead.
        radius = 0.165
        length = 0.74

        static = MeshBuilder()
        self._turb_radius = radius
        self._turb_length = length
        # The casing is a set of bands rather than a closed tube: an uncapped
        # cylinder still hides its own interior from the side, so the gaps
        # between bands are what let the compressor and turbine be seen at all.
        for i in range(5):
            x = -length * 0.5 + i * (length / 4.0)
            static.add_prism(
                (x - 0.012, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                0.024,
                radius,
                ALUMINIUM_DARK,
                segments=seg,
                cap_start=False,
                cap_end=False,
            )
        for i in range(6):
            a = i * TWO_PI / 6.0
            static.add_cylinder_between(
                (-length * 0.5, radius * 1.02 * math.cos(a), radius * 1.02 * math.sin(a)),
                (length * 0.5, radius * 1.02 * math.cos(a), radius * 1.02 * math.sin(a)),
                0.008,
                CAST_IRON,
                segments=6,
            )
        # combustor cans
        for i in range(6):
            a = i * TWO_PI / 6.0
            static.add_cylinder_between(
                (-length * 0.26, radius * 0.58 * math.cos(a), radius * 0.58 * math.sin(a)),
                (length * 0.08, radius * 0.46 * math.cos(a), radius * 0.46 * math.sin(a)),
                radius * 0.13,
                FLAME,
                segments=8,
            )
        static.add_prism(
            (-length * 0.62, 0.0, 0.0), (1.0, 0.0, 0.0), length * 1.24, radius * 0.09, STEEL, segments=10
        )
        self.static = static.build()

        for x, blade_r in ((-length * 0.38, radius * 0.74), (length * 0.32, radius * 0.68)):
            disc = MeshBuilder()
            disc.add_prism((x - 0.014, 0.0, 0.0), (1.0, 0.0, 0.0), 0.028, blade_r * 0.45, STEEL, segments=10)
            for i in range(14):
                a = i * TWO_PI / 14.0
                disc.add_box(
                    (x, blade_r * 0.62 * math.cos(a), blade_r * 0.62 * math.sin(a)),
                    (0.022, blade_r * 0.80, 0.020),
                    STEEL_DARK,
                )
            mover = Mover(disc.build())
            self.movers.append(mover)
            self._spin_movers.append((mover, 1.0, False))

        tail = MeshBuilder()
        tail.add_prism((length * 0.5, 0.0, 0.0), (1.0, 0.0, 0.0), length * 0.22, radius * 0.10, STEEL, segments=10)
        m = Mover(tail.build())
        self.movers.append(m)
        self._spin_movers.append((m, 1.0, False))

    # ------------------------------------------------------------------ #
    # per-frame update
    # ------------------------------------------------------------------ #
    def update(self, model) -> None:
        self._model = model
        theta = float(model.theta)

        if self.kind == "turbine":
            for mover, rate, _ in self._spin_movers:
                mover.angle = (theta * rate) % TWO_PI
            return

        if self.kind == "rotary":
            ecc = self._rotary_eccentricity
            for mover, k in self._rotor_movers:
                mover.angle = (theta / 3.0 + k * TWO_PI / 3.0) % TWO_PI
                centre = np.array([self._rotor_x[k], 0.0, 0.0])
                mover.offset = centre + np.array([0.0, ecc * math.cos(theta), ecc * math.sin(theta)])
            for mover, rate, _ in self._spin_movers:
                mover.angle = (theta * rate) % TWO_PI
            return

        dims = self.dims
        r = dims.crank_radius
        l = dims.rod_length

        for mover, rate, _ in self._spin_movers:
            mover.angle = (theta * rate) % TWO_PI

        for mover, cyl, plan in self._piston_movers:
            tilt = plan.tilt
            a = self._pin_angle(cyl, plan)
            alpha = a - tilt
            s = r * math.sin(alpha)
            d = r * math.cos(alpha) + math.sqrt(max(l * l - s * s, 1e-9))
            mover.angle = tilt
            mover.offset = rot_x(tilt) @ np.array([plan.x[cyl], d, 0.0])

        for mover, cyl, plan in self._rod_movers:
            tilt = plan.tilt
            a = self._pin_angle(cyl, plan)
            x = plan.x[cyl]
            pin = np.array([x, r * math.cos(a), r * math.sin(a)])
            alpha = a - tilt
            s = r * math.sin(alpha)
            d = r * math.cos(alpha) + math.sqrt(max(l * l - s * s, 1e-9))
            top = rot_x(tilt) @ np.array([x, d, 0.0])
            delta = top - pin
            mover.angle = math.atan2(delta[2], delta[1])
            mover.offset = pin

        for mover, pin_index in self._crank_movers:
            mover.angle = self._pin_angle_by_index(pin_index)
            mover.offset = np.array([self._pin_x(pin_index), 0.0, 0.0])

        if dims.cycle_revs < 1.5:
            return

        cam_angle = theta * 0.5
        for mover, cyl, plan, lateral, intake in self._valve_movers:
            ca = math.degrees(model.cylinder_angle(cyl)) % 720.0
            lift = _lift(ca, INTAKE_OPEN, INTAKE_CLOSE) if intake else _lift(ca, EXHAUST_OPEN, EXHAUST_CLOSE)
            base = dims.deck_y + dims.bore * 0.35
            y = base - lift * dims.bore * 0.18
            mover.angle = plan.tilt
            mover.offset = rot_x(plan.tilt) @ np.array([plan.x[cyl], y, lateral])

        for mover, plan in self._cam_movers:
            mover.angle = plan.tilt + cam_angle
            mover.offset = rot_x(plan.tilt) @ np.array([0.0, dims.cam_y, -dims.half_z * 0.30])

        for mover, plan, x in self._lobe_movers:
            mover.angle = plan.tilt + cam_angle
            mover.offset = rot_x(plan.tilt) @ np.array([x, dims.cam_y, -dims.half_z * 0.30])

    # ------------------------------------------------------------------ #
    def _pin_index(self, cyl: int) -> int:
        return cyl // 2 if self.paired else cyl

    def _pin_angle(self, cyl: int, plan: BankPlan) -> float:
        _ = plan
        return self._pin_angle_by_index(self._pin_index(cyl))

    def _pin_angle_by_index(self, index: int) -> float:
        step = 2 if self.paired else 1
        return self._model.cylinder_angle(index * step) % TWO_PI

    def _pin_x(self, index: int) -> float:
        cyl = index * (2 if self.paired else 1)
        for plan in self.plans:
            if cyl in plan.x:
                return plan.x[cyl]
        return 0.0

    # ------------------------------------------------------------------ #
    # buffer management
    # ------------------------------------------------------------------ #
    def _allocate(self) -> None:
        base = self.static
        nv = base.verts.shape[0]
        ni = base.index.shape[0]
        nf = base.face_count
        tris = base.triangle_count
        for mover in self.movers:
            mesh = mover.mesh
            nv += mesh.verts.shape[0]
            ni += mesh.index.shape[0]
            nf += mesh.face_count
            tris += mesh.triangle_count
        self._verts = np.zeros((nv, 3), dtype=np.float64)
        self._index = np.zeros((ni,), dtype=np.int32)
        self._starts = np.zeros((nf + 1,), dtype=np.int32)
        self._colors = np.zeros((nf, 3), dtype=np.float64)
        self._verts[: base.verts.shape[0]] = base.verts
        self._index[: base.index.shape[0]] = base.index
        self._starts[: base.face_count + 1] = base.starts
        self._colors[: base.face_count] = base.colors
        self._static_verts = base.verts.shape[0]
        self._static_index = base.index.shape[0]
        self._static_faces = base.face_count
        self._tris_equiv = int(tris)

    def build(self) -> Mesh:
        """Pack every part into one mesh for the renderer. Reuses its buffers."""
        verts = self._verts
        index = self._index
        starts = self._starts
        colors = self._colors
        voff = self._static_verts
        ioff = self._static_index
        foff = self._static_faces
        for mover in self.movers:
            mesh = mover.mesh
            nv = mesh.verts.shape[0]
            ni = mesh.index.shape[0]
            nf = mesh.face_count
            verts[voff : voff + nv] = mesh.verts @ rot_x(mover.angle).T + mover.offset
            index[ioff : ioff + ni] = mesh.index + voff
            # the face offsets are cumulative, so they are rebased, not just shifted
            starts[foff + 1 : foff + nf + 1] = mesh.starts[1:] + ioff
            colors[foff : foff + nf] = mesh.colors
            voff += nv
            ioff += ni
            foff += nf
        return Mesh(verts, index, starts, colors)

    # ------------------------------------------------------------------ #
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        dims = self.dims
        if self.kind == "turbine":
            r = self._turb_radius
            half = self._turb_length * 0.78
            return np.array([-half, -r, -r]), np.array([half, r, r])
        if self.kind == "rotary":
            _width, gap, housing_r, x0, total = self._rotary_geometry
            return (
                np.array([x0 - gap * 4.5, -housing_r, -housing_r]),
                np.array([x0 + total + gap, housing_r, housing_r]),
            )
        # wide enough to swallow the flywheel out front and the manifold logs
        span_y = max(dims.half_z * 1.8, dims.bore * 1.35)
        lo = np.array([dims.x_min - dims.bore * 0.55, dims.pan_y - dims.bore * 0.35, -span_y])
        hi = np.array([dims.x_max + dims.bore * 0.75, dims.cam_y + dims.bore * 0.75, span_y])
        if dims.cycle_revs < 1.5:
            # A two-stroke's tuned pipe runs out to 3.25 bore on one side only,
            # so the symmetric span above cut it off in the frame. It is also
            # what the montage and --render-png use to fit the camera.
            hi[2] = max(hi[2], dims.bore * 3.55)
            lo[2] = min(lo[2], -dims.bore * 1.55)
        return lo, hi

    def focus(self) -> np.ndarray:
        lo, hi = self.bounds()
        return 0.5 * (lo + hi)

    def framing_distance(self) -> float:
        lo, hi = self.bounds()
        extent = float(np.max(hi - lo))
        return max(extent * 1.25, 0.35)

    def face_count(self) -> int:
        return int(self._starts.shape[0] - 1)

    def triangle_count(self) -> int:
        """The equivalent triangle count, so the budget stays comparable."""
        return int(getattr(self, "_tris_equiv", 0))

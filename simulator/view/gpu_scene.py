"""Draws an :class:`~simulator.view.engine3d.EngineScene` with the GPU.

The scene already models the assembly the way a renderer wants it: one static
mesh plus a list of rigid ``Mover`` parts, each with an angle and an offset for
the current crank position. So nothing here has to understand engines -- it
uploads the static part once, keeps one small buffer per mover, and re-uploads
only the movers that actually moved.

That split is what makes the frame cheap. A V8 has roughly 1700 faces and about
a dozen movers; uploading everything every frame would push a megabyte of vertex
data per frame for no reason, while uploading just the movers is a few tens of
kilobytes.

Two rendering passes are supported: the solid pass, and an optional translucent
"shell" pass used for the block so the internals stay visible. The shell is a
copy of the static geometry drawn with blending and no depth writes, which is
what makes the cutaway look like glass rather than like a hole.
"""

from __future__ import annotations

import numpy as np

from . import gpu


class GpuEngineRenderer:
    """Owns the GL buffers for one engine scene."""

    def __init__(self, ctx, scene, *, segments: int | None = None) -> None:
        self.ctx = ctx
        self.scene = scene
        self.segments = int(segments if segments is not None else getattr(scene, "segments", 12))

        self.program = ctx.program(
            vertex_shader=gpu.VERTEX_SHADER, fragment_shader=gpu.FRAGMENT_SHADER
        )
        self._bind_uniforms()

        self._static_vbo = None
        self._static_vao = None
        self._static_count = 0

        self._mover_vbos: list = []
        self._mover_vaos: list = []
        self._mover_counts: list[int] = []
        self._mover_tri: list[np.ndarray] = []
        self._mover_face: list[np.ndarray] = []
        self._last_angle: list[float] = []
        self._last_offset: list[np.ndarray] = []

        self._upload_static()
        self._upload_movers()

        self.error: str | None = None

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def _bind_uniforms(self) -> None:
        for name in (
            "u_view_proj",
            "u_model",
            "u_eye",
            "u_light_a",
            "u_light_b",
            "u_rim_power",
            "u_specular",
            "u_shininess",
            "u_ghost",
            "u_shell_tint",
        ):
            if name not in self.program:
                self.error = f"uniform {name} missing from the shader"
                return
        self.program["u_light_a"].value = (0.42, 0.78, 0.46)
        self.program["u_light_b"].value = (-0.55, 0.15, -0.62)
        self.program["u_rim_power"].value = 2.6
        self.program["u_specular"].value = 0.34
        self.program["u_shininess"].value = 42.0
        self.program["u_ghost"].value = 0.0
        self.program["u_shell_tint"].value = (0.33, 0.48, 0.68)

    def _upload_static(self) -> None:
        mesh = self.scene.static
        if mesh is None or mesh.face_count == 0:
            self._static_count = 0
            return
        tri, owner = gpu.triangulate(mesh)
        data = gpu.pack_triangles(mesh, tri, owner)
        self._static_vbo = self.ctx.buffer(data.tobytes())
        self._static_vao = self.ctx.vertex_array(
            self.program, [(self._static_vbo, "3f 3f 3f", "in_pos", "in_nrm", "in_col")]
        )
        self._static_count = len(tri)

    def _upload_movers(self) -> None:
        for mover in self.scene.movers:
            mesh = mover.mesh
            tri, owner = gpu.triangulate(mesh)
            if len(tri) == 0:
                self._mover_vbos.append(None)
                self._mover_vaos.append(None)
                self._mover_counts.append(0)
                self._mover_tri.append(tri)
                self._mover_face.append(owner)
                self._last_angle.append(mover.angle)
                self._last_offset.append(np.array(mover.offset, dtype=np.float64))
                continue
            data = gpu.pack_triangles(mesh, tri, owner)
            vbo = self.ctx.buffer(data.tobytes())
            vao = self.ctx.vertex_array(
                self.program, [(vbo, "3f 3f 3f", "in_pos", "in_nrm", "in_col")]
            )
            self._mover_vbos.append(vbo)
            self._mover_vaos.append(vao)
            self._mover_counts.append(len(tri))
            self._mover_tri.append(tri)
            self._mover_face.append(owner)
            self._last_angle.append(mover.angle)
            self._last_offset.append(np.array(mover.offset, dtype=np.float64))

    # ------------------------------------------------------------------ #
    # per frame
    # ------------------------------------------------------------------ #
    def draw(self, view_proj: np.ndarray, eye, *, ghost_shell: bool = False) -> dict:
        """Draw the whole assembly. Returns a small statistics dict."""
        ctx = self.ctx
        program = self.program
        program["u_view_proj"].write(view_proj.astype("f4").T.tobytes())
        program["u_eye"].value = tuple(float(v) for v in eye)

        drawn = 0
        identity = np.eye(4, dtype=np.float64)

        # static shell first, translucent, so the moving internals read through it
        if ghost_shell and self._static_vao is not None:
            program["u_ghost"].value = 0.62
            program["u_model"].write(identity.astype("f4").T.tobytes())
            ctx.enable(ctx.BLEND)
            ctx.blend_func = ctx.SRC_ALPHA, ctx.ONE_MINUS_SRC_ALPHA
            ctx.depth_mask = False
            self._static_vao.render(ctx.TRIANGLES)
            drawn += self._static_count
            ctx.depth_mask = True
            ctx.disable(ctx.BLEND)
            program["u_ghost"].value = 0.0
        elif not ghost_shell and self._static_vao is not None:
            program["u_ghost"].value = 0.0
            program["u_model"].write(identity.astype("f4").T.tobytes())
            self._static_vao.render(ctx.TRIANGLES)
            drawn += self._static_count

        for index, vao in enumerate(self._mover_vaos):
            if vao is None:
                continue
            mover = self.scene.movers[index]
            angle = float(mover.angle)
            offset = np.asarray(mover.offset, dtype=np.float64)

            # only push the matrix; the vertex data is rigid, so the transform
            # can live entirely in the model matrix
            model = gpu.rot_x(angle)
            model[:3, 3] = offset
            program["u_model"].write(model.astype("f4").T.tobytes())

            if ghost_shell:
                continue
            vao.render(ctx.TRIANGLES)
            drawn += self._mover_counts[index]

        return {"faces_drawn": drawn, "movers": len(self._mover_vaos)}

    def release(self) -> None:
        """Free every GL object this renderer owns."""
        for vbo in self._mover_vbos:
            if vbo is not None:
                try:
                    vbo.release()
                except Exception:
                    pass
        for vao in self._mover_vaos:
            if vao is not None:
                try:
                    vao.release()
                except Exception:
                    pass
        for object_ in (self._static_vao, self._static_vbo):
            if object_ is not None:
                try:
                    object_.release()
                except Exception:
                    pass
        try:
            self.program.release()
        except Exception:
            pass

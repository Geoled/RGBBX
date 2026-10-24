"""The live 3D viewport: the GPU engine assembly inside the imgui interface.

The renderer is ModernGL drawing the assembly into an offscreen framebuffer,
which is then blitted into the interface as a texture. Nothing here knows about
engines: :class:`~simulator.view.engine3d.EngineScene` builds the geometry,
:class:`~simulator.view.gpu_scene.GpuEngineRenderer` owns the buffers, and this
module owns the camera, the input and the lifetime of the GL objects.

Camera input follows the DCC convention: drag orbits, wheel zooms, and the
framing is derived from the engine's bounding sphere so a Wankel and a V12 both
fill the frame (that logic lives in ``simulator.view.gpu``).
"""

from __future__ import annotations

import math

import numpy as np

from ..view import gpu
from ..view.engine3d import EngineScene
from ..view.gpu_scene import GpuEngineRenderer


class Viewport3D:
    """Owns the GL context, the scene buffers and the camera for one engine."""

    def __init__(self, spec, width: int = 960, height: int = 640, segments: int = 12,
                 standalone: bool = False) -> None:
        import moderngl

        if standalone:
            # headless: no window, used by --capture and the doc stills
            self.ctx = moderngl.create_standalone_context()
        else:
            self.ctx = moderngl.create_context()
        self.scene = EngineScene(spec, segments=segments)
        self.renderer = GpuEngineRenderer(self.ctx, self.scene, segments=segments)

        self.width = int(width)
        self.height = int(height)
        self.azimuth = 0.62
        self.elevation = 0.40
        self.zoom = 1.0
        self.show_shell = False
        self._center = None
        self._base_distance = 1.0
        self._frame_program = None
        self._quad = None
        self._fbo = None
        self._tex = None
        self._depth = None
        self._error_shown = False
        self._resize(self.width, self.height)
        self.frame_camera()

    # ------------------------------------------------------------------ #
    # resources
    # ------------------------------------------------------------------ #
    def _resize(self, width: int, height: int) -> None:
        width = max(int(width), 16)
        height = max(int(height), 16)
        if self._fbo is not None and width == self.width and height == self.height:
            return
        if self._fbo is not None:
            self._fbo.release()
            self._tex.release()
            if self._depth is not None:
                self._depth.release()
        self.width, self.height = width, height
        self._tex = self.ctx.texture((width, height), 4)
        self._tex.repeat_x = False
        self._tex.repeat_y = False
        # A real depth buffer is what makes the assembly draw correctly:
        # a colour-only FBO leaves glDepthTest comparing against uninitialised
        # memory, and the engine comes out as a flat dark silhouette.
        self._depth = self.ctx.depth_renderbuffer((width, height))
        self._fbo = self.ctx.framebuffer(color_attachments=[self._tex], depth_attachment=self._depth)

    def frame_camera(self) -> None:
        lo, hi = self.scene.bounds()
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        self._center = 0.5 * (lo + hi)
        aspect = self.width / max(self.height, 1)
        self._base_distance = gpu.fit_distance(lo, hi, self._fov(), aspect, fill=0.80)
        self.zoom = 1.0

    def _fov(self) -> float:
        return math.radians(42.0)

    # ------------------------------------------------------------------ #
    # input
    # ------------------------------------------------------------------ #
    def orbit(self, dx: float, dy: float) -> None:
        # screen-space drags: moving the mouse right must spin the engine to
        # the right (azimuth grows counter-clockwise, hence minus). Vertical
        # is the DCC trackball convention: drag down = orbit under, i.e. the
        # camera rises (elevation grows), because screen y grows downward.
        self.azimuth -= dx * 0.008
        self.elevation = max(-1.45, min(1.45, self.elevation + dy * 0.008))

    def zoom_by(self, factor: float) -> None:
        self.zoom = max(0.35, min(3.2, self.zoom * factor))

    # ------------------------------------------------------------------ #
    # per frame
    # ------------------------------------------------------------------ #
    def draw(self, width: float, height: float, model=None) -> int:
        """Render the assembly and return the texture id for imgui.

        The offscreen pass *borrows* the GL state: whichever framebuffer was
        bound before (the host's back buffer) is restored afterwards, because
        hello_imgui renders the whole interface into it once our callback
        returns. Forgetting to restore it leaves the entire UI rendering into
        this FBO -- the classic "my window went black" imgui+moderngl bug.

        ``model`` animates the movers: without it the crank, pistons and cams
        sit frozen at their build angle, which is exactly how the first live
        screenshots looked (the capture path always passed a model).
        """
        if model is not None:
            self.scene.update(model)
        self._resize(int(width), int(height))
        import os as _os

        if _os.environ.get("ENGINELAB_DEBUG"):
            if not self._error_shown:
                self._error_shown = True
                r = self.renderer
                static_tris = getattr(r, "_static_count", -1)
                movers = len(r._mover_vaos)
                print(
                    f"[viewport] ctx={self.ctx.version_code} error={r.error} "
                    f"static_tris={static_tris} mover_vaos={movers} "
                    f"fbo={self._fbo.size if self._fbo else None}",
                    flush=True,
                )
        w, h = self.width, self.height
        aspect = w / max(h, 1)
        distance = self._base_distance / max(self.zoom, 1e-3)
        eye = gpu.orbit_eye(self._center, distance, self.azimuth, self.elevation)
        view = gpu.look_at(eye, self._center)
        proj = gpu.perspective(self._fov(), aspect, 0.02, 30.0)
        view_proj = proj @ view

        ctx = self.ctx
        screen = ctx.screen  # None in a standalone context (no window)
        self._fbo.use()
        ctx.viewport = (0, 0, w, h)
        ctx.clear(0.043, 0.059, 0.082, 1.0)
        ctx.enable(ctx.DEPTH_TEST)
        # NOTE: no face culling here. The winding convention of the mesh
        # builder is *outward*, but a single global winding can still be
        # back-facing depending on the bank tilt transform, and culling in
        # the shared context made the whole assembly render as a flat
        # silhouette in the window (while the standalone capture looked
        # fine). The depth buffer already handles occlusion; culling is only
        # a micro-optimization for this face count.
        stats = self.renderer.draw(view_proj, eye, ghost_shell=self.show_shell)
        ctx.disable(ctx.DEPTH_TEST)
        # hand the back buffer back, exactly as we found it -- but a
        # standalone context (used by --capture) has no screen framebuffer,
        # and the FBO stays bound on purpose: capture() reads it directly.
        if screen is not None:
            screen.use()

        import os as _os

        if _os.environ.get("ENGINELAB_DUMP_FBO") and not self._error_shown:
            self._error_shown = True
            data = self._fbo.read(components=3, alignment=1)
            from PIL import Image

            img = Image.frombytes("RGB", (w, h), data)
            img.save("docs/preview/fbo_direct_dump.png")
            print(f"[viewport] dumped FBO {w}x{h}", flush=True)
        return stats.get("faces_drawn", 0)

    def texture_id(self) -> int:
        return self._tex.glo

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def switch_scene(self, spec, segments: int = 12) -> None:
        self.renderer.release()
        self.scene = EngineScene(spec, segments=segments)
        self.renderer = GpuEngineRenderer(self.ctx, self.scene, segments=segments)
        self.frame_camera()

    def release(self) -> None:
        self.renderer.release()
        if self._fbo is not None:
            try:
                self._fbo.release()
            except Exception:
                pass
        if self._tex is not None:
            try:
                self._tex.release()
            except Exception:
                pass
        if self._depth is not None:
            try:
                self._depth.release()
            except Exception:
                pass
        try:
            self.ctx.release()
        except Exception:
            pass


def moderngl_cull(ctx):
    return ctx.CULL_FACE

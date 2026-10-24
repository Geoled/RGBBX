"""The engine catalogue: a modal browser with a preview and a spec sheet.

The old picker was a combo box: eighteen near-identical strings. This window
makes the *list* the interface -- a large scrollable grid of cards, one per
engine, each with its own live 3D thumbnail rendered off-screen by the same
GPU pipeline the main viewport uses. A click inspects, a double click loads.

Layout (master-detail, list-dominant)::

    +-------------------------------------------------------------+-----------+
    | ENGINE CATALOGUE          click: inspect · dbl: load   [x] |           |
    +-------------------------------------------------------------+  detail   |
    |                                                             |  column  |
    |   [card] [card] [card] [card] [card] [card]                |           |
    |                                                             |  3D view  |
    |   [card] [card] [card] [card] [card] [card]                 |           |
    |                                                             |  name     |
    |   ...scrolls...                                             |  figures  |
    |                                                             |  tag      |
    |                                                             |  [LOAD]  |
    +-------------------------------------------------------------+-----------+

The list gets ~75% of the window because choosing an engine is *comparing*
engines; the detail column only has to answer "what am I looking at".
Thumbnails render once per engine the first time the catalogue opens, never
per frame, so the browser costs nothing while closed.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from imgui_bundle import imgui

from ..physics import EngineSpec
from ..view import gpu
from ..view.engine3d import EngineScene
from ..view.gpu_scene import GpuEngineRenderer
from . import theme, widgets

# window geometry, pixels
WIN_W = 1180.0
WIN_H = 760.0
LIST_FRACTION = 0.74  # of the inner width
HEADER_H = 44.0
DETAIL_PAD = 16.0
THUMB_W = 168.0
THUMB_H = 112.0
CARD_TEXT_H = 30.0
CARD_GAP = 12.0


def _spec_from_preset(preset):
    return EngineSpec.from_preset(preset)


def _character_tag(spec: EngineSpec) -> str:
    """A one-word read of what this engine is like."""
    if spec.kind == "turbine":
        return "TURBOSHAFT"
    if spec.layout == "rotary":
        return "WANKEL"
    if spec.supercharged:
        return "SUPERCHARGED"
    if spec.boost_bar >= 1.0:
        return "HEAVILY BOOSTED"
    if spec.boost_bar > 0.0:
        return "TURBO"
    if spec.cylinders >= 8:
        return "BIG-BORE V8" if spec.layout == "v" else "MULTI-CYL"
    if spec.redline_rpm >= 8000:
        return "HIGH-REVING NA"
    if spec.kind == "diesel":
        return "OIL BURNER"
    return "NATURALLY ASPIRATED"


class _Thumb:
    """One offscreen thumbnail: scene + renderer + FBO, rendered once."""

    __slots__ = ("ctx", "scene", "renderer", "fbo", "tex", "depth", "view_proj", "eye")

    def __init__(self, ctx, spec, w: int, h: int) -> None:
        self.ctx = ctx
        w, h = int(w), int(h)
        self.scene = EngineScene(spec, segments=10)
        self.renderer = GpuEngineRenderer(ctx, self.scene, segments=10)
        lo, hi = self.scene.bounds()
        centre = 0.5 * (np.asarray(lo) + np.asarray(hi))
        aspect = w / h
        distance = gpu.fit_distance(lo, hi, math.radians(42.0), aspect, fill=0.88)
        azimuth, elevation = 0.85, 0.50
        self.eye = gpu.orbit_eye(centre, distance, azimuth, elevation)
        view = gpu.look_at(self.eye, centre)
        proj = gpu.perspective(math.radians(42.0), aspect, 0.02, 30.0)
        self.view_proj = proj @ view
        self.tex = ctx.texture((w, h), 4)
        self.tex.repeat_x = self.tex.repeat_y = False
        self.depth = ctx.depth_renderbuffer((w, h))
        self.fbo = ctx.framebuffer(color_attachments=[self.tex], depth_attachment=self.depth)
        self._draw(w, h)

    def _draw(self, w: int, h: int) -> None:
        ctx = self.ctx
        screen = ctx.screen  # None in a standalone context (no window)
        self.fbo.use()
        ctx.viewport = (0, 0, w, h)
        ctx.clear(0.055, 0.075, 0.105, 1.0)
        ctx.enable(ctx.DEPTH_TEST)
        self.renderer.draw(self.view_proj, self.eye)
        ctx.disable(ctx.DEPTH_TEST)
        if screen is not None:
            screen.use()

    def release(self) -> None:
        for obj in (self.fbo, self.tex, self.depth):
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        try:
            self.renderer.release()
        except Exception:
            pass


class EngineCatalogue:
    """The modal engine browser. Owns one thumbnail per preset.

    Two frame-budget rules keep the catalogue from stuttering the app:

    * **Spec data is cached.** Loading a preset TOML and deriving its
      :class:`EngineSpec` per card per frame is 23 file reads a frame; the
      results never change, so they are parsed once per open.
    * **Thumbnails render a few per frame, not all at once.** Rendering 23
      engine scenes in the first frame after ``show()`` would freeze the
      window for hundreds of milliseconds (and starve the audio thread,
      which is what makes the sound tear during the opening). A small queue
      renders one thumbnail per frame; cards show a placeholder until
      theirs is ready -- the difference between a 16 ms frame and a 400 ms
      freeze.
    """

    # how many thumbnails to render per frame once the catalogue is open
    RENDER_PER_FRAME = 2

    def __init__(self) -> None:
        self.open = False
        self.selected = 0  # index into presets, for the detail column
        self._thumbs: dict = {}  # (index, w, h) -> _Thumb
        self._specs: dict = {}  # index -> (preset, spec), parsed once
        self._queue: deque = deque()  # indices still needing their thumbnail
        self._queued: set = set()  # dedupe of _queue
        self._ctx = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def show(self, state) -> None:
        import moderngl

        self.selected = state.preset_index
        self.open = True
        if self._ctx is None:
            # borrow the window's GL context: the textures must live in it
            self._ctx = moderngl.create_context()
        # (re)build the render queue lazily in _draw_body once specs are known

    def close(self) -> None:
        self.open = False
        for thumb in self._thumbs.values():
            thumb.release()
        self._thumbs.clear()
        self._specs.clear()
        self._queue.clear()
        self._queued.clear()

    def toggle(self, state) -> None:
        if self.open:
            self.close()
        else:
            self.show(state)

    # ------------------------------------------------------------------ #
    # cached spec data
    # ------------------------------------------------------------------ #
    def _spec(self, state, index: int):
        """The preset and its derived spec, parsed once per catalogue open."""
        cached = self._specs.get(index)
        if cached is not None:
            return cached
        from soundgen.config import load_preset

        try:
            preset = load_preset(str(state.presets[index]))
            spec = _spec_from_preset(preset)
        except Exception:
            return (None, None)
        cached = (preset, spec)
        self._specs[index] = cached
        return cached

    # ------------------------------------------------------------------ #
    # thumbnails
    # ------------------------------------------------------------------ #
    def _thumb(self, state, index: int, w: float, h: float, render: bool = False) -> _Thumb | None:
        """The thumbnail for one engine, or None while it is still queued.

        With ``render=False`` this only *looks* -- scheduling the actual GL
        work into the per-frame budget -- so drawing never triggers a
        several-hundred-ms stall.
        """
        key = (index, int(w), int(h))
        thumb = self._thumbs.get(key)
        if thumb is not None:
            return thumb
        if not render:
            self._enqueue(index, int(w), int(h))
            return None
        _, spec = self._spec(state, index)
        if spec is None:
            return None
        thumb = _Thumb(self._ctx, spec, int(w), int(h))
        self._thumbs[key] = thumb
        return thumb

    def _enqueue(self, index: int, w: int, h: int) -> None:
        entry = (index, w, h)
        if entry in self._queued:
            return
        self._queued.add(entry)
        self._queue.append(entry)

    def _pump(self, state) -> None:
        """Render up to RENDER_PER_FRAME queued thumbnails this frame.

        Called once per frame from draw(): keeps each frame inside its
        16 ms budget instead of paying for the whole grid up front.
        """
        for _ in range(self.RENDER_PER_FRAME):
            if not self._queue:
                return
            index, w, h = self._queue.popleft()
            self._queued.discard((index, w, h))
            self._thumb(state, index, w, h, render=True)

    # ------------------------------------------------------------------ #
    # the window
    # ------------------------------------------------------------------ #
    def draw(self, app, state) -> None:
        if not self.open:
            return
        io = imgui.get_io()
        display_w, display_h = io.display_size.x, io.display_size.y
        win_x = max((display_w - WIN_W) * 0.5, 8.0)
        win_y = max((display_h - WIN_H) * 0.5, 8.0)

        imgui.set_next_window_pos(imgui.ImVec2(win_x, win_y), imgui.Cond_.appearing)
        imgui.set_next_window_size(imgui.ImVec2(WIN_W, WIN_H), imgui.Cond_.appearing)
        imgui.set_next_window_focus()
        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_collapse
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_saved_settings
            | imgui.WindowFlags_.no_scrollbar
        )
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 6.0)
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 1.0)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(0, 0))
        imgui.push_style_color(imgui.Col_.window_bg, theme.PANEL_B)
        imgui.push_style_color(imgui.Col_.border, theme.EDGE_HOT)
        imgui.begin("engine catalogue", None, flags)
        widgets._push_window()
        imgui.pop_style_color(2)
        imgui.pop_style_var(3)
        try:
            if imgui.is_key_pressed(imgui.Key.escape):
                self.close()
                return
            self._draw_body(app, state)
            # spread the remaining thumbnail renders over the next frames
            self._pump(state)
        finally:
            widgets._pop_window()
            imgui.end()

    # ------------------------------------------------------------------ #
    def _draw_body(self, app, state) -> None:
        dl = imgui.get_window_draw_list()
        win_pos = imgui.get_window_pos()
        win_size = imgui.get_window_size()
        presets = state.presets
        n = len(presets)
        if n == 0 or self._ctx is None:
            return
        self.selected = max(0, min(self.selected, n - 1))

        inner_w = win_size.x - 2.0  # border
        list_w = (inner_w - DETAIL_PAD) * LIST_FRACTION
        detail_w = inner_w - list_w - DETAIL_PAD

        # --- header strip ------------------------------------------------- #
        header_a = imgui.ImVec2(win_pos.x + 1, win_pos.y + 1)
        header_b = imgui.ImVec2(win_pos.x + win_size.x - 1, win_pos.y + HEADER_H)
        dl.add_rect_filled(header_a, header_b, widgets._color_u32(theme.PANEL_RAISED), 0.0)
        dl.add_line(
            imgui.ImVec2(header_a.x, header_b.y - 1),
            imgui.ImVec2(header_b.x, header_b.y - 1),
            widgets._color_u32(theme.EDGE_HOT),
        )
        widgets.label_at(imgui.ImVec2(win_pos.x + 18, win_pos.y + 15), "engine catalogue", theme.INK)
        hint = "click: inspect · double-click: load · esc: close"
        widgets.label_at(imgui.ImVec2(win_pos.x + 210, win_pos.y + 17), hint, theme.INK_FAINT)

        # close cross, inside the catalogue's own header strip
        imgui.set_cursor_screen_pos(imgui.ImVec2(win_pos.x + win_size.x - 42.0, win_pos.y + 9))
        imgui.push_style_color(imgui.Col_.text, theme.INK_DIM)
        imgui.push_style_color(imgui.Col_.button, (0, 0, 0, 0))
        imgui.push_style_color(imgui.Col_.button_hovered, theme.with_alpha(theme.RED, 0.25))
        imgui.push_style_color(imgui.Col_.button_active, theme.with_alpha(theme.RED, 0.4))
        if imgui.button("x##cat_close", imgui.ImVec2(26.0, 26.0)):
            self.close()
        imgui.pop_style_color(4)

        body_top = win_pos.y + HEADER_H
        body_h = win_size.y - HEADER_H - 1.0

        # --- the list: most of the window, scrollable ---------------------- #
        list_x = win_pos.x + 1.0
        imgui.set_cursor_screen_pos(imgui.ImVec2(list_x, body_top))
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(12, 12))
        imgui.push_style_var(imgui.StyleVar_.scrollbar_size, 10.0)
        imgui.begin_child(
            "##cat_list",
            imgui.ImVec2(list_w, body_h),
            imgui.ChildFlags_.none,
            imgui.WindowFlags_.no_scroll_with_mouse if False else 0,
        )
        imgui.pop_style_var(2)
        try:
            area = imgui.get_cursor_screen_pos()
            avail_w = list_w - 24.0
            per_row = max(1, int(avail_w // (THUMB_W + CARD_GAP)))
            step_x = (avail_w - per_row * THUMB_W) / max(per_row - 1, 1) + THUMB_W if per_row > 1 else THUMB_W
            step_y = THUMB_H + CARD_TEXT_H + CARD_GAP
            for i in range(n):
                row, col = divmod(i, per_row)
                cx = area.x + col * step_x
                cy = area.y + row * step_y
                self._draw_card(app, state, i, cx, cy)
        finally:
            imgui.end_child()

        # --- the detail column (right) -------------------------------------- #
        self._draw_detail(app, state, list_x + list_w + DETAIL_PAD, body_top, detail_w, body_h)

    # ------------------------------------------------------------------ #
    def _draw_detail(self, app, state, x: float, y: float, w: float, h: float) -> None:
        dl = imgui.get_window_draw_list()
        preset, spec = self._spec(state, self.selected)
        if preset is None or spec is None:
            return

        # column backdrop
        a = imgui.ImVec2(x, y)
        b = imgui.ImVec2(x + w - 1.0, y + h)
        dl.add_rect_filled(a, b, widgets._color_u32(theme.PANEL_A, 0.55), 0.0)
        dl.add_line(imgui.ImVec2(a.x, a.y), imgui.ImVec2(a.x, b.y), widgets._color_u32(theme.EDGE))

        # 3D preview, square, at the top of the column
        preview_side = w - 24.0
        thumb = self._thumb(state, self.selected, preview_side, preview_side)
        if thumb is not None:
            imgui.set_cursor_screen_pos(imgui.ImVec2(x + 12.0, y + 12.0))
            imgui.image(
                imgui.ImTextureRef(thumb.tex.glo),
                imgui.ImVec2(preview_side, preview_side),
                imgui.ImVec2(0, 1),
                imgui.ImVec2(1, 0),
            )

        # name under the preview
        text_y = y + preview_side + 24.0
        name = preset.name.upper()
        if len(name) > 24:
            name = name[:22] + "..."
        dl.add_text(imgui.ImVec2(x + 12.0, text_y), widgets._color_u32(theme.INK), name)
        text_y += 26.0

        # figures, two per line to stay narrow
        rows = [
            ("layout", f"{spec.layout} {spec.cylinders}cyl"),
            ("displacement", f"{spec.displacement * 1000:.1f} L"),
            ("bore x stroke", f"{spec.bore * 1000:.0f}x{spec.stroke * 1000:.0f}"),
            ("redline", f"{spec.redline_rpm:.0f} rpm"),
            ("idle", f"{spec.idle_rpm:.0f} rpm"),
            (
                "induction",
                (f"{spec.boost_bar:.1f} bar" if spec.boost_bar > 0 else "atmo")
                + (" +sc" if spec.supercharged else ""),
            ),
        ]
        for label_text, value in rows:
            widgets.label_at(imgui.ImVec2(x + 12.0, text_y), label_text, theme.INK_FAINT)
            vw = imgui.calc_text_size(value).x
            dl.add_text(
                imgui.ImVec2(x + w - 12.0 - vw, text_y),
                widgets._color_u32(theme.INK),
                value,
            )
            text_y += 21.0

        # character tag
        text_y += 6.0
        imgui.set_cursor_screen_pos(imgui.ImVec2(x + 12.0, text_y))
        widgets.chip(_character_tag(spec), color=theme.ORANGE, lit=True)
        text_y += 30.0

        # load button, pinned near the bottom
        btn_h = 36.0
        btn_y = y + h - btn_h - 14.0
        imgui.set_cursor_screen_pos(imgui.ImVec2(x + 12.0, btn_y))
        is_current = self.selected == state.preset_index
        imgui.push_style_color(imgui.Col_.button, theme.ORANGE if not is_current else theme.PANEL_RAISED)
        imgui.push_style_color(
            imgui.Col_.text, (0.05, 0.06, 0.08, 1.0) if not is_current else theme.INK_DIM
        )
        imgui.push_style_var(imgui.StyleVar_.frame_rounding, 4.0)
        label_text = "LOADED" if is_current else "LOAD THIS ENGINE"
        if imgui.button(label_text + "##cat_select", imgui.ImVec2(w - 24.0, btn_h)):
            if not is_current:
                app.select_preset(self.selected)
                self.close()
        imgui.pop_style_var()
        imgui.pop_style_color(2)

    # ------------------------------------------------------------------ #
    def _draw_card(self, app, state, index: int, x: float, y: float) -> None:
        dl = imgui.get_window_draw_list()
        preset, spec = self._spec(state, index)
        if preset is None or spec is None:
            return
        is_current = index == state.preset_index
        is_selected = index == self.selected

        a = imgui.ImVec2(x, y)
        b = imgui.ImVec2(x + THUMB_W, y + THUMB_H + CARD_TEXT_H)

        # thumbnail (None while still queued: placeholder below)
        thumb = self._thumb(state, index, THUMB_W, THUMB_H)
        if thumb is not None:
            imgui.set_cursor_screen_pos(a)
            imgui.image(
                imgui.ImTextureRef(thumb.tex.glo),
                imgui.ImVec2(THUMB_W, THUMB_H),
                imgui.ImVec2(0, 1),
                imgui.ImVec2(1, 0),
            )
        else:
            # still rendering: a sunk placeholder so the grid does not
            # reflow when the pictures pop in one by one
            dl.add_rect_filled(a, imgui.ImVec2(x + THUMB_W, y + THUMB_H), widgets._color_u32(theme.PANEL_SUNK), 0.0)

        # caption band
        band_a = imgui.ImVec2(x, y + THUMB_H)
        dl.add_rect_filled(band_a, b, widgets._color_u32(theme.PANEL_RAISED, 0.85), 0.0)
        name = preset.name
        if len(name) > 24:
            name = name[:22] + "..."
        widgets.label_at(
            imgui.ImVec2(x + 8.0, y + THUMB_H + 5.0),
            name,
            theme.INK if is_selected else theme.INK_DIM,
        )
        sub = f"{spec.displacement * 1000:.1f}L · {spec.redline_rpm:.0f}rpm"
        widgets.label_at(imgui.ImVec2(x + 8.0, y + THUMB_H + 18.0), sub, theme.INK_FAINT)

        # frame: orange = inspected, lit edge = loaded
        border = theme.ORANGE if is_selected else (theme.BLUE if is_current else theme.EDGE)
        dl.add_rect(a, b, widgets._color_u32(border), 0.0)
        if is_current:
            # small 'LOADED' pip in the top-right corner of the thumb
            pip_a = imgui.ImVec2(x + THUMB_W - 52.0, y + 4.0)
            pip_b = imgui.ImVec2(x + THUMB_W - 4.0, y + 18.0)
            dl.add_rect_filled(pip_a, pip_b, widgets._color_u32(theme.with_alpha(theme.BLUE, 0.8)), 2.0)
            dl.add_text(imgui.ImVec2(pip_a.x + 5.0, pip_a.y + 2.0), widgets._color_u32((0, 0, 0, 1)), "CUR")

        # click target over the whole card
        imgui.set_cursor_screen_pos(a)
        clicked = imgui.invisible_button(f"##card{index}", imgui.ImVec2(THUMB_W, THUMB_H + CARD_TEXT_H))
        if imgui.is_item_hovered():
            dl.add_rect(a, b, widgets._color_u32(theme.INK, 0.30), 0.0)
        if clicked:
            if imgui.is_mouse_double_clicked(0):
                app.select_preset(index)
                self.close()
            elif not imgui.is_mouse_double_clicked(0):
                self.selected = index

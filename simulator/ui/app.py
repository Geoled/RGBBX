"""The simulator window.

Layout, top to bottom: a title strip, the live 3D cutaway beside the gauge
cluster, the dyno chart, and a control card.

Everything graphical -- including every control -- is drawn with Pillow and
pasted onto plain ``tk.Canvas`` widgets, so the whole window is one consistent
surface. tkinter owns only the top-level window, the keyboard bindings and the
preset dropdown menu; sliders, buttons and switches are drawn here and hit-
tested here, which is the only way to get a Windows 11 look out of tkinter.
"""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk

from ..bootstrap import preset_dir
from ..physics import TWO_PI, EngineModel, EngineSpec
from ..view.engine3d import EngineScene
from ..view.raster import SoftwareRenderer, frame_camera
from . import dyno as dyno_mod
from . import theme
from . import widgets

PAD = 10
HEADER_H = 44
VIEW_W, VIEW_H = 700, 440
GAUGE_W, GAUGE_H = 476, 440
WIN_W = PAD + VIEW_W + PAD + GAUGE_W + PAD
DYNO_H = 176
CTRL_H = 156

VIEW_INSET = (10, 8)
INNER_W = VIEW_W - 2 * VIEW_INSET[0]
INNER_H = VIEW_H - 2 * VIEW_INSET[1]

# 60 fps is the target; the delay below is a cap, not a promise, so a slower
# machine simply settles at whatever it can draw.
TARGET_FRAME = 1.0 / 60.0
GAUGE_EVERY = 2
HEADER_EVERY = 6

# --------------------------------------------------------------------------- #
# control geometry (panel coordinates inside the control card)
# --------------------------------------------------------------------------- #
COMBO_RECT = (18, 34, 330, 76)
PULL_RECT = (342, 36, 462, 74)
THROTTLE_RECT = (482, 34, 792, 76)
BLIP_RECT = (804, 36, 892, 74)
VOLUME_RECT = (912, 34, 1102, 76)

CRANK_RECT = (18, 100, 138, 138)
IGNITION_RECT = (150, 100, 318, 138)
BRAKE_RECT = (338, 100, 648, 138)
LIMITER_RECT = (668, 100, 816, 138)

SLIDER_TRAVEL = 13


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % rgb


class SimulatorApp:
    def __init__(
        self,
        root: tk.Tk,
        preset_path: str | None = None,
        audio: bool = True,
        audio_backend: str = "auto",
        audio_device: int | str | None = None,
    ) -> None:
        self.root = root
        self.audio_wanted = bool(audio)
        self.audio_backend = str(audio_backend or "auto")
        self.audio_device = audio_device
        from soundgen.config import discover_presets

        self.preset_paths = discover_presets(preset_dir())
        if not self.preset_paths:
            raise SystemExit("no presets found")

        self.model: EngineModel | None = None
        self.spec: EngineSpec | None = None
        self.preset = None
        self.scene: EngineScene | None = None
        self.camera = None
        self.renderer: SoftwareRenderer | None = None
        self.audio = None

        self.throttle_cmd = 0.0
        self.brake_cmd = 0.0
        self.cranking = False
        self.limiter_on = True
        self.ignition_on = True
        self.volume = 0.85

        self._auto_crank = 0.0
        self._dyno_state: str | None = None
        self._dyno_settle = 0.0
        self._dyno_points: list[tuple[float, float]] = []
        self._dyno_progress = 0.0
        self._dyno_message = "press RUN PULL to measure this engine"
        self._dyno_signature = None
        self._backfire_flash = 0.0
        self._knock_flash = 0.0
        self._was_running = False
        self._was_cranking = False

        self.status_text = "starting up"
        self._hover: str | None = None
        self._press: str | None = None
        self._drag: str | None = None
        self._orbit_last = None

        self._frame = 0
        self._fps = 0.0
        self._fps_clock = time.perf_counter()
        self._fps_count = 0
        self._last_time = time.perf_counter()
        self._closing = False

        root.title("Engine Lab - simulator")
        root.configure(bg=_hex(theme.BG))
        root.resizable(False, False)

        self._build_chrome()
        self._select_preset(self._index_of(preset_path), first=True)
        self._bind_keys()
        self._draw_controls()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._tick()

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _canvas(self, root, w: int, h: int) -> tuple[tk.Canvas, Image.Image, ImageTk.PhotoImage]:
        image = Image.new("RGB", (w, h), theme.BG)
        photo = ImageTk.PhotoImage(image)
        canvas = tk.Canvas(
            root, width=w, height=h, highlightthickness=0, bd=0, bg=_hex(theme.BG)
        )
        canvas.create_image(0, 0, anchor="nw", image=photo)
        return canvas, image, photo

    def _build_chrome(self) -> None:
        root = self.root

        self.header_canvas, self.header_image, self.header_photo = self._canvas(
            root, WIN_W, HEADER_H
        )
        self.header_canvas.grid(row=0, column=0, columnspan=2, padx=PAD, pady=(PAD, 0))

        self.view_canvas, self.view_image, self.view_photo = self._canvas(root, VIEW_W, VIEW_H)
        self.view_canvas.grid(row=1, column=0, padx=(PAD, 0), pady=(PAD, 0))

        self.gauge_canvas, self.gauge_image, self.gauge_photo = self._canvas(
            root, GAUGE_W, GAUGE_H
        )
        self.gauge_canvas.grid(row=1, column=1, padx=(PAD, PAD), pady=(PAD, 0))

        self.dyno_canvas, self.dyno_image, self.dyno_photo = self._canvas(root, WIN_W, DYNO_H)
        self.dyno_canvas.grid(row=2, column=0, columnspan=2, padx=PAD, pady=(PAD, 0))

        self.ctrl_canvas, self.ctrl_image, self.ctrl_photo = self._canvas(root, WIN_W, CTRL_H)
        self.ctrl_canvas.grid(row=3, column=0, columnspan=2, padx=PAD, pady=(PAD, PAD))

        self._header_bg = theme.panel((WIN_W, HEADER_H), (8, 4, WIN_W - 8, HEADER_H - 4), radius=14)
        self._view_bg = theme.panel((VIEW_W, VIEW_H), (8, 6, VIEW_W - 8, VIEW_H - 6), radius=22)
        self._gauge_bg = theme.panel((GAUGE_W, GAUGE_H), (8, 6, GAUGE_W - 8, GAUGE_H - 6), radius=22)
        self._dyno_bg = theme.panel((WIN_W, DYNO_H), (8, 6, WIN_W - 8, DYNO_H - 6), radius=22)
        self._ctrl_bg = theme.panel((WIN_W, CTRL_H), (8, 6, WIN_W - 8, CTRL_H - 6), radius=22)

        for column in (0, 1):
            root.grid_columnconfigure(column, weight=0)

    # ------------------------------------------------------------------ #
    # presets
    # ------------------------------------------------------------------ #
    @staticmethod
    def _preset_name(path: Path) -> str:
        from soundgen.config import load_preset

        try:
            return load_preset(str(path)).name
        except Exception:
            return path.stem

    def _index_of(self, preset_path: str | None) -> int:
        if not preset_path:
            # the naturally aspirated petrol four is the reference engine
            for i, path in enumerate(self.preset_paths):
                if path.stem == "i4_gasoline":
                    return i
            return 0
        wanted = Path(preset_path).name
        for i, path in enumerate(self.preset_paths):
            if path.name == wanted or path.stem == Path(preset_path).stem:
                return i
        return 0

    def _select_preset(self, index: int, first: bool = False) -> None:
        from soundgen.config import load_preset

        self.root.config(cursor="watch")
        self.status_text = "loading engine and calibrating audio..."
        self._draw_header()
        self.root.update_idletasks()

        path = self.preset_paths[index]
        preset = load_preset(str(path))
        spec = EngineSpec.from_preset(preset)
        self.preset = preset
        self.spec = spec
        self.model = EngineModel(spec)
        self.scene = EngineScene(spec, segments=12)
        self.renderer = SoftwareRenderer(
            INNER_W,
            INNER_H,
            background_top=theme.VIEW_TOP,
            background_bottom=theme.VIEW_BOTTOM,
            light=(0.36, 0.74, 0.57),
            ambient=0.46,
            shininess=30.0,
            specular=0.22,
            shadow_color=(96, 110, 136),
            shadow_strength=0.42,
        )
        lo, hi = self.scene.bounds()
        self.camera = frame_camera(lo, hi, INNER_W, INNER_H)

        self._dyno_points = []
        self._dyno_state = None
        self._dyno_progress = 0.0
        self._dyno_message = "press RUN PULL to measure this engine"
        self._dyno_signature = None
        self._auto_crank = 1.3
        self._was_running = False
        self._was_cranking = False
        self.cranking = False
        self.ignition_on = True
        self._knock_flash = 0.0
        self.brake_cmd = 0.0
        # a pull pins the throttle wide open, so switching engines during one
        # would otherwise hand the new engine a pedal it never asked for
        self.throttle_cmd = 0.0

        if self.audio is not None:
            self.audio.stop()
            self.audio = None
        self._audio_token = getattr(self, "_audio_token", 0) + 1
        if self.audio_wanted:
            # Build the audio in a worker: constructing EngineAudio calibrates
            # the gain and renders the one-shot bank, which costs 1-4 s on the
            # big presets. On the main thread that froze the whole window on
            # every engine switch -- the "everything lags" report. The Qt GUI
            # already opens in a thread; this is the same shape for Tk.
            self.root.config(cursor="watch")
            self._open_audio_async(preset)
        else:
            self.status_text = "audio disabled (--no-audio)"

        self.root.config(cursor="")
        if not first and self.audio_wanted and self.audio is not None and self.audio.enabled:
            self.status_text = f"loaded {preset.name}"
        self._draw_gauges()
        self._draw_dyno()
        self._draw_header()

    def _open_audio_async(self, preset) -> None:
        """Open EngineAudio off the main thread; keep the UI responsive.

        The switch handler stops the old stream immediately (audio gap is at
        most the calibration + shot-bank time, but the *window* stays live and
        the gauges keep drawing -- that is the difference the user hears as
        lag). The worker publishes the finished object back on the Tk thread
        via ``after``, and a stale worker (superseded by another switch) is
        discarded without ever reaching the stream.
        """
        import threading

        def done(audio, error):
            self.audio = audio
            self.root.config(cursor="")
            if error:
                self.status_text = f"audio error: {error}"
            elif audio is not None and not audio.enabled:
                self.status_text = f"audio unavailable: {audio.error}"
            elif audio is not None and audio.warnings:
                self.status_text = (
                    f"audio: {audio.backend} (fallback: {audio.warnings[-1]})"
                )
            elif audio is not None:
                self.status_text = f"loaded {preset.name}"

        def worker(token: list):
            from ..audio import EngineAudio

            try:
                # Asset mode: constant ~40 us playback callbacks, one small
                # uniform block for every engine (see gui/app.py note).
                audio = EngineAudio(
                    preset,
                    sr=48000,
                    block_frames=128,
                    buffers=3,
                    master=self.volume,
                    audio_backend=self.audio_backend,
                    audio_device=self.audio_device,
                    mode="assets",
                )
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda: done(None, f"{type(exc).__name__}: {exc}"))
                return
            if token[0] != self._audio_token:
                audio.stop()  # a newer switch superseded this worker
                return
            self.root.after(0, lambda: done(audio, None))

        self._audio_token = getattr(self, "_audio_token", 0) + 1
        threading.Thread(
            target=worker, args=([self._audio_token],), daemon=True
        ).start()
        self._draw_controls()

    def _open_preset_menu(self) -> None:
        menu = tk.Menu(
            self.root,
            tearoff=0,
            font=("Segoe UI", 10),
            bg="#ffffff",
            fg=_hex(theme.INK),
            activebackground=_hex(theme.ACCENT),
            activeforeground="#ffffff",
            bd=0,
        )
        for i, path in enumerate(self.preset_paths):
            menu.add_command(label=self._preset_name(path), command=lambda i=i: self._select_preset(i))
        x = self.ctrl_canvas.winfo_rootx() + COMBO_RECT[0]
        y = self.ctrl_canvas.winfo_rooty() + COMBO_RECT[3] + 2
        try:
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    # ------------------------------------------------------------------ #
    # controls
    # ------------------------------------------------------------------ #
    _SLIDERS = {"throttle": THROTTLE_RECT, "brake": BRAKE_RECT, "volume": VOLUME_RECT}

    def _hit_key(self, x: int, y: int) -> str | None:
        def inside(rect):
            return rect[0] - 4 <= x <= rect[2] + 4 and rect[1] - 8 <= y <= rect[3] + 8

        if inside(COMBO_RECT):
            return "engine"
        if inside(PULL_RECT):
            return "pull"
        if inside(THROTTLE_RECT):
            return "throttle"
        if inside(BLIP_RECT):
            return "blip"
        if inside(VOLUME_RECT):
            return "volume"
        if inside(CRANK_RECT):
            return "crank"
        if inside(IGNITION_RECT):
            return "ignition"
        if inside(BRAKE_RECT):
            return "brake"
        if inside(LIMITER_RECT):
            return "limiter"
        return None

    def _slider_fraction(self, key: str, x: int) -> float:
        x0, _, x1, _ = self._SLIDERS[key]
        lo, hi = x0 + SLIDER_TRAVEL, x1 - SLIDER_TRAVEL
        return theme.clamp01((x - lo) / max(hi - lo, 1))

    def _set_slider(self, key: str, x: int) -> None:
        value = self._slider_fraction(key, x)
        if key == "throttle":
            self.throttle_cmd = value
        elif key == "brake":
            self.brake_cmd = value
        else:
            self.volume = value
            if self.audio is not None:
                self.audio.set_volume(self.volume)

    def _ctrl_press(self, event) -> None:
        key = self._hit_key(event.x, event.y)
        self._press = key
        if key == "crank":
            self.cranking = True
            self.throttle_cmd = 0.0
        elif key in self._SLIDERS:
            self._drag = key
            self._set_slider(key, event.x)
        self._draw_controls()

    def _ctrl_motion(self, event) -> None:
        if self._drag is not None:
            self._set_slider(self._drag, event.x)
            self._draw_controls()
            return
        key = self._hit_key(event.x, event.y)
        if key != self._hover:
            self._hover = key
            self._draw_controls()

    def _ctrl_release(self, event) -> None:
        key = self._press
        self._press = None
        self._drag = None
        if self.cranking:
            self.cranking = False
        if key is not None and key == self._hit_key(event.x, event.y):
            self._activate(key)
        self._draw_controls()

    def _activate(self, key: str) -> None:
        if key == "blip":
            self._blip()
        elif key == "pull":
            self._start_dyno()
        elif key == "ignition":
            self.ignition_on = not self.ignition_on
            self._set_status("ignition " + ("on" if self.ignition_on else "off"))
        elif key == "limiter":
            self.limiter_on = not self.limiter_on
            self._set_status("limiter " + ("on" if self.limiter_on else "off"))
        elif key == "engine":
            self._open_preset_menu()

    def _state(self, key: str) -> str:
        if self._press == key:
            return "pressed"
        if self._hover == key:
            return "hover"
        return "normal"

    def _set_status(self, text: str) -> None:
        self.status_text = text
        self._draw_header()

    def _draw_controls(self) -> None:
        img = self._ctrl_bg.copy()
        draw = ImageDraw.Draw(img)
        spec = self.spec

        # --- engine selector ------------------------------------------- #
        sub = ""
        if spec is not None:
            kind = "turbine" if spec.kind == "turbine" else f"{spec.cylinders} cyl"
            sub = f"{kind}  ·  {spec.displacement * 1000:.1f} L  ·  {spec.redline_rpm:.0f} rpm"
        widgets.draw_combo(
            img,
            draw,
            COMBO_RECT,
            self.preset.name if self.preset is not None else "-",
            state=self._state("engine"),
            sub=sub,
        )
        widgets.draw_button(
            img,
            draw,
            PULL_RECT,
            "RUN PULL",
            primary=self._dyno_state is None,
            state=self._state("pull"),
            enabled=self._dyno_state is None,
        )

        # --- throttle --------------------------------------------------- #
        widgets.draw_caption(
            draw, THROTTLE_RECT[0], 18, "THROTTLE", f"{self.throttle_cmd * 100:.0f} %", right=THROTTLE_RECT[2]
        )
        widgets.draw_slider(
            img,
            draw,
            THROTTLE_RECT,
            self.throttle_cmd,
            accent=theme.ACCENT_2,
            hover=self._hover == "throttle",
        )
        widgets.draw_button(img, draw, BLIP_RECT, "BLIP", state=self._state("blip"))

        widgets.draw_caption(
            draw, VOLUME_RECT[0], 18, "VOLUME", f"{self.volume * 100:.0f} %", right=VOLUME_RECT[2]
        )
        widgets.draw_slider(
            img, draw, VOLUME_RECT, self.volume, accent=theme.ACCENT, hover=self._hover == "volume"
        )

        # --- starting --------------------------------------------------- #
        widgets.draw_button(
            img,
            draw,
            CRANK_RECT,
            "CRANKING" if self.cranking else "CRANK",
            primary=self.cranking,
            state=self._state("crank"),
        )
        widgets.draw_toggle(
            img, draw, IGNITION_RECT, self.ignition_on, "IGNITION", color=theme.ACCENT_2
        )

        # --- brake ------------------------------------------------------ #
        widgets.draw_caption(
            draw, BRAKE_RECT[0], 84, "BRAKE LOAD", f"{self.brake_cmd * 100:.0f} %", right=BRAKE_RECT[2]
        )
        widgets.draw_slider(
            img, draw, BRAKE_RECT, self.brake_cmd, accent=theme.WARM, hover=self._hover == "brake"
        )
        widgets.draw_toggle(img, draw, LIMITER_RECT, self.limiter_on, "LIMITER", color=theme.ACCENT)

        self.ctrl_image.paste(img)
        self.ctrl_photo.paste(self.ctrl_image)

    # ------------------------------------------------------------------ #
    # controls: keyboard
    # ------------------------------------------------------------------ #
    def _blip(self) -> None:
        self.throttle_cmd = 0.65
        self.root.after(260, self._blip_end)

    def _blip_end(self) -> None:
        if not self._dyno_state and self._press != "throttle" and self._drag != "throttle":
            self.throttle_cmd = 0.0

    def _start_dyno(self) -> None:
        if self._dyno_state or self.model is None:
            return
        self.model.start_dyno_pull()
        self.model.dyno.ramp_rpm_s = 0.0
        self._dyno_state = "settle"
        self._dyno_settle = 3.0
        self._dyno_points = []
        self._dyno_progress = 0.0
        self._dyno_message = ""
        self.throttle_cmd = 1.0
        self._set_status("dyno: settling the brake...")

    def _end_dyno(self, message: str) -> None:
        if self.model is not None and self.model.dyno.active:
            self.model.stop_dyno()
        self._dyno_state = None
        self._dyno_message = message
        self.throttle_cmd = 0.0
        self._set_status(message)

    # ------------------------------------------------------------------ #
    # input
    # ------------------------------------------------------------------ #
    def _bind_keys(self) -> None:
        root = self.root
        root.bind("<Escape>", lambda _e: self.close())
        root.bind("<space>", lambda _e: self._blip())
        root.bind("<KeyPress-s>", lambda _e: self._key_crank(True))
        root.bind("<KeyRelease-s>", lambda _e: self._key_crank(False))
        root.bind("<KeyPress-l>", lambda _e: self._activate("limiter"))
        root.bind("<KeyPress-d>", lambda _e: self._start_dyno())

        self.ctrl_canvas.bind("<ButtonPress-1>", self._ctrl_press)
        self.ctrl_canvas.bind("<B1-Motion>", self._ctrl_motion)
        self.ctrl_canvas.bind("<ButtonRelease-1>", self._ctrl_release)
        self.ctrl_canvas.bind("<Motion>", self._ctrl_motion)
        self.ctrl_canvas.bind("<Leave>", self._ctrl_leave)

        self.view_canvas.bind("<ButtonPress-1>", self._orbit_start)
        self.view_canvas.bind("<B1-Motion>", self._orbit_move)
        self.view_canvas.bind("<MouseWheel>", self._wheel)

    def _key_crank(self, down: bool) -> None:
        self.cranking = down
        if down:
            self.throttle_cmd = 0.0
        self._draw_controls()

    def _ctrl_leave(self, _event) -> None:
        if self._hover is not None:
            self._hover = None
            self._draw_controls()

    def _orbit_start(self, event) -> None:
        self._orbit_last = (event.x, event.y)

    def _orbit_move(self, event) -> None:
        last = self._orbit_last
        if last is None or self.camera is None:
            return
        dx = event.x - last[0]
        dy = event.y - last[1]
        self._orbit_last = (event.x, event.y)
        self.camera.orbit(dx * 0.010, -dy * 0.008)

    def _wheel(self, event) -> None:
        if self.camera is None:
            return
        self.camera.zoom(0.90 if event.delta > 0 else 1.1111)

    # ------------------------------------------------------------------ #
    # the loop
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        if self._closing:
            return
        now = time.perf_counter()
        dt = now - self._last_time
        self._last_time = now
        dt = min(max(dt, 0.0), 0.10)

        self._advance(dt)
        self._draw_view()

        self._frame += 1
        if self._frame % GAUGE_EVERY == 0:
            self._draw_gauges()
            self._draw_dyno()
        if self._frame % HEADER_EVERY == 0:
            self._draw_header()

        self._fps_count += 1
        if now - self._fps_clock >= 0.5:
            self._fps = self._fps_count / (now - self._fps_clock)
            self._fps_clock = now
            self._fps_count = 0

        elapsed = time.perf_counter() - now
        delay = max(1, int((TARGET_FRAME - elapsed) * 1000.0))
        self.root.after(delay, self._tick)

    def _advance(self, dt: float) -> None:
        model = self.model
        if model is None:
            return

        cranking = self.cranking or self._auto_crank > 0.0
        if self._auto_crank > 0.0:
            self._auto_crank -= dt
            if model.running:
                self._auto_crank = 0.0
                cranking = False

        model.cranking = cranking and not model.running
        model.limiter_enabled = self.limiter_on
        model.ignition_on = self.ignition_on
        if self._dyno_state:
            # a pull owns the throttle: it is flat out by definition
            self.throttle_cmd = 1.0
            model.set_pedal(1.0)
            model.load = 0.0
        else:
            model.set_pedal(self.throttle_cmd)
            model.load = self.brake_cmd

        was_running = self._was_running
        was_cranking = self._was_cranking
        self._was_cranking = model.cranking
        model.step(dt)
        running = model.running
        self._was_running = running

        # audio follows the model through exactly two channels: the operating
        # point, and the events the physics decided on
        if self.audio is not None:
            for event in model.drain_events():
                self.audio.trigger(event)
                if event == "backfire":
                    self._backfire_flash = 0.35
            if model.cranking and not was_cranking:
                self.audio.trigger("start_crank")
            if running and not was_running:
                self.audio.trigger("start_catch")
            elif was_running and not running and not model.cranking:
                self.audio.trigger("shutdown")
            self.audio.update(
                model.rpm,
                model.sound_load,
                model.throttle,
                model.boost,
                model.knock_level,
                model.firing_cycles,
                model.torque_pulse,
                model.pipe_temp,
            )
            # asset mode: the limiter layer rides on the cut state
            if hasattr(self.audio, "set_limiter"):
                self.audio.set_limiter(1.0 if model.limiter_cut else 0.0)

        self._backfire_flash = max(0.0, self._backfire_flash - dt)
        if model.knock_level > 0.04:
            self._knock_flash = 0.20
        else:
            self._knock_flash = max(0.0, self._knock_flash - dt)
        self._advance_dyno(dt)

    def _advance_dyno(self, dt: float) -> None:
        state = self._dyno_state
        if state is None:
            return
        model = self.model
        dyno = model.dyno

        if state == "settle":
            # hold the starting speed until the brake has actually caught the
            # engine. Recording before then plots the transient, not the engine.
            self._dyno_settle -= dt
            locked = abs(model.rpm - dyno.target_rpm) < 40.0 and model.rpm > dyno.target_rpm * 0.6
            if not model.running and self._dyno_settle < 2.0:
                self._end_dyno("dyno: engine is not running")
                return
            if locked or self._dyno_settle <= 0.0:
                dyno.ramp_rpm_s = 550.0
                self._dyno_state = "ramp"
                self._set_status("dyno: ramping to the redline...")
            return

        # while the brake is still catching up it commands zero, and a zero is
        # not a torque the engine made -- it is the absence of a measurement
        brake = model.brake_torque
        if brake > 0.0:
            self._dyno_points.append((model.rpm, brake))
            start = self.spec.idle_rpm + 400.0
            self._dyno_progress = min(1.0, (model.rpm - start) / max(dyno.end_rpm - start, 1.0))
        if dyno.target_rpm >= dyno.end_rpm * 0.995:
            self._dyno_progress = 1.0
            summary = dyno_mod.summarise(self._dyno_points)
            if summary:
                self._end_dyno(
                    f"dyno done: {summary['peak_torque']:.0f} Nm @ {summary['peak_torque_rpm']:.0f} rpm, "
                    f"{summary['peak_power_kw']:.1f} kW @ {summary['peak_power_rpm']:.0f} rpm"
                )
            else:
                self._end_dyno("dyno done")

    # ------------------------------------------------------------------ #
    # drawing
    # ------------------------------------------------------------------ #
    def _draw_header(self) -> None:
        img = self._header_bg.copy()
        draw = ImageDraw.Draw(img)

        draw.rounded_rectangle([22, 13, 40, 31], radius=7, fill=theme.ACCENT)
        draw.ellipse([27, 18, 35, 26], fill=(255, 255, 255))
        draw.text((52, 22), "ENGINE LAB", font=theme.font(15, bold=True), fill=theme.INK, anchor="lm")
        draw.text(
            (162, 23),
            "engine simulator  ·  3D cutaway  ·  physics  ·  live sound",
            font=theme.font(11),
            fill=theme.INK_FAINT,
            anchor="lm",
        )

        status = self.status_text
        if len(status) > 74:
            status = status[:71] + "..."
        draw.text((WIN_W - 24, 13), status, font=theme.font(11, bold=True), fill=theme.INK_DIM, anchor="ra")
        draw.text(
            (WIN_W - 24, 29),
            f"{self._fps:.0f} fps",
            font=theme.font(10),
            fill=theme.INK_FAINT,
            anchor="ra",
        )

        self.header_image.paste(img)
        self.header_photo.paste(self.header_image)

    def _draw_view(self) -> None:
        if self.scene is None or self.renderer is None:
            return
        self.scene.update(self.model)
        mesh = self.scene.build()
        self.renderer.render(mesh, self.camera)

        draw = self.renderer.draw
        draw.text(
            (18, 14), self.preset.name, font=theme.font(18, bold=True), fill=theme.INK, anchor="la"
        )
        if self.spec.kind == "turbine":
            sub = "continuous combustion · Brayton cycle"
        else:
            sub = f"{self.spec.cylinders} cylinders"
        draw.text(
            (18, 38),
            f"{sub}   {self.spec.displacement * 1000:.1f} L   redline {self.spec.redline_rpm:.0f} rpm",
            font=theme.font(12),
            fill=theme.INK_DIM,
            anchor="la",
        )
        draw.text(
            (INNER_W - 18, INNER_H - 18),
            f"{self._fps:.0f} fps  ·  drag to orbit, wheel to zoom",
            font=theme.font(11),
            fill=theme.INK_FAINT,
            anchor="ra",
        )

        x = 18
        y = INNER_H - 38
        model = self.model
        chip = theme.ACCENT_2 if model.running else theme.INK_FAINT
        if not self.ignition_on:
            state_label = "IGNITION OFF"
        else:
            state_label = "RUNNING" if model.running else "STOPPED"
        x = widgets.draw_chip(draw, x, y, state_label, color=chip)
        if model.limiter_cut:
            x = widgets.draw_chip(draw, x, y, "LIMITER", color=theme.DANGER)
        if model.overrun_cut:
            x = widgets.draw_chip(draw, x, y, "OVERRUN CUT", color=theme.WARM)
        if self._backfire_flash > 0.0:
            x = widgets.draw_chip(draw, x, y, "BACKFIRE", color=theme.WARM)
        if self._knock_flash > 0.0:
            x = widgets.draw_chip(draw, x, y, "KNOCK", color=theme.DANGER)
        if self._dyno_state:
            x = widgets.draw_chip(draw, x, y, "DYNO PULL", color=theme.ACCENT)
        if self.audio is None:
            widgets.draw_chip(draw, x, y, "NO AUDIO", color=theme.DANGER)

        img = self._view_bg.copy()
        img.paste(self.renderer.image, VIEW_INSET, theme.round_mask((INNER_W, INNER_H), 16))
        self.view_image.paste(img)
        self.view_photo.paste(self.view_image)

    def _draw_gauges(self) -> None:
        if self.model is None:
            return
        model = self.model
        spec = self.spec
        dyno_active = model.dyno.active
        shown_torque = model.brake_torque if dyno_active else model.net_torque
        shown_power_kw = max(0.0, shown_torque * model.rpm * TWO_PI / 60.0 / 1000.0)
        img = self._gauge_bg.copy()
        draw = ImageDraw.Draw(img)

        turbine = spec.kind == "turbine"
        if turbine:
            t_max, p_max = 2400.0, 6400.0
        else:
            t_max = max(model.full_load_torque * 1.18, 40.0)
            p_max = max(t_max * spec.redline_rpm * 0.62 * TWO_PI / 60.0 / 1000.0, 20.0)

        widgets.draw_dial(
            draw,
            (22, 16, 226, 200),
            model.rpm,
            0.0,
            spec.redline_rpm * 1.05,
            "rpm",
            "REV",
            redline=spec.redline_rpm,
            accent=theme.ACCENT,
        )
        widgets.draw_dial(
            draw,
            (250, 16, 454, 200),
            shown_power_kw,
            0.0,
            p_max,
            "kW",
            "POWER",
            accent=theme.WARM,
        )

        bars = [
            ("THROTTLE", model.throttle, f"{model.throttle * 100:.0f} %", theme.ACCENT_2, None),
            (
                "MANIFOLD",
                (model.map_bar - 0.10) / max(1.05 + model.spec.boost_bar - 0.10, 0.2),
                f"{model.map_bar:.2f} bar",
                theme.COOL,
                None,
            ),
            ("LOAD", model.sound_load, f"{model.sound_load * 100:.0f} %", theme.ACCENT, None),
            ("TORQUE", shown_torque / t_max, f"{shown_torque:.0f} Nm", theme.WARM, 0.9),
        ]
        y = 210
        for title, fraction, value, color, warn in bars:
            widgets.draw_bar(img, draw, (22, y, 454, y + 30), fraction, title, value, color=color, warn_from=warn)
            y += 36

        stats = [
            (
                "BOOST",
                f"{model.boost_bar:.2f} bar" if model.spec.boost_bar > 0.0 else "n/a (NA)",
                theme.COOL if model.spec.boost_bar > 0.0 else theme.INK_FAINT,
            ),
            ("POWER", f"{shown_power_kw:.1f} kW  ·  {shown_power_kw * 1000 / 745.7:.0f} hp", theme.WARM),
            ("PEAK TORQUE", f"{model.full_load_torque:.0f} Nm", theme.ACCENT),
        ]
        if spec.kind == "petrol":
            stats.append(
                (
                    "KNOCK",
                    f"{model.knock_level * 100:.0f} %  ·  retard {model.knock_retard * 100:.0f} %",
                    theme.DANGER if model.knock_level > 0.04 else theme.INK_FAINT,
                )
            )
        y = 344 if len(stats) > 3 else 356
        step = 22 if len(stats) > 3 else 24
        for label, value, color in stats:
            widgets.draw_stat(draw, (22, y, 454, y + 22), label, value, color)
            y += step

        self.gauge_image.paste(img)
        self.gauge_photo.paste(self.gauge_image)

    def _draw_dyno(self) -> None:
        if self.spec is None:
            return
        # the chart is the most expensive panel to draw and it only changes while
        # a pull is running, so it is skipped when nothing it shows has moved
        signature = (
            len(self._dyno_points),
            self._dyno_state,
            round(self._dyno_progress, 4),
            self._dyno_message,
        )
        if signature == self._dyno_signature:
            return
        self._dyno_signature = signature

        img = self._dyno_bg.copy()
        draw = ImageDraw.Draw(img)
        dyno_mod.draw_chart(
            img,
            draw,
            (20, 12, WIN_W - 20, DYNO_H - 12),
            self._dyno_points,
            redline_rpm=self.spec.redline_rpm,
            running=self._dyno_state is not None,
            progress=self._dyno_progress,
            message=self._dyno_message,
        )
        self.dyno_image.paste(img)
        self.dyno_photo.paste(self.dyno_image)

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self.audio is not None:
            self.audio.stop()
            self.audio = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def window_image(app) -> Image.Image:
    """Stitch the app's panels into one image, exactly as the window lays them out.

    Used for documentation stills and screenshots of a headless run: the panels
    are already complete Pillow images, so no screen capture is involved.
    """
    width = WIN_W
    height = PAD + HEADER_H + PAD + VIEW_H + PAD + DYNO_H + PAD + CTRL_H + PAD
    sheet = Image.new("RGB", (width, height), theme.BG)
    y = PAD
    sheet.paste(app.header_image, (PAD, y))
    y += HEADER_H + PAD
    sheet.paste(app.view_image, (PAD, y))
    sheet.paste(app.gauge_image, (PAD + VIEW_W + PAD, y))
    y += VIEW_H + PAD
    sheet.paste(app.dyno_image, (PAD, y))
    y += DYNO_H + PAD
    sheet.paste(app.ctrl_image, (PAD, y))
    return sheet


def run(
    preset_path: str | None = None,
    audio: bool = True,
    audio_backend: str = "auto",
    audio_device: int | str | None = None,
) -> int:
    # Real-time hygiene for the audio path (dial-up fix): a gen-2 GC pause
    # inside the 2.67 ms audio callback is a multi-millisecond gap, and it
    # repeats with the allocation rate of the render -- the modem texture.
    # Freeze the import graph once, then keep collections to gen-0.
    import gc

    gc.collect()
    gc.freeze()
    gc.set_threshold(50000, 50, 50)

    root = tk.Tk()
    SimulatorApp(
        root,
        preset_path=preset_path,
        audio=audio,
        audio_backend=audio_backend,
        audio_device=audio_device,
    )
    root.mainloop()
    return 0


def capture(preset_path: str | None, out_path: str, seconds: float = 1.5) -> int:
    """Render the window off-screen and save it, for docs and smoke tests."""
    root = tk.Tk()
    app = SimulatorApp(root, preset_path=preset_path, audio=False)
    steps = max(1, int(seconds * 60.0))
    for _ in range(steps):
        app._advance(1.0 / 60.0)
    app._draw_view()
    app._draw_gauges()
    app._draw_dyno()
    app._draw_header()
    app._draw_controls()
    root.update_idletasks()
    image = window_image(app)
    app.close()
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    print(f"wrote {target} ({image.width}x{image.height})")
    return 0

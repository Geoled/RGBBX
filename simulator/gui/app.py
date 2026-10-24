"""The simulator application: state machine, frame loop, panels and shortcuts.

Everything the user can see and touch is here, assembled from the panels in
this package. The app object owns the *state* (engine model, audio, dyno pull,
camera) and the panels own the drawing, so either can change without the other.

Layout (the default, dockable and rearrangeable):

    +-----------------------------------------------------------+
    |  engine bar: preset picker | big readouts | chips | fps   |
    +----------------------------+------------------------------+
    |                            |                              |
    |      3D assembly view      |      cluster + telemetry    |
    |        (orbit / zoom)      |      (dials, bars, stats)    |
    |                            |                              |
    +----------------------------+------------------------------+
    |              dyno curve (torque & power)                  |
    +-----------------------------------------------------------+
    | throttle | starter | ignition | limiter | brake | volume  |
    +-----------------------------------------------------------+

Shortcuts: hold **S** to crank, **Space** to blip, **D** dyno pull, **L**
limiter, **I** ignition, **R** re-frame camera, **G** ghost shell.
"""

from __future__ import annotations

import threading
import time

from .. import bootstrap  # noqa: F401  (sys.path for soundgen)
from ..audio import EngineAudio
from ..physics import EngineModel, EngineSpec
from . import theme
from . import widgets
from .panel_3d import Viewport3D
from .panel_controls import draw_controls
from .panel_dyno import draw_dyno
from .panel_telemetry import draw_telemetry
from .state import summarise

import numpy as np


class AppState:
    """Mutable state shared by every panel, one row per concept."""

    def __init__(self, presets, audio_wanted: bool, audio_backend: str, audio_device) -> None:
        self.presets = presets
        self.preset_index = 0
        self.preset_path = None
        self.model: EngineModel | None = None
        self.spec: EngineSpec | None = None
        self.preset = None

        self.audio_wanted = audio_wanted
        self.audio_backend = audio_backend
        self.audio_device = audio_device
        self.audio: EngineAudio | None = None

        self.throttle = 0.0
        self.brake = 0.0
        self.volume = 0.85
        self.cranking = False
        self.ignition = True
        self.limiter = True

        self.auto_crank = 1.3
        self.was_running = False
        self.was_cranking = False

        self.dyno_state: str | None = None
        self.dyno_settle = 0.0
        self.dyno_points: list[tuple[float, float]] = []
        self.dyno_progress = 0.0
        self.dyno_message = "press RUN PULL to measure this engine"
        self.dyno_summary = None

        self.status = "starting up"
        self.backfire_flash = 0.0
        self.knock_flash = 0.0
        self.rpm_peak = 0.0
        self.kw_peak = 0.0
        self.fps = 0.0
        self.faces = 0

        # the audio thread renders the shots in the background; keep the queue
        # in the state so the panel can show a spinner without touching audio
        self.loading_audio = False
        # one traceback per panel: imgui python bindings can swallow callback
        # exceptions, so a failing panel prints once instead of 60x a second
        self._panel_errors: dict[str, bool] = {}
        # preset display names, loaded once (they never change at runtime)
        self._preset_names: list[str] | None = None


class SimulatorApp:
    """Owns the state, the loop and the panels."""

    def __init__(self, state: AppState, viewport: Viewport3D) -> None:
        from .catalogue import EngineCatalogue

        self.state = state
        self.viewport = viewport
        self.catalogue = EngineCatalogue()
        self._last = time.perf_counter()
        self._fps_clock = time.perf_counter()
        self._fps_count = 0

    # ------------------------------------------------------------------ #
    # engine switching
    # ------------------------------------------------------------------ #
    def select_preset(self, index: int) -> None:
        import threading

        from soundgen.config import load_preset

        state = self.state
        state.preset_index = max(0, min(index, len(state.presets) - 1))
        path = state.presets[state.preset_index]
        state.preset_path = path
        state.status = f"loading {path.stem}..."

        # swap the model under the sim thread's lock: it steps the old model
        # right up to the swap and the new one right after, with no window
        # where it touches a half-initialised object
        with getattr(self, "_sim_lock", threading.Lock()):
            preset = load_preset(str(path))
            spec = EngineSpec.from_preset(preset)
            state.preset = preset
            state.spec = spec
            state.model = EngineModel(spec)
            state.dyno_points = []
            state.dyno_state = None
            state.dyno_summary = None
            state.dyno_progress = 0.0
            state.dyno_message = "press RUN PULL to measure this engine"
            state.auto_crank = 1.3
            state.was_running = False
            state.was_cranking = False
            state.cranking = False
            state.ignition = True
            state.rpm_peak = 0.0
            state.kw_peak = 0.0
            state.brake = 0.0
            state.throttle = 0.0

        # the GL scene swap happens on the GUI thread (GL objects must be
        # created on the context's thread); the sim thread never touches it
        self.viewport.switch_scene(spec)

        if state.audio is not None:
            state.audio.stop()
            state.audio = None
        if state.audio_wanted:
            state.loading_audio = True

            def open_audio():
                try:
                    # Asset mode: the callback only plays pre-rendered loops
                    # (~40 us per block, margin ~60x), so a uniform small
                    # block serves every engine including the 16-cylinder W16;
                    # no per-engine size switch is needed any more.
                    audio = EngineAudio(
                        preset,
                        sr=48000,
                        block_frames=128,
                        buffers=3,
                        master=state.volume,
                        audio_backend=state.audio_backend,
                        audio_device=state.audio_device,
                        mode="assets",
                    )
                    state.audio = audio
                    if not audio.enabled:
                        state.status = f"audio unavailable: {audio.error}"
                    else:
                        state.status = f"loaded {preset.name}"
                except Exception as exc:  # noqa: BLE001
                    state.status = f"audio error: {exc}"
                finally:
                    state.loading_audio = False

            threading.Thread(target=open_audio, daemon=True).start()
        else:
            state.status = "audio disabled (--no-audio)"

    # ------------------------------------------------------------------ #
    # actions
    # ------------------------------------------------------------------ #
    def blip(self) -> None:
        state = self.state
        if state.dyno_state is None:
            state.throttle = 0.65
            self._blip_until = time.perf_counter() + 0.26

    def start_dyno(self) -> None:
        state = self.state
        if state.dyno_state is not None or state.model is None:
            return
        model = state.model
        model.start_dyno_pull()
        model.dyno.ramp_rpm_s = 0.0
        state.dyno_state = "settle"
        state.dyno_settle = 3.0
        state.dyno_points = []
        state.dyno_progress = 0.0
        state.dyno_summary = None
        state.dyno_message = ""
        state.throttle = 1.0
        state.status = "dyno: settling the brake..."

    def end_dyno(self, message: str) -> None:
        state = self.state
        if state.model is not None and state.model.dyno.active:
            state.model.stop_dyno()
        state.dyno_state = None
        state.dyno_message = message
        state.throttle = 0.0
        state.status = message

    # ------------------------------------------------------------------ #
    # the loop
    # ------------------------------------------------------------------ #
    _blip_until = 0.0
    # The simulation runs on its own wall-clock thread at a fixed 120 Hz, so
    # the physics (and the audio parameter stream feeding on it) keep flowing
    # when the GUI thread stalls -- which is exactly what happens on Windows
    # while the window is being dragged or a modal opens: the OS runs the
    # move-loop modally and the renderer simply stops receiving frames. The
    # GUI thread keeps only the drawing-side state (peaks, flashes) in sync.
    _SIM_HZ = 120.0

    def __init__(self, state: AppState, viewport: Viewport3D) -> None:
        from .catalogue import EngineCatalogue

        self.state = state
        self.viewport = viewport
        self.catalogue = EngineCatalogue()
        self._last = time.perf_counter()
        self._fps_clock = time.perf_counter()
        self._fps_count = 0
        self._sim_lock = threading.Lock()
        self._sim_stop = threading.Event()
        self._sim_thread: threading.Thread | None = None

    def start_sim_thread(self) -> None:
        if self._sim_thread is not None:
            return

        def run_sim():
            period = 1.0 / self._SIM_HZ
            # Real-time hygiene for the audio path (dial-up fix): CPython's
            # cyclic GC fires a full collection every few thousand
            # allocations, and the render allocates numpy temporaries every
            # block. A gen-2 pause inside the 2.67 ms audio callback is a
            # multi-millisecond silence -- periodic enough to read as a
            # modem. Freezing the already-built import graph and widening
            # the thresholds pushes collections out of the callback's way;
            # the per-block garbage stays gen-0, which is a few hundred
            # microseconds at worst.
            import gc

            gc.collect()
            gc.freeze()
            gc.set_threshold(50000, 50, 50)
            next_t = time.perf_counter()
            while not self._sim_stop.is_set():
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(min(next_t - now, period))
                    continue
                dt = min(next_t + period - next_t, 0.10) if now - next_t > period else period
                # if we slept far too long (breakpoint, suspend), resync
                if now - next_t > 0.25:
                    next_t = now
                    continue
                with self._sim_lock:
                    self._sim_step(period)
                next_t += period

        self._sim_thread = threading.Thread(target=run_sim, daemon=True, name="engine-sim")
        self._sim_thread.start()

    def stop_sim_thread(self) -> None:
        self._sim_stop.set()
        thread = self._sim_thread
        if thread is not None:
            thread.join(timeout=0.5)
        self._sim_thread = None

    def _sim_step(self, dt: float) -> None:
        """One fixed physics step, on the simulation thread."""
        state = self.state
        model = state.model
        if model is None:
            return

        if self._blip_until and time.perf_counter() > self._blip_until:
            if state.dyno_state is None and state.throttle >= 0.64:
                state.throttle = 0.0
            self._blip_until = 0.0

        cranking = state.cranking or state.auto_crank > 0.0
        if state.auto_crank > 0.0:
            state.auto_crank -= dt
            if model.running:
                state.auto_crank = 0.0
                cranking = False

        model.cranking = cranking and not model.running
        model.limiter_enabled = state.limiter
        model.ignition_on = state.ignition
        if state.dyno_state:
            state.throttle = 1.0
            model.set_pedal(1.0)
            model.load = 0.0
        else:
            model.set_pedal(state.throttle)
            model.load = state.brake

        was_running = state.was_running
        was_cranking = state.was_cranking
        state.was_cranking = model.cranking
        model.step(dt)
        running = model.running
        state.was_running = running

        if state.audio is not None:
            for event in model.drain_events():
                state.audio.trigger(event)
                if event == "backfire":
                    state.backfire_flash = 0.35
            if model.cranking and not was_cranking:
                state.audio.trigger("start_crank")
            if running and not was_running:
                state.audio.trigger("start_catch")
            elif was_running and not running and not model.cranking:
                state.audio.trigger("shutdown")
            state.audio.update(
                model.rpm,
                model.sound_load,
                model.throttle,
                model.boost,
                model.knock_level,
                # the model's own firing-cycle count: the audio grid locks to it,
                # so a backfire or limiter cut lands on the cylinder that caused it
                model.firing_cycles,
                # the Wiebe pulse train: every firing's gain tracks the physics'
                # instantaneous torque, so the audio lugs with the crank
                model.torque_pulse,
                # heat-soaked exhaust gas temperature: the pipe's quarter-wave
                # resonance climbs with the collector temperature
                model.pipe_temp,
            )
            # asset mode: the limiter layer rides on the cut state
            if hasattr(state.audio, "set_limiter"):
                state.audio.set_limiter(1.0 if model.limiter_cut else 0.0)

        state.backfire_flash = max(0.0, state.backfire_flash - dt)
        if model.knock_level > 0.04:
            state.knock_flash = 0.20
        else:
            state.knock_flash = max(0.0, state.knock_flash - dt)

        state.rpm_peak = max(state.rpm_peak, model.rpm)
        state.kw_peak = max(state.kw_peak, model.power_kw)
        self._advance_dyno(dt)

    def advance(self, dt: float) -> None:
        """GUI-thread side: keep drawing state coherent between sim steps.

        The physics itself runs on the sim thread (see start_sim_thread);
        the drawing code samples model fields without a lock because the
        model only ever moves forward in fixed steps -- reading a field
        mid-step gives at worst a one-step-stale gauge, never a torn one.
        """
        state = self.state
        model = state.model
        if model is None:
            return
        _ = dt

    def _advance_dyno(self, dt: float) -> None:
        state = self.state
        stage = state.dyno_state
        if stage is None or state.model is None:
            return
        model = state.model
        dyno = model.dyno

        if stage == "settle":
            state.dyno_settle -= dt
            locked = abs(model.rpm - dyno.target_rpm) < 40.0 and model.rpm > dyno.target_rpm * 0.6
            if not model.running and state.dyno_settle < 2.0:
                self.end_dyno("dyno: engine is not running")
                return
            if locked or state.dyno_settle <= 0.0:
                dyno.ramp_rpm_s = 550.0
                state.dyno_state = "ramp"
                state.status = "dyno: ramping to the redline..."
            return

        brake = model.brake_torque
        if brake > 0.0:
            state.dyno_points.append((model.rpm, brake))
            start = state.spec.idle_rpm + 400.0
            state.dyno_progress = min(1.0, (model.rpm - start) / max(dyno.end_rpm - start, 1.0))
        if dyno.target_rpm >= dyno.end_rpm * 0.995:
            state.dyno_progress = 1.0
            summary = summarise(state.dyno_points)
            if summary:
                state.dyno_summary = summary
                self.end_dyno(
                    f"dyno done: {summary['peak_torque']:.0f} Nm @ {summary['peak_torque_rpm']:.0f} rpm, "
                    f"{summary['peak_power_kw']:.1f} kW @ {summary['peak_power_rpm']:.0f} rpm"
                )
            else:
                self.end_dyno("dyno done")

    # ------------------------------------------------------------------ #
    # per-frame UI
    # ------------------------------------------------------------------ #
    def tick_fps(self) -> None:
        now = time.perf_counter()
        self._fps_count += 1
        if now - self._fps_clock >= 0.5:
            state = self.state
            state.fps = self._fps_count / (now - self._fps_clock)
            self._fps_clock = now
            self._fps_count = 0


def _load_presets():
    from soundgen.config import discover_presets

    return discover_presets(bootstrap.preset_dir())


def _default_preset_index(presets, wanted: str | None) -> int:
    from pathlib import Path

    if not wanted:
        for i, path in enumerate(presets):
            if path.stem == "i4_gasoline":
                return i
        return 0
    name = Path(wanted).name
    stem = Path(wanted).stem
    for i, path in enumerate(presets):
        if path.name == name or path.stem == stem:
            return i
    return 0


# --------------------------------------------------------------------------- #
# keyboard
# --------------------------------------------------------------------------- #
class Shortcuts:
    """Global key bindings, polled each frame (no OS hook needed)."""

    HOLD_KEYS = {"s": "crank"}
    TAP_KEYS = {
        " ": "blip",
        "d": "dyno",
        "e": "catalogue",
        "l": "limiter",
        "i": "ignition",
        "r": "reframe",
        "g": "shell",
    }

    def __init__(self, app: "SimulatorApp") -> None:
        self.app = app
        self._held: set[str] = set()

    def poll(self) -> None:
        from imgui_bundle import imgui

        state = self.app.state

        for key, action in self.HOLD_KEYS.items():
            imgui_key = self._imgui_key(key)
            down = imgui_key is not None and imgui.is_key_down(imgui_key)
            if action == "crank":
                if down and not state.cranking:
                    state.cranking = True
                    state.throttle = 0.0
                elif not down and state.cranking and key in self._held:
                    state.cranking = False
                if down:
                    self._held.add(key)
                else:
                    self._held.discard(key)

        for key, action in self.TAP_KEYS.items():
            imgui_key = self._imgui_key(key)
            if imgui_key is None:
                continue
            if not imgui.is_key_down(imgui_key):
                self._held.discard(f"tap:{key}")
                continue
            if f"tap:{key}" in self._held:
                continue
            self._held.add(f"tap:{key}")
            if action == "blip":
                self.app.blip()
            elif action == "dyno":
                self.app.start_dyno()
            elif action == "catalogue":
                self.app.catalogue.toggle(state)
            elif action == "limiter":
                state.limiter = not state.limiter
                state.status = "limiter " + ("on" if state.limiter else "off")
            elif action == "ignition":
                state.ignition = not state.ignition
                state.status = "ignition " + ("on" if state.ignition else "off")
            elif action == "reframe":
                self.app.viewport.frame_camera()
            elif action == "shell":
                self.app.viewport.show_shell = not self.app.viewport.show_shell

    @staticmethod
    def _imgui_key(key: str):
        if len(key) != 1:
            return None
        from imgui_bundle import imgui

        if key.isalpha():
            return getattr(imgui.Key, key.lower(), None)
        if key == " ":
            return imgui.Key.space
        return None


# --------------------------------------------------------------------------- #
# the frame
# --------------------------------------------------------------------------- #
def show_gui(app: SimulatorApp, state: AppState) -> None:
    """One interface frame: physics step, 3D render, panels.

    The panels are plain (non-docking) imgui windows sized by hand, so the
    layout is explicit and the whole frame is one draw pass: with
    ``no_default_window`` there is no host window to dock into, and
    ``SetNextWindowPos``'s ``Always`` flag keeps our grid geometry immune to
    stray mouse drags.
    """
    import time as _time

    from imgui_bundle import imgui

    now = _time.perf_counter()
    dt = min(max(now - app._last, 0.0), 0.10)
    app._last = now
    app.advance(dt)
    app.tick_fps()
    Shortcuts(app).poll()

    io = imgui.get_io()
    display_w = io.display_size.x
    display_h = io.display_size.y
    if display_w < 64 or display_h < 64:
        # minimized / headless: skip drawing instead of dividing by a
        # collapsed viewport and laying the panels over each other
        return
    pad = 10.0
    header_h = 54.0
    controls_h = 212.0
    dyno_h = 200.0
    gap = 8.0

    # --- engine bar (drawn over the top of the window) ------------------- #
    # Foreground draw list: these rects/texts must sit above every panel,
    # including the child windows the panels live in.
    dl = imgui.get_foreground_draw_list()
    bar_a = imgui.ImVec2(pad, pad)
    bar_b = imgui.ImVec2(display_w - pad, pad + header_h)
    dl.add_rect_filled(bar_a, bar_b, widgets._color_u32(theme.PANEL_B), 0.0)
    dl.add_rect(bar_a, bar_b, widgets._color_u32(theme.EDGE), 0.0)
    dl.add_line(
        imgui.ImVec2(bar_a.x + 1, bar_b.y), imgui.ImVec2(bar_b.x - 1, bar_b.y),
        widgets._color_u32(theme.EDGE_HOT),
    )
    # brand mark
    dl.add_rect_filled(
        imgui.ImVec2(bar_a.x + 14, bar_a.y + 16), imgui.ImVec2(bar_a.x + 30, bar_a.y + 38),
        widgets._color_u32(theme.ORANGE), 4.0,
    )
    brand = "ENGINE LAB"
    widgets.label_at(imgui.ImVec2(bar_a.x + 40, bar_a.y + 12), brand, theme.INK)
    sub = f"{state.spec.displacement * 1000:.1f} L · {state.spec.redline_rpm:.0f} rpm redline" if state.spec else ""
    widgets.label_at(imgui.ImVec2(bar_a.x + 40, bar_a.y + 32), sub, theme.INK_FAINT)
    # live readout in the bar: rpm / kW / fps
    model = state.model
    if model is not None:
        rpm_text = f"{model.rpm:,.0f} rpm"
        kw_text = f"{model.power_kw:.1f} kW"
        fps_text = f"{state.fps:.0f} fps"
        x = bar_b.x - 14
        for text, color in (
            (fps_text, theme.INK_FAINT),
            (kw_text, theme.YELLOW),
            (rpm_text, theme.ORANGE),
        ):
            tw = imgui.calc_text_size(text).x
            x -= tw
            dl.add_text(imgui.ImVec2(x, bar_a.y + 20), widgets._color_u32(color), text)
            x -= 26.0
    # status line
    dl.add_text(
        imgui.ImVec2(bar_a.x + 300, bar_a.y + 32),
        widgets._color_u32(theme.INK_DIM),
        state.status,
    )

    # --- panels: the whole grid is positioned explicitly ------------------ #
    body_top = pad + header_h + gap
    body_h = display_h - body_top - dyno_h - controls_h - 2 * gap - pad
    left_w = display_w * 0.56
    right_w = display_w - left_w - pad * 3

    assembly_pos = (pad, body_top)
    telemetry_pos = (pad + left_w + gap, body_top)
    dyno_pos = (pad, body_top + body_h + gap)
    controls_pos = (pad, body_top + body_h + gap + dyno_h + gap)

    # --- 3D view -------------------------------------------------------- #
    widgets.panel("assembly", left_w, body_h, accent=theme.BLUE, pos=assembly_pos)
    try:
        inner = imgui.get_content_region_avail()
        view_w = max(inner.x, 64)
        view_h = max(inner.y - 22.0, 64)
        pos = imgui.get_cursor_screen_pos()
        imgui.dummy(imgui.ImVec2(view_w, view_h))

        # camera input over the image region
        hovered = imgui.is_item_hovered()
        if hovered:
            io_local = imgui.get_io()
            mouse = io_local.mouse_delta
            if io_local.mouse_down[0]:
                app.viewport.orbit(mouse.x, mouse.y)
            wheel = io_local.mouse_wheel
            if wheel:
                # wheel up = zoom in (the universal convention)
                app.viewport.zoom_by(1.1111 if wheel > 0 else 0.90)
            if imgui.is_key_pressed(imgui.Key.f):
                app.viewport.frame_camera()

        state.faces = app.viewport.draw(view_w, view_h, model=state.model)
        # imgui.image() draws at the *current cursor*, which the dummy just
        # pushed below the reserved box -- step back onto the dummy's top-left
        # or the picture lands one viewport-height too low (outside the panel,
        # clipped to a sliver).
        imgui.set_cursor_screen_pos(pos)
        # glReadPixels/the FBO texture are bottom-up while imgui expects a
        # top-down image: flip the V axis so the engine sits head-up.
        uv0 = imgui.ImVec2(0, 1)
        uv1 = imgui.ImVec2(1, 0)
        imgui.image(
            imgui.ImTextureRef(app.viewport.texture_id()),
            imgui.ImVec2(view_w, view_h),
            uv0,
            uv1,
        )

        # overlay: engine name and hint
        widgets.label_at(imgui.ImVec2(pos.x + 10, pos.y + 8), "drag orbits · wheel zooms · F re-frames", theme.INK_FAINT)
    finally:
        widgets.end_panel()

    # Each panel is isolated: an exception inside one panel must not kill the
    # rest of the frame (and the imgui window stack stays balanced because every
    # panel's finally clause calls end_panel).
    import traceback

    for name, draw in (
        ("telemetry", lambda: draw_telemetry(app, state, right_w, body_h, pos=telemetry_pos)),
        ("dyno", lambda: draw_dyno(app, state, display_w - pad * 2, dyno_h, pos=dyno_pos)),
        ("controls", lambda: draw_controls(app, state, display_w - pad * 2, controls_h, pos=controls_pos)),
    ):
        try:
            draw()
        except Exception:
            if not state._panel_errors.get(name):
                state._panel_errors[name] = True
                traceback.print_exc()
                import sys as _sys

                _sys.stderr.flush()

    # the engine catalogue, last so it layers above every panel
    if app.catalogue.open:
        try:
            app.catalogue.draw(app, state)
        except Exception:
            if not state._panel_errors.get("catalogue"):
                state._panel_errors["catalogue"] = True
                traceback.print_exc()

# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def run(
    preset_path: str | None = None,
    audio: bool = True,
    audio_backend: str = "auto",
    audio_device: int | str | None = None,
    screenshot: bool | str | None = None,
) -> int:
    from imgui_bundle import immapp

    # the enum lives on the params object itself in this binding version
    _DefaultWindowType = type(immapp.RunnerParams().imgui_window_params.default_imgui_window_type)

    presets = _load_presets()
    if not presets:
        raise SystemExit("no presets found")

    state = AppState(presets, audio, audio_backend, audio_device)
    viewport_holder: dict[str, Viewport3D] = {}
    app_ref: dict[str, SimulatorApp] = {}

    def post_init():
        from soundgen.config import load_preset

        theme.install_style()
        index = _default_preset_index(presets, preset_path)
        state.preset_index = index
        path = presets[index]
        state.preset_path = path
        preset = load_preset(str(path))
        spec = EngineSpec.from_preset(preset)
        state.preset = preset
        state.spec = spec
        state.model = EngineModel(spec)
        viewport_holder["vp"] = Viewport3D(spec)
        app_ref["app"] = SimulatorApp(state, viewport_holder["vp"])
        app_ref["app"].select_preset(index)
        # physics + audio feed on their own 120 Hz thread: keeps running at
        # full simulation rate even while the GUI thread is stalled (window
        # drag, catalogue opening)
        app_ref["app"].start_sim_thread()

    def gui():
        app = app_ref.get("app")
        if app is None:
            return
        show_gui(app, state)

    def before_exit():
        vp = viewport_holder.get("vp")
        if vp is not None:
            vp.release()
        app_obj = app_ref.get("app")
        if app_obj is not None:
            app_obj.stop_sim_thread()
            if app_obj.catalogue.open:
                app_obj.catalogue.close()
        if state.audio is not None:
            state.audio.stop()

    runner = immapp.RunnerParams()
    runner.app_window_params.window_title = "Engine Lab"
    runner.app_window_params.window_geometry.size = [1520, 980]
    runner.app_window_params.resizable = True
    runner.imgui_window_params.show_menu_bar = False
    runner.imgui_window_params.show_status_bar = False
    runner.imgui_window_params.default_imgui_window_type = (
        _DefaultWindowType.no_default_window
    )
    runner.imgui_window_params.remember_theme = False
    runner.callbacks.post_init = post_init
    runner.callbacks.show_gui = gui
    runner.callbacks.load_additional_fonts = theme.load_fonts
    runner.imgui_window_params.background_color = (0.016, 0.02, 0.027, 1.0)
    runner.ini_disable = True

    if screenshot:
        # take the shot in before_exit, while the GL context is still alive:
        # screenshot first, then the teardown of the GL objects.
        # The window closes itself after a moment -- a still only needs the
        # engine to have settled at idle, and an unattended run must not hang.
        import time as _t

        deadline = _t.perf_counter() + 6.0

        def gui_with_deadline():
            gui()
            import os as _os

            if _os.environ.get("ENGINELAB_CATALOGUE") and app_ref.get("app") is not None:
                # test hook: open the catalogue once so the still shows it
                a = app_ref["app"]
                if not a.catalogue.open and not a.state._panel_errors:
                    a.catalogue.show(a.state)
                del _os.environ["ENGINELAB_CATALOGUE"]
            if _t.perf_counter() > deadline:
                # bool property on the runner, checked every frame by the host
                runner.app_shall_exit = True

        def before_exit_shot():
            # The window may have been born minimized (the OS does that to
            # background-started processes): restore it so the final frame
            # actually contains the interface instead of an empty surface.
            import ctypes

            hwnd = ctypes.windll.user32.FindWindowW(None, "Engine Lab")
            if hwnd:
                user32 = ctypes.windll.user32
                if user32.IsIconic(hwnd):
                    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
                user32.SetForegroundWindow(hwnd)
                # give the host one rendered frame at full size before reading
                import time as _t2

                _t2.sleep(0.30)

            from imgui_bundle import hello_imgui

            image = hello_imgui.final_app_window_screenshot()
            before_exit()
            import numpy as np
            from PIL import Image
            from pathlib import Path as _P

            arr = np.asarray(image)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[..., :3]
            out = Image.fromarray(arr)
            target = _P(str(screenshot))
            target.parent.mkdir(parents=True, exist_ok=True)
            out.save(target)
            print(f"wrote {target}")

        runner.callbacks.show_gui = gui_with_deadline
        runner.callbacks.before_exit = before_exit_shot
    else:
        runner.callbacks.before_exit = before_exit

    immapp.run(runner)
    return 0


def capture(preset_path: str | None, out_path: str, seconds: float = 1.5) -> int:
    """Render one frame of the 3D assembly off-screen and save it.

    Uses the same GPU pipeline the window uses (headless standalone context),
    so documentation stills and smoke tests check the real renderer rather than
    the retired software one.
    """
    from pathlib import Path

    from PIL import Image

    from ..physics import EngineModel
    from ..view.repro import load_spec
    from .panel_3d import Viewport3D

    preset, spec = load_spec(preset_path)
    model = EngineModel(spec)
    model.cranking = False
    steps = max(1, int(seconds * 60.0))
    for _ in range(steps):
        model.step(1.0 / 60.0)

    viewport = Viewport3D(spec, width=1280, height=800, segments=12, standalone=True)
    try:
        viewport.scene.update(model)
        viewport.draw(viewport.width, viewport.height)
        data = viewport._fbo.read(components=3, alignment=1)
        image = Image.frombytes("RGB", (viewport.width, viewport.height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target)
        print(f"wrote {target} ({image.width}x{image.height})")
    finally:
        viewport.release()
    return 0

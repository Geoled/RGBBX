"""Offline still renders of the 3D assembly.

Used by ``python -m simulator --render-png`` and by the tests, so the geometry can
be inspected without opening a window.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..bootstrap import preset_dir
from ..physics import EngineModel, EngineSpec
from .engine3d import EngineScene
from .raster import SoftwareRenderer, frame_camera

# The stills use the same light viewport as the app, so a preset looks the same
# in the montage, in the docs and on screen.
VIEW_TOP = (250, 252, 255)
VIEW_BOTTOM = (226, 232, 241)
SHEET_BG = (228, 233, 241)


def viewport_renderer(width: int, height: int) -> SoftwareRenderer:
    return SoftwareRenderer(
        width,
        height,
        background_top=VIEW_TOP,
        background_bottom=VIEW_BOTTOM,
        light=(0.36, 0.74, 0.57),
        ambient=0.46,
        shininess=30.0,
        specular=0.22,
        shadow_color=(96, 110, 136),
        shadow_strength=0.42,
    )


def load_spec(preset_path: str | Path | None = None):
    from soundgen.config import load_preset

    if preset_path is None:
        preset_path = preset_dir() / "i4_gasoline.toml"
    path = Path(preset_path)
    if not path.is_absolute() and not path.is_file():
        # accept the file name, the bare stem or an absolute path
        candidates = [preset_dir() / path.name]
        if not path.suffix:
            candidates.append(preset_dir() / f"{path.name}.toml")
        for candidate in candidates:
            if candidate.is_file():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"preset not found: {preset_path}")
    preset = load_preset(str(path))
    return preset, EngineSpec.from_preset(preset)


def render_preview(
    preset_path: str | Path | None,
    out_path: str | Path,
    width: int = 900,
    height: int = 620,
    rpm: float = 2400.0,
    load: float = 0.55,
    azimuth: float = 0.62,
    elevation: float = 0.40,
    distance_scale: float = 1.0,
    segments: int = 12,
) -> int:
    """Render one frame of one engine to a PNG. Returns 0 on success."""
    _preset, spec = load_spec(preset_path)
    model = EngineModel(spec)
    model.rpm = float(rpm)
    model.pedal = float(load)
    model.cranking = False
    # Let the crank turn so the moving parts are not all at dead centre. The
    # angle is deterministic for a given preset and rpm (a fixed number of fixed
    # steps), so repeated renders of one preview are identical.
    for _ in range(400):
        model.step(0.0005)

    scene = EngineScene(spec, segments=segments)
    scene.update(model)
    mesh = scene.build()

    lo, hi = scene.bounds()
    camera = frame_camera(
        lo, hi, width, height, azimuth=azimuth, elevation=elevation, fill=0.80 / max(distance_scale, 0.1)
    )
    renderer = viewport_renderer(width, height)
    renderer.render(mesh, camera)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    renderer.image.save(out)
    return 0


def render_montage(out_path: str | Path, width: int = 420, height: int = 300) -> int:
    """One labelled frame per preset, in a grid, for eyeballing the geometry."""
    tiles = []
    from soundgen.config import discover_presets

    # One renderer for the whole sheet: it owns the background gradient and the
    # draw context, and every tile is the same size anyway.
    renderer = viewport_renderer(width, height)
    for path in discover_presets(preset_dir()):
        try:
            preset, spec = load_spec(path)
        except Exception as exc:  # a broken preset should not stop the montage
            print(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        model = EngineModel(spec)
        model.rpm = max(spec.idle_rpm * 2.2, 0.35 * spec.redline_rpm)
        model.pedal = 0.5
        model.cranking = False
        for _ in range(400):
            model.step(0.0005)
        scene = EngineScene(spec, segments=10)
        scene.update(model)
        lo, hi = scene.bounds()
        renderer.render(scene.build(), frame_camera(lo, hi, width, height))
        tiles.append((preset.name, renderer.image.copy()))

    if not tiles:
        return 1
    try:
        label_font = ImageFont.truetype("segoeui.ttf", 16)
    except OSError:
        label_font = ImageFont.load_default()

    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height), SHEET_BG)
    for i, (name, img) in enumerate(tiles):
        x = (i % cols) * width
        y = (i // cols) * height
        sheet.paste(img, (x, y))
        draw = ImageDraw.Draw(sheet)
        # the tiles all carry the same light gradient, so a hairline frame is
        # what actually separates one engine from the next
        draw.rectangle([x, y, x + width - 1, y + height - 1], outline=(203, 212, 226))
        draw.rounded_rectangle([x + 12, y + height - 38, x + 12 + 26, y + height - 12], radius=6, fill=(59, 130, 246))
        draw.text((x + 46, y + height - 25), name, font=label_font, fill=(26, 33, 46), anchor="lm")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return 0

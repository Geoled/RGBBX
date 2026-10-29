"""Export a game-ready sound pack from the desktop app.

The app itself plays the procedural bank; this is the "hand the sounds to a
sound designer" path. Everything here is a thin wrapper around
``soundgen.pack`` so the GUI and the CLI produce byte-identical folders.

Packs go to ``%LOCALAPPDATA%\\EngineLab\\exports`` rather than next to the exe:
the install directory may be read-only (``Program Files``), and an export is
per-user output, like the loop-bank cache.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import bootstrap  # noqa: F401  (sets sys.path up before soundgen import)

from soundgen import pack  # noqa: E402
from soundgen.config import Preset  # noqa: E402


def exports_dir() -> Path:
    """Root folder for exported packs (never inside the repository)."""
    root = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_DATA_HOME")
    base = Path(root) if root else Path.home() / ".local" / "share"
    return base / "EngineLab" / "exports"


def export_preset(
    preset: Preset,
    out_root: str | Path | None = None,
    *,
    archive: bool = True,
    sample_rate: int | None = 48000,
    bit_depth: int = 16,
    channels: int = 2,
    progress=None,
) -> pack.PackResult:
    """Render one engine into ``<out_root>/<slug>`` and return what was written.

    Defaults are aimed at a game engine rather than at a mixing desk: 48 kHz,
    16-bit, stereo (the exhaust banks are panned and the loop decorrelation
    lives in the second channel). The zip is on because the next step is
    usually "send this to someone".
    """
    root = Path(out_root) if out_root is not None else exports_dir()
    opts = pack.PackOptions(
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        channels=channels,
        include_events=True,
        archive=archive,
        dither=True,
    )
    return pack.export_single(preset, root, opts, quiet=True)


def reveal(path: str | Path) -> None:
    """Open a folder in the desktop file manager, best effort.

    Never raises: a headless or locked-down machine should still get a status
    line with the path instead of a dialog in the middle of a render.
    """
    target = Path(path)
    if not target.exists():
        return
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
    except Exception:  # noqa: BLE001 - opening Explorer is cosmetic
        pass

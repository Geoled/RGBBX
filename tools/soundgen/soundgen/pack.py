"""Game-ready asset packs.

``render`` writes a flat pile of WAVs and two manifests. That is the right
thing for the offline tool, but a sound designer integrating an engine into a
game wants a folder they can drag into a project: one directory per engine,
loops and one-shots separated, and a README that says which loop plays at
which rpm and load.

This module is the packaging layer on top of :mod:`soundgen.render`:

    pack/
        README.md                 how to wire it up
        manifest.json             machine-readable, one entry per preset
        EngineSoundManifest.luau  ready-to-require Roblox module
        <slug>/
            layers/<slug>__idle.wav ...     seamless, exactly periodic
            events/<slug>__backfire.wav ...  one-shots (turbo_spool/knock loop)

Nothing here re-implements synthesis: it calls ``render_many`` with
``subdirs=True`` and adds the README and the archive.
"""

from __future__ import annotations

import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import events as events_mod, layers as layers_mod, render
from .config import Preset


@dataclass
class PackOptions:
    sample_rate: int | None = None
    bit_depth: int | None = None
    channels: int | None = None
    include_events: bool = True
    archive: bool = False
    dither: bool = False


@dataclass
class PackResult:
    root: Path
    slugs: list[str]
    files: int
    bytes: int
    archive: Path | None
    seconds: float

    def summary(self) -> str:
        size = self.bytes / 1e6
        text = f"{self.files} files, {size:.1f} MB, {self.seconds:.1f}s -> {self.root}"
        if self.archive is not None:
            # The archive holds the same bytes again; reporting it inside
            # ``bytes`` would double-count the pack.
            text += f"\narchive: {self.archive} ({self.archive.stat().st_size / 1e6:.1f} MB)"
        return text


def _readme(presets: list[Preset], sample_rate: int, bit_depth: int, channels: int) -> str:
    lines: list[str] = []
    lines.append("# Engine sound pack\n")
    lines.append(
        "Rendered by `tools/soundgen`. Every loop is an exact whole number of "
        "engine cycles, so it repeats without a seam and needs no crossfade "
        "at the loop point."
    )
    lines.append("")
    lines.append(f"* sample rate: {sample_rate} Hz")
    lines.append(f"* bit depth: {bit_depth} bit")
    lines.append(f"* channels: {channels} ({'mono' if channels == 1 else 'stereo'})")
    lines.append("* `manifest.json` / `EngineSoundManifest.luau` list every file with")
    lines.append("  `looping`, `loopStart`, `loopEnd` and `referenceRpm`.")
    lines.append("")
    lines.append("## How to play it\n")
    lines.append(
        "1. Pick the two loops that bracket the current operating point: one by "
        "rpm, one by load (closed throttle / part throttle / wide open)."
    )
    lines.append(
        "2. Blend them with equal-power crossfades. `load` and `throttle` in "
        "the manifest are the operating points the loops were rendered at."
    )
    lines.append(
        "3. Pitch each layer with `playbackRate = rpm / referenceRpm`. A loop "
        "covers roughly +-25 % around its reference before it sounds shifted, "
        "which is why there are five of them."
    )
    lines.append(
        "4. Fire the one-shots from gameplay events: starter, catch, shutdown, "
        "backfire, blow-off, turbo spool, detonation. `turbo_spool` and `knock` "
        "are loops; the rest are one-shots."
    )
    lines.append("")

    for preset in presets:
        lines.append(f"## {preset.name} (`{preset.slug}`)\n")
        lines.append(
            f"{preset.block.cylinders} cylinders, {preset.block.layout}, "
            f"{preset.engine.idle_rpm:.0f}-{preset.engine.redline_rpm:.0f} rpm."
        )
        lines.append("")
        lines.append("| layer | rpm | load | throttle | length | file |")
        lines.append("|---|---|---|---|---|---|")
        for spec, rpm, length in layers_mod.layer_descriptors(preset, sample_rate):
            tag = " (limiter)" if spec.limiter else ""
            lines.append(
                f"| {spec.key}{tag} | {rpm:.0f} | {spec.load:.2f} | {spec.throttle:.2f} "
                f"| {length / sample_rate:.2f} s | `{preset.slug}/layers/{preset.slug}__{spec.key}.wav` |"
            )
        events = events_mod.events_for(preset)
        if events:
            lines.append("")
            lines.append("| event | kind | length | trigger |")
            lines.append("|---|---|---|---|")
            for spec in events:
                kind = "loop" if spec.looping else "one-shot"
                lines.append(
                    f"| {spec.key} | {kind} | {spec.duration_s:.2f} s "
                    f"| {spec.label} |"
                )
        lines.append("")
    return "\n".join(lines)


def export_pack(
    presets: list[Preset],
    out_dir: str | Path,
    opts: PackOptions | None = None,
    quiet: bool = False,
    progress=None,
) -> PackResult:
    """Render one engine (or the whole catalogue) into a game-ready folder.

    ``progress`` is called as ``progress(done, total, label)`` before each
    preset, so a GUI can show something moving without knowing the render
    internals. It is called from the rendering thread, not the UI thread.
    """
    opts = opts or PackOptions()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    if progress is not None:
        progress(0, len(presets), presets[0].name if presets else "")

    render_opts = render.RenderOptions(
        sample_rate=opts.sample_rate,
        bit_depth=opts.bit_depth,
        channels=opts.channels,
        layers=True,
        events=opts.include_events,
        per_file_normalize=False,
        dither=opts.dither,
        subdirs=True,
    )
    files, json_path, luau_path = render.render_many(
        presets, out_dir, render_opts, quiet=quiet
    )

    sample_rate = int(opts.sample_rate or presets[0].render.sample_rate)
    bit_depth = int(opts.bit_depth or presets[0].render.bit_depth)
    channels = int(opts.channels or presets[0].render.channels)
    readme = out_dir / "README.md"
    readme.write_text(_readme(presets, sample_rate, bit_depth, channels), encoding="utf-8")

    total_bytes = sum(f.path.stat().st_size for f in files) + readme.stat().st_size
    total_bytes += json_path.stat().st_size + luau_path.stat().st_size

    archive = None
    if opts.archive:
        archive = out_dir.with_suffix(".zip")
        if archive.exists():
            archive.unlink()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(out_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(out_dir.parent))

    return PackResult(
        root=out_dir,
        slugs=[p.slug for p in presets],
        files=len(files) + 3,
        bytes=total_bytes,
        archive=archive,
        seconds=time.perf_counter() - start,
    )


def export_single(
    preset: Preset,
    out_dir: str | Path,
    opts: PackOptions | None = None,
    quiet: bool = False,
) -> PackResult:
    """Convenience wrapper for the GUI: one engine, its own folder."""
    out_dir = Path(out_dir)
    if out_dir.name != preset.slug:
        out_dir = out_dir / preset.slug
    return export_pack([preset], out_dir, opts, quiet=quiet)


def clean(out_dir: str | Path) -> None:
    """Remove a previously exported pack (the GUI re-exports into the same path)."""
    path = Path(out_dir)
    if path.is_dir():
        shutil.rmtree(path)

"""Offline rendering: presets -> WAV files + manifest."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import dsp, events as events_mod, layers as layers_mod, manifest, voice
from .config import Preset, discover_presets, load_preset


@dataclass
class RenderOptions:
    sample_rate: int | None = None
    bit_depth: int | None = None
    channels: int | None = None
    block_size: int | None = None
    layers: bool = True
    events: bool = True
    per_file_normalize: bool = False
    dither: bool = False
    # Game packs want ``<slug>/layers`` and ``<slug>/events``; the offline tool
    # keeps its flat ``<slug>__<key>.wav`` layout by default so existing
    # manifest paths (and anything reading them) do not change.
    subdirs: bool = False


@dataclass
class RenderedFile:
    preset_slug: str
    kind: str
    key: str
    path: Path
    sample_rate: int
    duration_s: float
    peak_db: float
    rms_db: float
    looping: bool
    reference_rpm: float | None = None
    loop_start: float = 0.0
    loop_end: float = 0.0
    asset_id: int = 0
    lufs: float | None = None
    # Operating point the loop was rendered at. A game mixer needs this to know
    # which layer to favour at a given load, not just at a given rpm.
    load: float | None = None
    throttle: float | None = None

    def to_record(self, out_dir: Path) -> dict:
        return {
            "file": str(self.path.relative_to(out_dir)).replace("\\", "/"),
            "assetId": self.asset_id,
            "looping": self.looping,
            "loopStart": round(self.loop_start, 6),
            "loopEnd": round(self.loop_end, 6),
            "duration": round(self.duration_s, 6),
            "peakDb": round(self.peak_db, 2),
            "rmsDb": round(self.rms_db, 2),
            **({"lufs": round(self.lufs, 2)} if self.lufs is not None else {}),
            **({"referenceRpm": round(self.reference_rpm, 1)} if self.reference_rpm is not None else {}),
            **({"load": round(self.load, 3)} if self.load is not None else {}),
            **({"throttle": round(self.throttle, 3)} if self.throttle is not None else {}),
        }


def _to_stereo(x: np.ndarray, sr: int, looping: bool = False) -> np.ndarray:
    """Cheap decorrelation so the pair is not perfectly identical.

    Loops get a *circular* shift. A zero-padded shift leaves ``d`` silent
    samples at the end of the right channel, which breaks the exact periodicity
    of the file and clicks once per loop; rolling keeps the tail wrapping into
    the head, so the seam survives.
    """
    d = max(int(0.0004 * sr), 1)
    if looping:
        right = np.roll(x, d) * 0.97
    else:
        right = np.zeros_like(x)
        right[d:] = x[:-d]
        right *= 0.97
    return np.stack([x, right], axis=1)


def _measure(x: np.ndarray, sr: int) -> tuple[float, float, float | None]:
    """Return peak dBFS, RMS dBFS and, when available, integrated LUFS."""
    mono = x.mean(axis=1) if x.ndim == 2 else x
    lufs = None
    try:
        import pyloudnorm as pyln  # type: ignore

        try:
            meter = pyln.Meter(int(sr), block_size=0.400)
        except TypeError:  # older pyloudnorm used ``block``
            meter = pyln.Meter(int(sr), block=0.400)
        # pyloudnorm wants at least one loudness block; short one-shots do not
        # have a meaningful integrated loudness, so leave them unlabelled.
        if mono.size >= int(0.400 * int(sr)):
            value = float(meter.integrated_loudness(np.asarray(mono, dtype=np.float64)))
            if math.isfinite(value):
                lufs = value
    except Exception:
        lufs = None
    return dsp.lin_to_db(dsp.peak(mono)), dsp.lin_to_db(dsp.rms(mono)), lufs


def render_preset(
    preset: Preset,
    out_dir: str | Path,
    opts: RenderOptions | None = None,
    quiet: bool = False,
) -> list[RenderedFile]:
    opts = opts or RenderOptions()
    out_dir = Path(out_dir)
    sr = int(opts.sample_rate or preset.render.sample_rate)
    bits = int(opts.bit_depth or preset.render.bit_depth)
    channels = int(opts.channels or preset.render.channels)
    block = int(opts.block_size or preset.render.block_size)

    # Calibrate the shared gain against the exact operating points that actually
    # get exported. A generic sweep would be much more conservative (the peak is
    # not monotonic in rpm) and would throw away several dB of headroom.
    points = [(rpm, spec.load, spec.throttle) for spec, rpm, _ in layers_mod.layer_descriptors(preset, sr)]
    gain = voice.calibrate_gain(
        preset, sr=sr, block=block, operating_points=points, target_peak_db=-2.0
    )
    results: list[RenderedFile] = []

    def emit(
        kind: str,
        key: str,
        x: np.ndarray,
        looping: bool,
        rpm: float | None,
        loop_end: float,
        load: float | None = None,
        throttle: float | None = None,
    ) -> None:
        if opts.per_file_normalize:
            x = dsp.normalize(x, preset.render.peak_db)
        if channels == 2 and x.ndim == 1:
            x = _to_stereo(x, sr, looping=looping)
        base = out_dir
        if opts.subdirs:
            base = out_dir / preset.slug / ("layers" if kind == "layer" else "events")
            base.mkdir(parents=True, exist_ok=True)
        path = base / f"{preset.slug}__{key}.wav"
        peak_db, rms_db, lufs = _measure(x, sr)
        from .wavio import write_wav

        # Dither only one-shots: layer loops must stay exactly periodic.
        write_wav(path, x, sr, bits, dither=opts.dither and not looping)
        results.append(
            RenderedFile(
                preset_slug=preset.slug,
                kind=kind,
                key=key,
                path=path,
                sample_rate=sr,
                duration_s=x.shape[0] / sr,
                peak_db=peak_db,
                rms_db=rms_db,
                looping=looping,
                reference_rpm=rpm,
                loop_start=0.0,
                loop_end=loop_end,
                lufs=lufs,
                load=load,
                throttle=throttle,
            )
        )
        if not quiet:
            loudness = f"  lufs {lufs:6.1f}" if lufs is not None else ""
            print(
                f"  [{kind:5}] {key:12} {x.shape[0] / sr:6.2f}s  "
                f"peak {peak_db:6.2f} dBFS  rms {rms_db:6.2f} dB{loudness}"
            )

    if opts.layers:
        for spec, rpm, length in layers_mod.layer_descriptors(preset, sr):
            x = layers_mod.render_layer(preset, spec, gain, sr=sr, block=block)
            emit(
                "layer",
                spec.key,
                x,
                True,
                rpm,
                x.shape[0] / sr,
                load=spec.load,
                throttle=spec.throttle,
            )

    if opts.events:
        for spec in events_mod.events_for(preset):
            x = events_mod.render_event(preset, spec, gain, sr=sr, block=block)
            emit("event", spec.key, x, spec.looping, None, x.shape[0] / sr if spec.looping else 0.0)

    return results


def render_many(
    presets: list[Preset],
    out_dir: str | Path,
    opts: RenderOptions | None = None,
    quiet: bool = False,
) -> tuple[list[RenderedFile], Path, Path]:
    opts = opts or RenderOptions()
    out_dir = Path(out_dir)
    all_files: list[RenderedFile] = []
    entries = []

    # The file name is derived from the preset name, so two presets with the
    # same name would silently overwrite each other's WAVs and manifest entry.
    seen_slugs: dict[str, str] = {}
    for preset in presets:
        other = seen_slugs.get(preset.slug)
        if other is not None:
            raise ValueError(
                f"presets '{other}' and '{preset.name}' share the output slug "
                f"'{preset.slug}'; give them distinct names"
            )
        seen_slugs[preset.slug] = preset.name

    # The manifest carries one sample rate for the whole set, so it cannot be
    # right if the presets were rendered at different rates: taking the maximum
    # would label 44.1 kHz files as 48 kHz. All bundled presets agree, so this
    # only fires on a genuinely mixed set. Checked before anything is written,
    # so a bad set cannot leave a half-rendered directory behind.
    rates = {int(opts.sample_rate or p.render.sample_rate) for p in presets}
    if len(rates) > 1:
        raise ValueError(
            "presets use different sample rates "
            f"({', '.join(str(r) for r in sorted(rates))}); pass --sr to render them into one set"
        )

    for preset in presets:
        if not quiet:
            print(f"Rendering '{preset.name}' ({manifest.layout.describe(preset.block)})")
        files = render_preset(preset, out_dir, opts, quiet=quiet)
        all_files.extend(files)
        entries.append(
            manifest.preset_entry(
                preset,
                layers=[f.to_record(out_dir) for f in files if f.kind == "layer"],
                events=[f.to_record(out_dir) for f in files if f.kind == "event"],
                sample_rate=int(opts.sample_rate or preset.render.sample_rate),
                filename_prefix=preset.slug,
            )
        )
    # The manifest carries one sample rate for the whole set, so it cannot be
    # right if the presets were rendered at different rates: taking the maximum
    # would label 44.1 kHz files as 48 kHz. All bundled presets agree, so this
    # only fires on a genuinely mixed set.
    sr = rates.pop() if rates else 48000
    bits = int(opts.bit_depth or max(p.render.bit_depth for p in presets))
    json_path, luau_path = manifest.write_manifest(entries, out_dir, sr, bits)

    # Prune stale assets. Changing a preset (for example disabling a blow-off
    # valve) removes a file from the plan, and an orphaned WAV left in the
    # output directory is very easy to upload by accident. Only files that
    # belong to a preset rendered in *this* run are considered.
    #
    # A partial run must not prune: with --layers-only the event WAVs are
    # absent from `all_files` simply because they were not asked for, and the
    # glob below would delete a perfectly good set of one-shots.
    written = {f.path.resolve() for f in all_files}
    removed: list[str] = []
    full_run = opts.layers and opts.events
    if full_run:
        pattern = "**/" if opts.subdirs else ""
        for prefix in sorted({f.preset_slug for f in all_files}):
            for stale in out_dir.glob(f"{pattern}{prefix}__*.wav"):
                if stale.resolve() not in written:
                    stale.unlink()
                    removed.append(stale.name)

    if not quiet:
        print(f"\n{len(all_files)} files -> {out_dir}")
        if removed:
            print(f"pruned {len(removed)} stale file(s): {', '.join(sorted(removed))}")
        elif not full_run:
            print("prune skipped: partial run (--layers-only/--events-only)")
        print(f"manifest: {json_path.name}, {luau_path.name}")
    return all_files, json_path, luau_path


def load_presets(paths: list[str | Path]) -> list[Preset]:
    out = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in discover_presets(p):
                out.append(load_preset(f))
        else:
            out.append(load_preset(p))
    return out

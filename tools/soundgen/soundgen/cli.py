"""Command-line interface for the engine-sound pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import events as events_mod, layers as layers_mod, realtime, render
from .config import load_preset, discover_presets
from . import layout as layout_mod

PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"


def _resolve_presets(args) -> list[Path]:
    paths: list[Path] = []
    if args.preset:
        for p in args.preset:
            paths.append(Path(p))
    else:
        paths.append(PRESETS_DIR)
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(discover_presets(p))
        elif p.suffix == ".toml" and p.exists():
            out.append(p)
        else:
            raise SystemExit(f"preset not found: {p}")
    if not out:
        raise SystemExit("no presets found")
    return out


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_list(args) -> int:
    files = _resolve_presets(argparse.Namespace(preset=None))
    print(f"Presets in {PRESETS_DIR}:\n")
    seen_slugs: dict[str, str] = {}
    for f in files:
        try:
            p = load_preset(f)
        except Exception as exc:  # noqa: BLE001
            print(f"  {f.name:24} !! {exc}")
            continue
        if p.slug in seen_slugs:
            print(f"  !! {f.name} uses the same output slug '{p.slug}' as {seen_slugs[p.slug]}")
        else:
            seen_slugs[p.slug] = f.name
        turbo = f", turbo ({p.turbo.kind})" if p.turbo.enabled else ""
        print(f"  {f.name:24} {layout_mod.describe(p.block):28} "
              f"{p.engine.idle_rpm:.0f}-{p.engine.redline_rpm:.0f} rpm{turbo}")
    return 0


def cmd_new(args) -> int:
    """Scaffold a new preset from the annotated template."""
    import re

    stem = re.sub(r"[^a-z0-9]+", "_", args.name.strip().lower()).strip("_") or "engine"
    out = Path(args.out) if args.out else PRESETS_DIR / f"{stem}.toml"
    if out.exists() and not args.force:
        print(f"{out} already exists (use --force to overwrite)")
        return 1
    template = (PRESETS_DIR / "_template.toml").read_text(encoding="utf-8")
    template = re.sub(
        r'^name = ".*"$', f'name = "{args.name}"', template, count=1, flags=re.MULTILINE
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template, encoding="utf-8")
    print(f"wrote {out}")
    print(f"next: edit the file, then  python -m soundgen info --preset {out}")
    return 0


def cmd_info(args) -> int:
    preset = load_preset(args.preset[0])
    b = preset.block
    print(f"Preset      : {preset.name}")
    if preset.notes:
        print(f"Notes       : {preset.notes}")
    print(f"Engine      : {layout_mod.describe(b)}")
    print(f"Displacement: {layout_mod.displacement_litres(b):.2f} L "
          f"({layout_mod.cylinder_cc(b):.0f} cc/cyl)")
    print(f"Bore x stroke: {b.bore_mm:.0f} x {b.stroke_mm:.0f} mm "
          f"(bore/stroke {layout_mod.bore_stroke_ratio(b):.2f})")
    print(f"Compression : {b.compression:.1f}:1   octane {b.octane:.0f} RON")
    print(f"Cycle revs  : {layout_mod.cycle_revolutions(b):.0f}")
    print(f"Events/cycle: {layout_mod.events_per_cycle(b)}   "
          f"events/rev: {layout_mod.events_per_revolution(b):.2f}")
    for rpm in (preset.engine.idle_rpm, preset.engine.redline_rpm):
        ff = layout_mod.firing_frequency(b, rpm)
        print(f"  @ {rpm:6.0f} rpm -> firing {ff:7.1f} Hz, crank {layout_mod.crankshaft_frequency(rpm):6.1f} Hz")
    banks = layout_mod.bank_assignment(b)
    n_banks = len(set(banks))
    _bank = "single" if n_banks <= 1 else f"{n_banks} banks (burble via pipe asymmetry)"
    print(f"Banks       : {_bank}")
    print(f"Size factor : {layout_mod.size_factor(b):.3f} (resonance scaling)")
    print(f"Turbo       : {'on, ' + preset.turbo.kind if preset.turbo.enabled else 'off'}")
    if b.layout == "turbine":
        gt = preset.turbine
        print(
            f"Turbine     : PR {gt.pressure_ratio:.1f}, "
            f"TIT {gt.idle_turbine_inlet_c:.0f}-{gt.max_turbine_inlet_c:.0f} C, "
            f"{gt.mass_flow_kg_s:.1f} kg/s"
        )
    print("\nLayer plan:")
    sr = preset.render.sample_rate
    for spec, rpm, length in layers_mod.layer_descriptors(preset, sr):
        print(f"  {spec.key:9} {rpm:6.0f} rpm  load {spec.load:4.2f}  thr {spec.throttle:4.2f}  "
              f"{length / sr:5.2f}s{'  [limiter]' if spec.limiter else ''}")
    print("\nEvents:")
    for spec in events_mod.events_for(preset):
        print(f"  {spec.key:12} {spec.duration_s:4.2f}s{'  [loop]' if spec.looping else ''}  {spec.label}")
    return 0


def cmd_render(args) -> int:
    paths = _resolve_presets(args)
    presets = [load_preset(p) for p in paths]
    opts = render.RenderOptions(
        sample_rate=args.sr,
        bit_depth=args.bits,
        channels=args.channels,
        block_size=args.block,
        layers=not args.events_only,
        events=not args.layers_only,
        per_file_normalize=args.normalize,
        dither=args.dither,
    )
    start = time.perf_counter()
    files, json_path, luau_path = render.render_many(presets, args.out, opts, quiet=args.quiet)
    elapsed = time.perf_counter() - start
    print(f"\nDone in {elapsed:.2f}s  |  manifest: {json_path.name}, {luau_path.name}")
    print("Next: upload the WAV files, then set assetId in EngineSoundManifest.luau.")
    return 0


def cmd_bench(args) -> int:
    presets = [load_preset(p) for p in _resolve_presets(args)]
    rates = [args.sr] if args.sr else [48000, 96000]
    for preset in presets:
        print(f"Benchmarking '{preset.name}' ...")
        for sr in rates:
            result = realtime.bench(preset, seconds=args.seconds, sr=sr, block=args.block)
            label = "OK       " if result.realtime_factor >= 1.0 else "TOO SLOW "
            print(f"  {sr:6d} Hz : {label}{result}")
    return 0


def cmd_realtime(args) -> int:
    preset = load_preset(_resolve_presets(args)[0])
    realtime.play(
        preset,
        seconds=args.seconds,
        sr=args.sr,
        block=args.block,
        rpm=args.rpm,
        load=args.load,
        throttle=args.throttle,
        interactive=args.interactive,
    )
    return 0


def cmd_selftest(args) -> int:
    """Quick end-to-end sanity check: synth, loop, one-shots, manifest."""
    import numpy as np

    from . import dsp, wavio

    here = PRESETS_DIR / "i4_gasoline.toml"
    preset = load_preset(here)
    sr, block = 24000, 512
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        mark = "PASS" if cond else "FAIL"
        print(f"  [{mark}] {name}{(' -- ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print("Self test")
    check("preset loads", preset.block.cylinders == 4 and preset.block.layout == "inline", preset.name)
    check("displacement sane", 1.8 < layout_mod.displacement_litres(preset.block) < 2.2,
          f"{layout_mod.displacement_litres(preset.block):.2f} L")
    check("firing frequency sensible",
          abs(layout_mod.firing_frequency(preset.block, 3000) - 100.0) < 1e-6,
          f"{layout_mod.firing_frequency(preset.block, 3000):.1f} Hz @ 3000 rpm")

    gain = 1.0
    layer = layers_mod.render_layer(preset, layers_mod.DEFAULT_LAYERS[1], gain, sr=sr, block=block)
    check("layer renders", layer.size > 0)
    check("layer finite", bool(np.all(np.isfinite(layer))))
    check("layer not silent", dsp.rms(layer) > 1e-6, f"rms {dsp.lin_to_db(dsp.rms(layer)):.1f} dB")

    # True loop quality: the voice output must repeat *exactly* after one loop
    # length, which is what makes the asset seam-free.
    from . import voice as voice_mod

    rpm = layers_mod.quantized_rpm(preset, layers_mod.layer_rpm(preset, layers_mod.DEFAULT_LAYERS[1]), sr)
    total = layers_mod.loop_length_samples(preset, rpm, sr)
    probe = voice_mod.EngineVoice(
        preset,
        rpm=rpm,
        load=0.45,
        throttle=0.42,
        sr=sr,
        block=block,
        output_gain=1.0,
        quantize_period=True,
        loop_samples=total,
    )
    probe.process(layers_mod.warmup_samples(preset, rpm, sr))
    first = probe.process(total)
    following = probe.process(block)
    seam = dsp.rms(first[:block] - following)
    level = max(dsp.rms(first), 1e-12)
    check(
        "loop is exactly periodic",
        seam < 1e-5 * level,
        f"seam {seam:.2e} ({dsp.lin_to_db(seam / level):.0f} dB below signal)",
    )

    ev = events_mod.render_event(preset, events_mod.EVENTS[3], gain, sr=sr, block=block)
    check("backfire renders", ev.size > 0 and bool(np.all(np.isfinite(ev))))

    out = Path(args.out) if args.out else Path("out/_selftest")
    wavio.write_wav(out / "selftest.wav", layer, sr, 16)
    back, sr2 = wavio.read_wav(out / "selftest.wav")
    check("wav roundtrip", sr2 == sr and abs(back.size - layer.size) == 0, f"{back.size} frames @ {sr2} Hz")

    files, _, _ = render.render_many([preset], out, render.RenderOptions(sample_rate=sr, block_size=block), quiet=True)
    check("full render", len(files) >= len(layers_mod.DEFAULT_LAYERS) + 5, f"{len(files)} files")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soundgen",
        description="Offline/real-time engine sound synthesizer for the Roblox engine simulator.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="list available presets")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("info", help="show derived engine timing for a preset")
    p.add_argument("--preset", nargs="+", required=True)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("new", help="scaffold a new engine preset from the template")
    p.add_argument("name", help='display name, for example "My Engine 2.0"')
    p.add_argument("--out", help="output file (default: presets/<slug>.toml)")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("render", help="render WAV layers/events and write a manifest")
    p.add_argument("--preset", nargs="+", help="preset files or a directory (default: bundled presets)")
    p.add_argument("--out", default="out", help="output directory")
    p.add_argument("--sr", type=int, help="override sample rate")
    p.add_argument("--bits", type=int, choices=(16, 24), help="override bit depth")
    p.add_argument("--channels", type=int, choices=(1, 2), help="override channels")
    p.add_argument("--block", type=int, help="override DSP block size")
    p.add_argument("--layers-only", action="store_true")
    p.add_argument("--events-only", action="store_true")
    p.add_argument("--normalize", action="store_true", help="peak-normalize each file (loses relative loudness)")
    p.add_argument("--dither", action="store_true", help="TPDF-dither one-shot assets (loops stay periodic)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("bench", help="measure synthesis throughput (no audio hardware needed)")
    p.add_argument("--preset", nargs="+", required=True)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--block", type=int, default=512)
    p.add_argument("--sr", type=int)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("realtime", help="stream to speakers (needs 'sounddevice')")
    p.add_argument("--preset", nargs="+", required=True)
    p.add_argument("--seconds", type=float)
    p.add_argument("--block", type=int, default=512)
    p.add_argument("--sr", type=int)
    p.add_argument("--rpm", type=float)
    p.add_argument("--load", type=float, default=0.5)
    p.add_argument("--throttle", type=float, default=0.5)
    p.add_argument("--interactive", action="store_true", help="drive with the keyboard")
    p.set_defaults(func=cmd_realtime)

    p = sub.add_parser("selftest", help="run built-in sanity checks")
    p.add_argument("--out", default="out/_selftest")
    p.set_defaults(func=cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

"""Build the native DSP core (``_dspcore.pyd``).

The C++ core is optional: :mod:`soundgen.native` falls back to the numpy path
when the extension is missing, so a checkout without a compiler still runs.
This script makes the native path available:

    python tools/soundgen/native/build.py            # build if needed
    python tools/soundgen/native/build.py --force    # rebuild
    python tools/soundgen/native/build.py --check    # exit 1 if it is missing

It drives MSVC directly (``cl.exe``) rather than setuptools, because the module
is one translation unit with no dependencies beyond pybind11 and doing it in
place keeps the artefact next to its source instead of in ``build/``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "dspcore.cpp"
GAS_SOURCE = HERE / "gascore.cpp"
# The extension is imported as a top-level module, so it has to sit where
# ``tools/soundgen`` is already on sys.path (next to the ``soundgen`` package).
OUT_DIR = HERE.parent
TARGET = OUT_DIR / ("_dspcore" + (sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"))
STALE = HERE / ("_dspcore" + (sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"))
GAS_TARGET = OUT_DIR / ("_gascore" + (sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"))

# Visual Studio installs found on this machine and the usual default location.
_VS_CANDIDATES = (
    Path(r"Z:\!VSStudio\Cash"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Professional"),
    Path(r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise"),
    Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community"),
)


def _pybind_include() -> str | None:
    try:
        import pybind11

        return pybind11.get_include()
    except Exception:
        return None


def _vcvars(root: Path) -> Path | None:
    for rel in ("VC/Auxiliary/Build/vcvars64.bat", "VC/Auxiliary/Build/vcvarsall.bat"):
        candidate = root / rel
        if candidate.is_file():
            return candidate
    return None


def _find_vcvars() -> Path | None:
    forced = os.environ.get("RGBBX_VCVARS")
    if forced and Path(forced).is_file():
        return Path(forced)
    for root in _VS_CANDIDATES:
        if not root.is_dir():
            continue
        found = _vcvars(root)
        if found is not None:
            return found
    return None


def _sources() -> tuple[list[str], str, str]:
    """Include dirs, library dir, library name."""
    py_inc = sysconfig.get_paths()["include"]
    pb_inc = _pybind_include()
    if pb_inc is None:
        raise SystemExit("pybind11 is not installed: python -m pip install pybind11")
    libs = sysconfig.get_config_var("LIBDIR") or str(Path(sys.base_prefix) / "libs")
    return [str(HERE), py_inc, pb_inc], libs, "python" + "".join(
        str(sys.version_info[i]) for i in range(2)
    )


def _compile(source: Path, target: Path, obj_name: str, vcvars: Path, incs, libdir, libname):
    obj = HERE / obj_name
    cmd = ["cl.exe", "/nologo", "/O2", "/std:c++17", "/EHsc", "/MD", "/LD",
           "/DNDEBUG", "/D_CRT_SECURE_NO_WARNINGS", "/favor:smallCode"]
    # Stage 6: the live audio callback runs at 64 frames (1.33 ms at 48 kHz).
    # The convolver work per call is dominated by fixed-size FFTs, and each
    # Python->C++ crossing costs ~1 us of marshalling, so /favor:smallCode
    # keeps the hot loops in the near cache and the GIL never comes into it.
    cmd += ["/I" + inc for inc in incs]
    cmd += [str(source), "/Fe:" + str(target), "/Fo:" + str(obj),
            "/link", "/LIBPATH:" + libdir, libname + ".lib"]
    bat = HERE / "_build.bat"
    bat.write_text(
        "@echo off\r\n"
        f'call "{vcvars}" >nul 2>&1\r\n'
        + " ".join(f'"{c}"' for c in cmd)
        + "\r\nexit /b %ERRORLEVEL%\r\n",
        encoding="utf-8",
    )
    try:
        proc = subprocess.run(
            ["cmd.exe", "/c", str(bat)], cwd=str(HERE), capture_output=True, text=True
        )
    finally:
        bat.unlink(missing_ok=True)
    if proc.returncode != 0 or not target.is_file():
        raise SystemExit(
            f"native build failed ({source.name})\n"
            + (proc.stdout or "") + (proc.stderr or "")
        )
    obj.unlink(missing_ok=True)
    # cl writes the .pyd plus an import library; drop the extra artefacts
    stem = target.stem
    for extra in OUT_DIR.glob(stem + "*.lib"):
        extra.unlink(missing_ok=True)
    for extra in OUT_DIR.glob(stem + "*.exp"):
        extra.unlink(missing_ok=True)


def build(force: bool = False) -> Path:
    vcvars = _find_vcvars()
    if vcvars is None:
        raise SystemExit(
            "no MSVC found; set RGBBX_VCVARS to VC/Auxiliary/Build/vcvars64.bat"
        )
    incs, libdir, libname = _sources()

    if force or not TARGET.is_file() or TARGET.stat().st_mtime < SOURCE.stat().st_mtime:
        _compile(SOURCE, TARGET, "dspcore.obj", vcvars, incs, libdir, libname)
    if STALE.is_file():
        STALE.unlink(missing_ok=True)

    # gas core: rebuilt when its source changes or the artefact is missing
    if force or not GAS_TARGET.is_file() or GAS_TARGET.stat().st_mtime < GAS_SOURCE.stat().st_mtime:
        _compile(GAS_SOURCE, GAS_TARGET, "gascore.obj", vcvars, incs, libdir, libname)
    return TARGET


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="build the native DSP core")
    ap.add_argument("--force", action="store_true", help="rebuild even if up to date")
    ap.add_argument("--check", action="store_true", help="only report whether it exists")
    args = ap.parse_args(argv)
    if args.check:
        print(TARGET if TARGET.is_file() else "missing")
        return 0 if TARGET.is_file() else 1
    path = build(force=args.force)
    print(f"built {path.name}")
    sys.path.insert(0, str(OUT_DIR))
    import importlib

    import numpy as np

    mod = importlib.import_module("_dspcore")
    print(f"  {mod.__doc__.splitlines()[0]}")
    print(f"  fft size for block 256: {mod.Convolver(np.array([1.0, 0.5]), 256).fft_size}")
    print(f"  probe rfft ok: {len(mod.rfft(np.zeros(512)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

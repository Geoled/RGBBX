# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build: a portable EngineLab folder for Windows x64.

Layout produced by ``build_exe.bat``:

    dist/EngineLab/
        EngineLab.exe          the app (no console window)
        EngineLabDebug.exe     same app with a console, for diagnostics
        _internal/             interpreter, DLLs, imgui assets, presets ...

The app finds its data relative to the bundle, so the folder can be copied
anywhere (USB stick, another PC) and started without Python installed.
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH)
SOUNDGEN = ROOT / "tools" / "soundgen"

datas = [
    # Engine presets: bootstrap.preset_dir() resolves to this path in a
    # frozen app (..\tools\soundgen\presets next to the simulator package).
    (str(SOUNDGEN / "presets"), "tools/soundgen/presets"),
]

binaries = []
for pyd in sorted(SOUNDGEN.glob("_dspcore*.pyd")):
    # soundgen.native does importlib.import_module("_dspcore"): the module
    # name carries the ABI tag, so it must ship as a plain binary and be
    # importable from the bundle root (launcher.py puts it on sys.path).
    binaries.append((str(pyd), "."))

hiddenimports = [
    name for name in collect_submodules("soundgen") if not name.endswith("gasbank")
]

# imgui_bundle carries native libraries and the fonts hello_imgui falls back
# to when a system font is missing; collect_all is the supported way to ship
# the whole binding. moderngl is a single extension but collect_all is cheap.
for package in ("imgui_bundle", "moderngl"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

a = Analysis(
    ["launcher.py"],
    pathex=[str(ROOT), str(SOUNDGEN)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # The tkinter window in simulator/ui is a retired alternate front end;
        # the shipped app is the imgui one. Dropping tkinter saves ~10 MB and
        # a pile of Tcl diagnostics on machines with a broken Tcl install.
        "tkinter",
        "pytest",
        "matplotlib",
        "IPython",
        "pyloudnorm",  # optional LUFS metering in the offline tool
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EngineLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

dbg = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EngineLabDebug",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    dbg,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="EngineLab",
)

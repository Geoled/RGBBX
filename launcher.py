"""Frozen-app entry point: the same CLI as ``python -m simulator``.

PyInstaller imports this module instead of running the package with ``-m``,
so it has to do the two things the package machinery would do for us:

* put the bundle directory on ``sys.path`` -- ``soundgen.native`` loads
  ``_dspcore`` by name at import time, and an ABI-tagged extension module
  cannot be a static hidden import;
* delegate to ``simulator.__main__.main`` so every flag (``--preset``,
  ``--selftest``, ``--capture`` ...) behaves exactly as in a source checkout.
"""

from __future__ import annotations

import sys


def _bootstrap_path() -> None:
    import os

    base = getattr(sys, "_MEIPASS", None)
    if base and base not in sys.path:
        sys.path.insert(0, base)
    # One-file builds unpack to a temp dir; keep it on the path as well.
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


def main() -> int:
    _bootstrap_path()
    from simulator.__main__ import main as run

    return run()


if __name__ == "__main__":
    sys.exit(main())

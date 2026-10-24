"""Make the sibling ``tools/soundgen`` package importable.

The sound pipeline is developed and tested as a standalone tool, so it is not
installed into site-packages. The simulator imports it directly from the
repository instead of copying code out of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOUNDGEN_DIR = REPO_ROOT / "tools" / "soundgen"

if SOUNDGEN_DIR.is_dir() and str(SOUNDGEN_DIR) not in sys.path:
    sys.path.insert(0, str(SOUNDGEN_DIR))


def preset_dir() -> Path:
    """Directory holding the bundled engine presets."""
    return SOUNDGEN_DIR / "presets"

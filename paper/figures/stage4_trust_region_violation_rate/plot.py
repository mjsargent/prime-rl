from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from figure_utils import build_figure


if __name__ == "__main__":
    build_figure(Path(__file__).parent)

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    for plot in sorted(Path(__file__).parent.glob("figures/*/plot.py")):
        subprocess.run([sys.executable, "plot.py"], cwd=plot.parent, check=True)


if __name__ == "__main__":
    main()

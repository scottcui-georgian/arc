"""
Workspace entrypoint for challenge data download (CPU / Modal prepare).
"""

from __future__ import annotations

import runpy
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "data" / "cached_challenge_fineweb.py"


def main() -> None:
    runpy.run_path(str(_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()

"""Wrapper for scripts/check_data.py."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts" / "check_data.py"), run_name="__main__")


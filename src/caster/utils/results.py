"""Run-directory resolution for result aggregation.

A run directory is named ``{method}-{timestamp}``. Selecting runs with the glob
``{method}-*`` is unsafe: ``caster-*`` also matches ``caster-no-gate-*`` and
``caster-no-transport-*``. Every selector here matches on the ``method`` field
recorded inside ``summary.json``, which is written by the run itself and is the
authoritative record of what was executed.
"""

from __future__ import annotations

import json
from pathlib import Path


def summary_method(path: str | Path) -> str | None:
    """Return the method recorded in a summary.json, or None if unreadable."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    method = payload.get("method")
    return str(method) if method is not None else None


def iter_method_summaries(output_root: str | Path, method: str) -> list[Path]:
    """Return every summary.json under output_root recorded as exactly `method`.

    Ordered by run-directory name, which is chronological because run directories
    are suffixed with a sortable timestamp.
    """
    root = Path(output_root)
    if not root.is_dir():
        return []
    matches = [
        path
        for path in root.glob("*/summary.json")
        if summary_method(path) == method
    ]
    return sorted(matches, key=lambda path: path.parent.name)


def latest_method_summary(output_root: str | Path, method: str) -> Path | None:
    """Return the most recent summary.json for exactly `method`, or None."""
    matches = iter_method_summaries(output_root, method)
    return matches[-1] if matches else None


def has_method_run(output_root: str | Path, method: str) -> bool:
    """Return whether a completed run for exactly `method` exists."""
    return bool(iter_method_summaries(output_root, method))

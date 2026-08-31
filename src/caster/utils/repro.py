"""Reproducibility and metadata helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import socket
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_hash(cwd: str | Path = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "nogit"


def gpu_info() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    info = []
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        info.append(
            {
                "index": idx,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
            }
        )
    return info


def package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    distributions = {
        "torchvision": "torchvision",
        "timm": "timm",
        "scipy": "scipy",
        "sklearn": "scikit-learn",
        "pandas": "pandas",
        "yaml": "PyYAML",
        "matplotlib": "matplotlib",
    }
    for name, distribution in distributions.items():
        try:
            versions[name] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def run_metadata(config: dict[str, Any], *, seed: int, command: list[str]) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "command": command,
        "config": config,
        "git_hash": git_hash(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "gpu_info": gpu_info(),
        "package_versions": package_versions(),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

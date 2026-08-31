#!/usr/bin/env python3
"""Validate a dataset root and report basic metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from torch.utils.data import DataLoader

from caster.data import build_dataset, build_transform


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--root", default="./data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--severity", type=int, default=1)
    args = parser.parse_args()

    family = "cifar" if "cifar" in args.name or args.name == "fake" else "imagenet"
    transform = build_transform(image_size=args.image_size, dataset_family=family)
    dataset = build_dataset(
        name=args.name,
        root=Path(args.root),
        split=args.split,
        transform=transform,
        max_samples=args.max_samples,
        num_classes=args.num_classes,
        image_size=args.image_size,
        corruption=args.corruption,
        severity=args.severity,
    )
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)))
    images, labels = next(iter(loader))
    payload = {
        "name": args.name,
        "root": str(Path(args.root)),
        "split": args.split,
        "num_examples_checked": len(dataset),
        "first_batch_shape": list(images.shape),
        "first_labels": labels.tolist(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

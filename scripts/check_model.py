#!/usr/bin/env python3
"""Load a model and run one feature/logit pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from caster.models import create_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny_cnn")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--non-strict-checkpoint", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    feature_model = create_model(
        args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        feature_dim=args.feature_dim,
        checkpoint_path=args.checkpoint_path,
        strict_checkpoint=not args.non_strict_checkpoint,
    ).to(device).eval()
    images = torch.randn(2, 3, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        logits, features = feature_model.logits_and_features(images)
    total_params = sum(p.numel() for p in feature_model.model.parameters())
    trainable_params = sum(p.numel() for p in feature_model.model.parameters() if p.requires_grad)
    payload = {
        "model": args.model,
        "checkpoint_path": args.checkpoint_path,
        "logits_shape": list(logits.shape),
        "features_shape": list(features.shape),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

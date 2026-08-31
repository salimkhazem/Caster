#!/usr/bin/env python3
"""Extract CASTER source statistics from labeled source data."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from caster.data import build_dataset, build_transform
from caster.methods.caster import CasterConfig, compute_source_statistics_from_features
from caster.models import create_model
from caster.utils.config import load_yaml
from caster.utils.repro import seed_everything


def collect_features(config: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    family = "cifar" if "cifar" in dataset_cfg["name"] or dataset_cfg["name"] == "fake" else "imagenet"
    transform = build_transform(
        image_size=int(dataset_cfg.get("image_size", 224)),
        train=False,
        dataset_family=family,
    )
    if dataset_cfg.get("source_name"):
        source_name = dataset_cfg["source_name"]
    elif dataset_cfg["name"] == "fake":
        source_name = "fake"
    elif dataset_cfg["name"] in {"cifar10c", "cifar100c"}:
        source_name = "cifar10" if dataset_cfg["name"] == "cifar10c" else "cifar100"
    else:
        source_name = dataset_cfg["name"]
    dataset = build_dataset(
        name=source_name,
        root=dataset_cfg.get("source_root", dataset_cfg["root"]),
        split=dataset_cfg.get("source_split", "train"),
        transform=transform,
        max_samples=dataset_cfg.get("max_source_samples"),
        num_classes=int(dataset_cfg["num_classes"]),
        image_size=int(dataset_cfg.get("image_size", 224)),
        domain=dataset_cfg.get("source_domain"),
        partition=dataset_cfg.get("partition", 1),
        download=bool(dataset_cfg.get("download", False)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(dataset_cfg.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
    )
    model = create_model(
        model_cfg["name"],
        num_classes=int(dataset_cfg["num_classes"]),
        pretrained=bool(model_cfg.get("pretrained", True)),
        feature_dim=int(model_cfg.get("feature_dim", 32)),
        checkpoint_path=model_cfg.get("checkpoint_path"),
        strict_checkpoint=bool(model_cfg.get("strict_checkpoint", True)),
    ).to(device).eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in tqdm(loader, desc="source features"):
            images = images.to(device)
            _, batch_features = model.logits_and_features(images)
            features.append(batch_features.detach().cpu())
            labels.append(batch_labels.detach().cpu())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    seed_everything(int(config.get("seed", 1)))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    features, labels = collect_features(config, device)
    caster_cfg = CasterConfig(**config.get("caster", {}))
    stats = compute_source_statistics_from_features(
        features,
        labels,
        num_classes=int(config["dataset"]["num_classes"]),
        subspace_dim=caster_cfg.subspace_dim,
        lambda_shrinkage=caster_cfg.lambda_shrinkage,
        beta_noise=caster_cfg.beta_noise,
    )
    output = Path(
        args.output
        or config.get("source_stats_path")
        or Path(config.get("output_root", "results/raw")) / "source_stats.pt"
    )
    stats.save(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

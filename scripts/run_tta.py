#!/usr/bin/env python3
"""Run one CASTER/baseline evaluation job."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import csv
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from caster.baselines import BaselineOutput, create_baseline
from caster.baselines.factory import split_gated_name
from caster.methods.gate import CertifiedGate, GateConfig
from caster.data import build_dataset, build_transform
from caster.eval.runner import evaluate_predictions, save_evaluation_summary
from caster.methods.caster import CasterAdapter, CasterConfig, SourceStats, compute_source_statistics_from_features
from caster.models import create_model
from caster.utils.config import load_yaml
from caster.utils.repro import run_metadata, seed_everything


def build_loaders(config: dict) -> tuple[DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    family = "cifar" if "cifar" in dataset_cfg["name"] or dataset_cfg["name"] == "fake" else "imagenet"
    transform = build_transform(
        image_size=int(dataset_cfg.get("image_size", 224)),
        train=False,
        dataset_family=family,
    )
    target = build_dataset(
        name=dataset_cfg["name"],
        root=dataset_cfg["root"],
        split=dataset_cfg.get("target_split", "test"),
        transform=transform,
        max_samples=dataset_cfg.get("max_target_samples"),
        num_classes=int(dataset_cfg["num_classes"]),
        image_size=int(dataset_cfg.get("image_size", 224)),
        corruption=dataset_cfg.get("corruption", "gaussian_noise"),
        severity=int(dataset_cfg.get("severity", 1)),
        synthetic_corruption=dataset_cfg.get("synthetic_corruption"),
        synthetic_severity=int(dataset_cfg.get("synthetic_severity", dataset_cfg.get("severity", 1))),
        synthetic_seed=int(dataset_cfg.get("synthetic_seed", config.get("seed", 0))),
        domain=dataset_cfg.get("target_domain"),
        partition=dataset_cfg.get("partition", 1),
        download=bool(dataset_cfg.get("download", False)),
    )
    source_name = dataset_cfg.get("source_name")
    if source_name is None:
        if dataset_cfg["name"] == "fake":
            source_name = "fake"
        elif dataset_cfg["name"] == "cifar10c":
            source_name = "cifar10"
        elif dataset_cfg["name"] == "cifar100c":
            source_name = "cifar100"
        else:
            source_name = dataset_cfg["name"]
    source = build_dataset(
        name=source_name,
        root=dataset_cfg.get("source_root", dataset_cfg["root"]),
        split=dataset_cfg.get("source_split", "train"),
        transform=transform,
        max_samples=dataset_cfg.get("max_source_samples"),
        num_classes=int(dataset_cfg["num_classes"]),
        # Arrow-backed sources sample a fixed number of images per class rather
        # than reading the whole corpus; ignored by the other loaders.
        per_class=int(dataset_cfg.get("source_per_class", 50)),
        image_size=int(dataset_cfg.get("image_size", 224)),
        domain=dataset_cfg.get("source_domain"),
        partition=dataset_cfg.get("partition", 1),
        download=bool(dataset_cfg.get("download", False)),
    )
    # batch_size is deliberately NOT overridable here: CASTER's transport
    # conditioning depends on the batch-to-subspace ratio, so changing it changes
    # the method rather than only its speed. Dataloader parallelism is free to
    # vary, and must when many jobs share one machine.
    loader_kwargs = {
        "batch_size": int(dataset_cfg.get("batch_size", 64)),
        "num_workers": int(os.environ.get("CASTER_NUM_WORKERS", dataset_cfg.get("num_workers", 0))),
        "pin_memory": torch.cuda.is_available(),
    }
    # Test-time adaptation assumes an i.i.d. test stream. ImageNet-C is stored in
    # class order, so an unshuffled loader yields 2-3 of 1000 classes per batch and
    # BatchNorm batch statistics become meaningless -- worth 78 points on
    # brightness s1 with no gradient step at all. Default False keeps the
    # completed CIFAR matrix reproducible (its stream is already class-diverse);
    # ImageNet configs set it True.
    shuffle_target = bool(dataset_cfg.get("shuffle_target", False))
    generator = torch.Generator()
    generator.manual_seed(int(config.get("seed", 0)))
    return (
        DataLoader(source, shuffle=False, **loader_kwargs),
        DataLoader(target, shuffle=shuffle_target, generator=generator, **loader_kwargs),
    )


def measure_batch_diversity(loader: DataLoader, probe_batches: int = 20) -> dict:
    """Record what the target stream actually looks like, per batch.

    Returned in every summary.json. A result that depends on batch composition
    as strongly as BatchNorm adaptation does should carry the composition with
    it, so a reader can tell an i.i.d. stream from a class-ordered one without
    re-deriving it from the dataset layout.
    """
    generator = getattr(loader, "generator", None)
    state = generator.get_state() if generator is not None else None
    distinct, dominant, seen = [], [], 0
    for _images, labels in loader:
        counts = Counter(labels.tolist())
        distinct.append(len(counts))
        dominant.append(max(counts.values()) / max(len(labels), 1))
        seen += 1
        if seen >= probe_batches:
            break
    if state is not None:
        generator.set_state(state)   # leave the run's batch order untouched
    if not distinct:
        return {}
    return {
        "probed_batches": seen,
        "mean_distinct_classes": sum(distinct) / len(distinct),
        "min_distinct_classes": min(distinct),
        "max_dominant_class_share": max(dominant),
    }


def assert_stream_is_usable(diversity: dict, config: dict) -> None:
    """Refuse a degenerate stream unless the config chose it deliberately."""
    if not diversity:
        return
    dataset_cfg = config["dataset"]
    if "shuffle_target" in dataset_cfg:      # an explicit, recorded choice
        return
    batch = int(dataset_cfg.get("batch_size", 64))
    reachable = min(int(dataset_cfg.get("num_classes", 1)), batch)
    if reachable < 8:                        # too few classes for this to mean anything
        return
    if diversity["mean_distinct_classes"] < 0.25 * reachable:
        raise SystemExit(
            f"target stream is class-ordered: {diversity['mean_distinct_classes']:.1f} "
            f"distinct classes per batch of {batch} out of {reachable} reachable, "
            f"largest single class {diversity['max_dominant_class_share']:.0%} of a batch.\n"
            "Batch-statistic methods measure stream order rather than adaptation on "
            "such a stream. Set dataset.shuffle_target explicitly (true for the "
            "standard i.i.d. protocol, false to study the non-i.i.d. setting on "
            "purpose)."
        )


def build_or_load_stats(
    config: dict,
    source_loader: DataLoader,
    feature_model,
    device: torch.device,
) -> SourceStats:
    stats_path = config.get("source_stats_path")
    if stats_path and Path(stats_path).exists():
        loaded = SourceStats.load(stats_path)
        requested = int(CasterConfig(**config.get("caster", {})).subspace_dim)
        found = int(loaded.metadata.get("subspace_dim", -1))
        if found not in (-1, requested):
            print(
                f"WARNING: {stats_path} holds subspace_dim={found} but the config "
                f"requests {requested}; the loaded value wins. Repoint "
                f"source_stats_path or delete the cache.",
                file=sys.stderr,
            )
        return loaded
    features = []
    labels = []
    with torch.no_grad():
        for images, batch_labels in source_loader:
            images = images.to(device)
            _, batch_features = feature_model.logits_and_features(images)
            features.append(batch_features.detach().cpu())
            labels.append(batch_labels.detach().cpu())
    caster_cfg = CasterConfig(**config.get("caster", {}))
    stats = compute_source_statistics_from_features(
        torch.cat(features, dim=0),
        torch.cat(labels, dim=0),
        num_classes=int(config["dataset"]["num_classes"]),
        subspace_dim=caster_cfg.subspace_dim,
        lambda_shrinkage=caster_cfg.lambda_shrinkage,
        beta_noise=caster_cfg.beta_noise,
    )
    # Write back to the path we READ from. These previously diverged -- the
    # lookup used config["source_stats_path"] while the save went to
    # output_root/source_stats.pt -- so the cache was never populated where it
    # was looked for and every run recomputed source statistics from scratch,
    # a full pass over the source training set per job.
    output = Path(stats_path) if stats_path else (
        Path(config.get("output_root", "results/raw")) / "source_stats.pt"
    )
    # Concurrent workers legitimately race for the same statistics file, since
    # it is shared across every corruption for a given (source, backbone, seed).
    # Write to a private temporary and rename, which is atomic within a
    # filesystem, so a reader never observes a partial file.
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        stats.save(tmp)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default="caster")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    seed = int(config.get("seed", 1))
    seed_everything(seed)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    source_loader, target_loader = build_loaders(config)
    stats = None
    stream_diversity = measure_batch_diversity(target_loader)
    assert_stream_is_usable(stream_diversity, config)
    model_cfg = config["model"]
    feature_model = create_model(
        model_cfg["name"],
        num_classes=int(config["dataset"]["num_classes"]),
        pretrained=bool(model_cfg.get("pretrained", True)),
        feature_dim=int(model_cfg.get("feature_dim", 32)),
        checkpoint_path=model_cfg.get("checkpoint_path"),
        strict_checkpoint=bool(model_cfg.get("strict_checkpoint", True)),
    ).to(device).eval()

    # Harm rate is defined against the *unadapted* source model. Tent and EATA
    # mutate `feature_model` in place, so reading the reference prediction from
    # it would compare the adapted model against itself one step earlier and
    # report harm ~0 by construction, which is not comparable to a non-mutating
    # method like CASTER. Keep a pristine copy that nothing is allowed to touch.
    frozen_model = copy.deepcopy(feature_model).to(device).eval()
    for _param in frozen_model.model.parameters():
        _param.requires_grad_(False)

    def source_predict(images: torch.Tensor) -> BaselineOutput:
        with torch.no_grad():
            logits, _ = frozen_model.logits_and_features(images)
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        abstain = torch.zeros_like(pred, dtype=torch.bool)
        return BaselineOutput(logits, probs, pred, abstain, False, "source_only")

    method_name = args.method.lower()
    caster_family = {
        "caster",
        "source-gda",
        "caster-no-gate",
        "caster-no-transport",
        "caster-legacy-cert",
    }
    if method_name in caster_family:
        stats = build_or_load_stats(config, source_loader, feature_model, device)
        caster_config = dict(config.get("caster", {}))
        if method_name == "source-gda":
            # Pure source-statistic baseline: never touch the model head, or this
            # collapses into source-only and stops being a fair GDA comparison.
            caster_config.update(
                {
                    "use_transport": False,
                    "use_gate": False,
                    "fallback_policy": "source",
                    "fallback_to_head": False,
                }
            )
        elif method_name == "caster-no-gate":
            caster_config["use_gate"] = False
        elif method_name == "caster-no-transport":
            caster_config.update(
                {"use_transport": False, "fallback_policy": "source", "fallback_to_head": False}
            )
        elif method_name == "caster-legacy-cert":
            # Reproduces the pre-fix certificate: Euclidean residual against a
            # Mahalanobis margin, no head agreement, GDA fallback. Kept as the
            # ablation baseline showing why the corrected certificate is needed.
            caster_config.update(
                {
                    "certificate_metric": "euclidean",
                    "use_head_agreement": False,
                    "transport_form": "full",
                    "fallback_to_head": False,
                    # Pin the legacy threshold too: tau is metric-dependent, and
                    # the point of this variant is to reproduce the pre-fix
                    # behaviour exactly, not to mix an old metric with a new tau.
                    "tau_gate": 0.2,
                }
            )
        adapter = CasterAdapter(stats, CasterConfig(**caster_config)).to(device)

        def predict(images: torch.Tensor):
            with torch.no_grad():
                head_logits, features = feature_model.logits_and_features(images)
                return adapter.predict_from_features(features, head_logits=head_logits)

    else:
        inner_name, gated = split_gated_name(method_name)
        baseline = create_baseline(inner_name, **config.get("baselines", {}).get(inner_name, {}))

        if gated:
            # The gate needs source statistics to compute the certificate, so the
            # same source-stat artifact the CASTER family uses is required here.
            stats = build_or_load_stats(config, source_loader, feature_model, device)
            adapter = CasterAdapter(stats, CasterConfig(**config.get("caster", {}))).to(device)
            gate_kwargs = dict(config.get("gate", {}))
            baseline = CertifiedGate(
                baseline, adapter, GateConfig(**gate_kwargs), reference_model=frozen_model
            )

        def predict(images: torch.Tensor):
            return baseline.predict_batch(images, feature_model)

    output_root = Path(config.get("output_root", "results/raw"))
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"{method_name}-{timestamp}"
    raw_jsonl = run_dir / "batches.jsonl"
    result = evaluate_predictions(
        method_name=method_name,
        dataloader=target_loader,
        predict_fn=predict,
        source_predict_fn=source_predict,
        device=device,
        max_steps=args.max_steps,
        output_jsonl=raw_jsonl,
    )
    payload = {
        "metadata": run_metadata(config, seed=seed, command=sys.argv),
        "method": method_name,
        "summary": result.summary,
        "stream": stream_diversity,
        # A run stopped early is not comparable with a complete one. Recorded so
        # aggregation can drop it rather than average it in silently.
        "truncated": args.max_steps is not None,
        "max_steps": args.max_steps,
        # What the run actually loaded, not what the config asked for.
        "source_stats": {
            "path": config.get("source_stats_path"),
            **({} if stats is None else dict(stats.metadata)),
        },
        "raw_jsonl": str(raw_jsonl),
    }
    summary_path = run_dir / "summary.json"
    save_evaluation_summary(summary_path, payload)
    csv_path = run_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(result.summary))
        writer.writeheader()
        writer.writerow(result.summary)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

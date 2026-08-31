"""Shared evaluation loop."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from caster.metrics import classification_summary
from caster.utils.repro import write_json


@dataclass
class EvaluationResult:
    summary: dict[str, Any]
    batches: list[dict[str, Any]]


def evaluate_predictions(
    *,
    method_name: str,
    dataloader: Iterable,
    predict_fn,
    source_predict_fn,
    device: torch.device,
    max_steps: int | None = None,
    output_jsonl: str | Path | None = None,
) -> EvaluationResult:
    """Evaluate a method and optionally write per-batch raw logs."""
    all_logits = []
    all_source_logits = []
    all_labels = []
    all_abstain = []
    batches: list[dict[str, Any]] = []
    start = time.perf_counter()
    num_images = 0

    handle = None
    if output_jsonl is not None:
        output_path = Path(output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", encoding="utf-8")

    try:
        for step, batch in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break
            images, labels = batch[:2]
            images = images.to(device)
            labels = labels.to(device)
            source_out = source_predict_fn(images)
            method_out = predict_fn(images)
            method_predictions = method_out.logits.argmax(dim=-1)
            source_predictions = source_out.logits.argmax(dim=-1)
            method_correct = method_predictions.eq(labels)
            source_correct = source_predictions.eq(labels)
            harm_mask = source_correct & ~method_correct
            method_confidence = torch.softmax(method_out.logits, dim=-1).max(dim=-1).values
            source_confidence = torch.softmax(source_out.logits, dim=-1).max(dim=-1).values
            all_logits.append(method_out.logits.detach().cpu())
            all_source_logits.append(source_out.logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_abstain.append(method_out.abstain_mask.detach().cpu())
            num_images += int(labels.numel())
            batch_payload = {
                "step": step,
                "method": method_name,
                "num_examples": int(labels.numel()),
                "adapted": bool(getattr(method_out, "adapted", False)),
                "reason": str(getattr(method_out, "reason", "")),
                "certificate": float(getattr(method_out, "certificate", float("nan"))),
                # `adapted` is true for a prototype fallback as well as an accepted
                # transport, so it cannot be read as a transport-acceptance rate.
                "transport_accepted": bool(getattr(method_out, "transport_accepted", False)),
                "certificate_euclidean": float(
                    getattr(method_out, "certificate_euclidean", float("nan"))
                ),
                "certificate_mahalanobis": float(
                    getattr(method_out, "certificate_mahalanobis", float("nan"))
                ),
                "head_disagreement": float(getattr(method_out, "head_disagreement", float("nan"))),
                "transport_form": str(getattr(method_out, "transport_form", "")),
                "confident_total": int(getattr(method_out, "confident_total", 0)),
                "valid_classes": int(getattr(method_out, "valid_classes", 0)),
                "method_top1": float(method_correct.float().mean().item()),
                "source_top1": float(source_correct.float().mean().item()),
                "top1_delta": float((method_correct.float().mean() - source_correct.float().mean()).item()),
                "harm_rate": float(harm_mask.float().mean().item()),
                "method_confidence": float(method_confidence.mean().item()),
                "source_confidence": float(source_confidence.mean().item()),
                "abstention_rate": float(method_out.abstain_mask.float().mean().item()),
            }
            batches.append(batch_payload)
            if handle is not None:
                import json

                handle.write(json.dumps(batch_payload, sort_keys=True) + "\n")
    finally:
        if handle is not None:
            handle.close()

    elapsed = time.perf_counter() - start
    if not all_logits:
        raise RuntimeError("evaluation produced no batches")
    logits = torch.cat(all_logits, dim=0)
    source_logits = torch.cat(all_source_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    abstain = torch.cat(all_abstain, dim=0)
    metrics = classification_summary(logits, labels, source_logits=source_logits, abstain_mask=abstain)
    metrics["num_examples"] = int(num_images)
    metrics["elapsed_sec"] = float(elapsed)
    metrics["throughput_img_s"] = float(num_images / max(elapsed, 1e-12))
    if torch.cuda.is_available() and device.type == "cuda":
        metrics["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
    else:
        metrics["peak_memory_bytes"] = 0
    metrics["adaptation_rate"] = float(sum(1 for b in batches if b["adapted"]) / max(len(batches), 1))
    # Distinct from adaptation_rate: counts only batches where the certificate
    # accepted the affine transport, excluding prototype/head fallbacks.
    metrics["transport_accept_rate"] = float(
        sum(1 for b in batches if b.get("transport_accepted")) / max(len(batches), 1)
    )
    finite_certificates = [b["certificate"] for b in batches if b["certificate"] == b["certificate"] and b["certificate"] != float("inf")]
    metrics["mean_certificate"] = float(sum(finite_certificates) / len(finite_certificates)) if finite_certificates else float("nan")
    for key in ("certificate_euclidean", "certificate_mahalanobis", "head_disagreement"):
        finite = [b[key] for b in batches if key in b and b[key] == b[key] and b[key] != float("inf")]
        metrics[f"mean_{key}"] = float(sum(finite) / len(finite)) if finite else float("nan")
    return EvaluationResult(summary=metrics, batches=batches)


def save_evaluation_summary(path: str | Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)

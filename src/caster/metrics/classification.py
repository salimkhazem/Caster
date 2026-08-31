"""Classification, calibration, and safety metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, topk: tuple[int, ...] = (1,)) -> dict[str, float]:
    labels = labels.long()
    maxk = min(max(topk), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1))
    out = {}
    for k in topk:
        kk = min(k, logits.shape[1])
        out[f"top{kk}"] = float(correct[:kk].reshape(-1).float().sum().item() / labels.numel())
    return out


def negative_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(logits, labels.long(), reduction="mean").item())


def brier_score(logits: torch.Tensor, labels: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    target = F.one_hot(labels.long(), num_classes=logits.shape[1]).to(probs.dtype)
    return float(((probs - target) ** 2).sum(dim=-1).mean().item())


def calibration_error(logits: torch.Tensor, labels: torch.Tensor, *, bins: int = 15) -> float:
    probs = torch.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    correct = pred.eq(labels.long()).float()
    ece = torch.zeros((), dtype=logits.dtype, device=logits.device)
    edges = torch.linspace(0, 1, bins + 1, device=logits.device, dtype=logits.dtype)
    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.any():
            ece += mask.float().mean() * (correct[mask].mean() - conf[mask].mean()).abs()
    return float(ece.item())


def harm_rate(method_logits: torch.Tensor, source_logits: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.long()
    method_correct = method_logits.argmax(dim=-1).eq(labels)
    source_correct = source_logits.argmax(dim=-1).eq(labels)
    harmed = source_correct & ~method_correct
    return float(harmed.float().mean().item())


def selective_risk(logits: torch.Tensor, labels: torch.Tensor, abstain_mask: torch.Tensor | None = None) -> float:
    labels = labels.long()
    if abstain_mask is None:
        abstain_mask = torch.zeros(labels.shape, dtype=torch.bool, device=labels.device)
    keep = ~abstain_mask
    if not keep.any():
        return float("nan")
    incorrect = ~logits[keep].argmax(dim=-1).eq(labels[keep])
    return float(incorrect.float().mean().item())


def classification_summary(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    source_logits: torch.Tensor | None = None,
    abstain_mask: torch.Tensor | None = None,
    ece_bins: int = 15,
) -> dict[str, float]:
    metrics = topk_accuracy(logits, labels, topk=(1, 5))
    metrics["nll"] = negative_log_likelihood(logits, labels)
    metrics["brier"] = brier_score(logits, labels)
    metrics["ece"] = calibration_error(logits, labels, bins=ece_bins)
    metrics["selective_risk"] = selective_risk(logits, labels, abstain_mask)
    if abstain_mask is not None:
        metrics["abstention_rate"] = float(abstain_mask.float().mean().item())
    else:
        metrics["abstention_rate"] = 0.0
    if source_logits is not None:
        metrics["harm_rate"] = harm_rate(logits, source_logits, labels)
    else:
        metrics["harm_rate"] = 0.0
    return metrics


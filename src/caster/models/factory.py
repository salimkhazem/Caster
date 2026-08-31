"""Model loading and feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


class TinyConvNet(nn.Module):
    """Small deterministic model for CPU smoke tests."""

    def __init__(self, num_classes: int = 10, feature_dim: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


@dataclass
class FeatureModel:
    """Wrap a classifier with a consistent logits/features interface."""

    model: nn.Module

    def to(self, device: torch.device | str) -> "FeatureModel":
        self.model.to(device)
        return self

    def eval(self) -> "FeatureModel":
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> "FeatureModel":
        self.model.train(mode)
        return self

    def logits_and_features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.model, "forward_features"):
            raw_features = self.model.forward_features(images)
            features = self._pooled_representation(raw_features)
            logits = self._logits_from_raw_features(raw_features, features)
            return logits, features
        logits = self.model(images)
        return logits, logits

    def _pooled_representation(self, raw_features: torch.Tensor) -> torch.Tensor:
        """Penultimate representation, using the model's own pooling when available.

        Shape-based pooling cannot be done safely for every backbone: timm's Swin
        returns NHWC ``(B, H, W, C)``, which is indistinguishable from NCHW by rank
        alone. Averaging dims (2, 3) there collapses width *and channels* and
        yields a 7-dimensional vector instead of 768 -- the reason swin_tiny sat
        at chance in every downstream result. `forward_head(pre_logits=True)` is
        the model's own definition of that representation, so prefer it.
        """
        if hasattr(self.model, "forward_head"):
            try:
                pooled = self.model.forward_head(raw_features, pre_logits=True)
                if isinstance(pooled, torch.Tensor) and pooled.ndim == 2:
                    return pooled
            except (TypeError, NotImplementedError):
                pass
        return self._pool_features(raw_features)

    def _logits_from_raw_features(self, raw_features: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_head"):
            try:
                logits = self.model.forward_head(raw_features, pre_logits=False)
                if logits.ndim == 2:
                    return logits
            except TypeError:
                try:
                    logits = self.model.forward_head(raw_features)
                    if logits.ndim == 2:
                        return logits
                except Exception:
                    pass
        classifier = getattr(self.model, "head", None) or getattr(self.model, "fc", None)
        if isinstance(classifier, nn.Module):
            return classifier(pooled)
        return self.model(raw_features)

    @staticmethod
    def _pool_features(features: torch.Tensor) -> torch.Tensor:
        """Fallback pooling for models without a usable ``forward_head``."""
        if features.ndim == 4:
            # Disambiguate NCHW from NHWC by which end looks like a channel axis.
            # A channel dim is normally far larger than a spatial dim at the final
            # stage (e.g. 2048x7x7 vs 7x7x768).
            _, d1, d2, d3 = features.shape
            if d3 > d1 and d2 == d1:
                return features.mean(dim=(1, 2))  # NHWC -> average over H, W
            return features.mean(dim=(2, 3))  # NCHW -> average over H, W
        if features.ndim == 3:
            return features[:, 0]
        return features.flatten(1)


def create_model(
    model_name: str,
    *,
    num_classes: int,
    pretrained: bool = True,
    feature_dim: int = 32,
    checkpoint_path: str | Path | None = None,
    strict_checkpoint: bool = True,
) -> FeatureModel:
    model_name = model_name.lower()
    if model_name == "tiny_cnn":
        model = TinyConvNet(num_classes=num_classes, feature_dim=feature_dim)
        _load_checkpoint_if_requested(model, checkpoint_path, strict=strict_checkpoint)
        return FeatureModel(model)
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("timm is required for non-tiny models") from exc
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    _load_checkpoint_if_requested(model, checkpoint_path, strict=strict_checkpoint)
    return FeatureModel(model)


def _load_checkpoint_if_requested(
    model: nn.Module,
    checkpoint_path: str | Path | None,
    *,
    strict: bool,
) -> None:
    if checkpoint_path in {None, ""}:
        return
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"model.checkpoint_path does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    state_dict = _strip_state_prefixes(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if strict:
        return
    if len(state_dict) == 0 or (missing and len(missing) == len(model.state_dict())):
        raise RuntimeError(f"checkpoint {path} did not match any model parameters")


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict", "net", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    raise RuntimeError("checkpoint must contain a model/state_dict mapping")


def _strip_state_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "model.")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    changed = True
        cleaned[normalized] = value
    return cleaned

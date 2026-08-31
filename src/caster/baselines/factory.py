"""Baseline methods for the shared evaluation loop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn

from caster.models.factory import FeatureModel


@dataclass
class BaselineOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    predictions: torch.Tensor
    abstain_mask: torch.Tensor
    adapted: bool
    reason: str


class Baseline(Protocol):
    name: str

    def predict_batch(self, images: torch.Tensor, feature_model: FeatureModel) -> BaselineOutput:
        ...


class SourceOnlyBaseline:
    name = "source-only"

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor, feature_model: FeatureModel) -> BaselineOutput:
        logits, _ = feature_model.logits_and_features(images)
        return _baseline_output(logits, adapted=False, reason="source_only")


class TemperatureScalingBaseline:
    name = "temperature"

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor, feature_model: FeatureModel) -> BaselineOutput:
        logits, _ = feature_model.logits_and_features(images)
        logits = logits / max(self.temperature, 1e-6)
        return _baseline_output(logits, adapted=False, reason="temperature_scaling")


class TentBaseline:
    """Minimal Tent implementation over BatchNorm affine parameters."""

    name = "tent"

    def __init__(self, lr: float = 1e-3, steps: int = 1) -> None:
        self.lr = float(lr)
        self.steps = int(steps)
        self._optimizer: torch.optim.Optimizer | None = None

    def _configure(self, model: nn.Module) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        model.train()
        for param in model.parameters():
            param.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                    params.append(module.weight)
                if module.bias is not None:
                    module.bias.requires_grad_(True)
                    params.append(module.bias)
        return params

    def predict_batch(self, images: torch.Tensor, feature_model: FeatureModel) -> BaselineOutput:
        if self._optimizer is None:
            params = self._configure(feature_model.model)
            if not params:
                # _configure switched the model to train mode before discovering
                # there is nothing to adapt. Leaving it there keeps dropout and
                # stochastic depth active at test time, which silently degrades
                # the baseline: on Swin-T that read as 85.65 against
                # source-only's 87.19, an apparent adaptation effect that was
                # only dropout. With no BatchNorm, Tent must be exactly
                # source-only.
                feature_model.model.eval()
                with torch.no_grad():
                    logits, _ = feature_model.logits_and_features(images)
                return _baseline_output(logits, adapted=False, reason="no_batchnorm_params")
            self._optimizer = torch.optim.Adam(params, lr=self.lr)

        logits = None
        for _ in range(self.steps):
            logits, _ = feature_model.logits_and_features(images)
            entropy = -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
            loss = entropy.mean()
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()
        assert logits is not None
        return _baseline_output(logits.detach(), adapted=True, reason="tent_entropy_step")


class EataBaseline(TentBaseline):
    """Efficient anti-forgetting test-time adaptation (Niu et al., 2022).

    Three mechanisms distinguish this from Tent, all of which must be present or
    the method degenerates back into Tent:

    1. **Reliability filter** -- drop high-entropy samples, keeping those below
       ``E_0 = entropy_margin * ln(C)``. The paper uses 0.4; at 0.9 the threshold
       admits nearly every sample and the filter becomes a no-op, which is what
       the previous implementation did.
    2. **Redundancy filter** -- drop samples whose prediction is nearly parallel
       to a running average of already-used predictions, so repeated, similar
       gradients do not dominate the update.
    3. **Entropy reweighting** -- weight each surviving sample by
       ``exp(E_0 - entropy)`` so confident samples contribute more.

    The Fisher anti-forgetting regulariser of the full method needs labelled
    source data at adaptation time; omitting it yields ETA, the paper's own
    ablation, which is what is implemented here. Reported as such.
    """

    name = "eata"

    def __init__(
        self,
        lr: float = 1e-3,
        steps: int = 1,
        entropy_margin: float = 0.4,
        d_margin: float = 0.05,
    ) -> None:
        super().__init__(lr=lr, steps=steps)
        self.entropy_margin = float(entropy_margin)
        self.d_margin = float(d_margin)
        self._probe_average: torch.Tensor | None = None

    def _select(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mask, weights, entropy) for the reliable, non-redundant samples."""
        probs = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        threshold = self.entropy_margin * math.log(logits.shape[1])

        mask = entropy < threshold
        if self._probe_average is not None and mask.any():
            similarity = F.cosine_similarity(
                self._probe_average.unsqueeze(0), probs, dim=1
            ).abs()
            mask = mask & (similarity < 1.0 - self.d_margin)

        if mask.any():
            selected = probs[mask].mean(dim=0).detach()
            self._probe_average = (
                selected
                if self._probe_average is None
                else 0.9 * self._probe_average + 0.1 * selected
            )

        weights = torch.exp(threshold - entropy.detach()).clamp(max=1e4)
        return mask, weights, entropy

    def predict_batch(self, images: torch.Tensor, feature_model: FeatureModel) -> BaselineOutput:
        if self._optimizer is None:
            params = self._configure(feature_model.model)
            if not params:
                feature_model.model.eval()   # see TentBaseline: do not leave train mode on
                with torch.no_grad():
                    logits, _ = feature_model.logits_and_features(images)
                return _baseline_output(logits, adapted=False, reason="no_batchnorm_params")
            self._optimizer = torch.optim.Adam(params, lr=self.lr)

        logits = None
        kept = 0
        for _ in range(self.steps):
            logits, _ = feature_model.logits_and_features(images)
            mask, weights, entropy = self._select(logits)
            kept = int(mask.sum().item())
            if kept == 0:
                # No reliable sample: skip the update rather than train on noise.
                continue
            loss = (weights[mask] * entropy[mask]).mean()
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._optimizer.step()
        assert logits is not None
        return _baseline_output(
            logits.detach(),
            adapted=kept > 0,
            reason=f"eata_step:kept={kept}/{logits.shape[0]}",
        )


class T3ABaseline:
    """Test-time template adjustment (Iwasawa & Matsuo, 2021).

    Keeps a support set of confident target features per class, initialised from
    the classifier head weights, and classifies by similarity to the resulting
    templates. No backward pass and no parameter mutation: the model is never
    modified, only the templates built on top of it.
    """

    name = "t3a"

    def __init__(self, filter_k: int = 20) -> None:
        self.filter_k = int(filter_k)
        self._entropies: list[list[float]] = []
        self._features: list[list[torch.Tensor]] = []

    def _head_weight(self, model: nn.Module) -> torch.Tensor | None:
        for attr in ("head", "fc", "classifier"):
            module = getattr(model, attr, None)
            if isinstance(module, nn.Linear):
                return module.weight.detach()
            if isinstance(module, nn.Sequential):
                for layer in reversed(module):
                    if isinstance(layer, nn.Linear):
                        return layer.weight.detach()
        return None

    def _templates(self) -> torch.Tensor:
        """Per-class support sums, L2-normalised.

        Normalisation is load-bearing, not cosmetic: a support set mixes the
        classifier weight row with penultimate features whose norms are ~50x
        larger, so an unnormalised mean is dominated by whichever features
        happened to be added and collapses to chance accuracy.
        """
        sums = torch.stack([torch.stack(f, dim=0).sum(dim=0) for f in self._features], dim=0)
        return torch.nn.functional.normalize(sums, dim=1)

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor, feature_model: FeatureModel) -> BaselineOutput:
        logits, features = feature_model.logits_and_features(images)

        if not self._features:
            weight = self._head_weight(feature_model.model)
            if weight is None or weight.shape[1] != features.shape[1]:
                # Template space must match the feature space; otherwise T3A is
                # not applicable to this backbone and we say so rather than
                # silently degrading to source-only.
                return _baseline_output(logits, adapted=False, reason="t3a_unsupported_head")
            num_classes = weight.shape[0]
            self._features = [[weight[c].detach().clone()] for c in range(num_classes)]
            # The initial weight row is never evicted by the entropy filter.
            self._entropies = [[float("-inf")] for _ in range(num_classes)]

        # Pseudo-labels come from the current templates, as in the original
        # method, not from the frozen head.
        scores = features @ self._templates().T
        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs * torch.log_softmax(scores, dim=-1)).sum(dim=-1)
        pseudo = scores.argmax(dim=-1)

        for i in range(features.shape[0]):
            cls = int(pseudo[i].item())
            self._features[cls].append(features[i].detach())
            self._entropies[cls].append(float(entropy[i].item()))
            if len(self._features[cls]) > self.filter_k + 1:
                order = sorted(range(len(self._entropies[cls])), key=lambda j: self._entropies[cls][j])
                keep = sorted(order[: self.filter_k + 1])
                self._features[cls] = [self._features[cls][j] for j in keep]
                self._entropies[cls] = [self._entropies[cls][j] for j in keep]

        adjusted = features @ self._templates().T
        return _baseline_output(adjusted, adapted=True, reason="t3a_template_update")


class _SAM(torch.optim.Optimizer):
    """Sharpness-aware minimisation wrapper (Foret et al., 2021).

    Two-step update: ascend to the worst-case point within an epsilon ball, take
    the gradient there, then step from the original weights. SAR uses this so
    that test-time updates land in flat regions of the entropy surface, which is
    what keeps it stable where plain entropy minimisation collapses.
    """

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs) -> None:
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        super().__init__(params, dict(rho=rho, **kwargs))
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group.get("rho", 0.05) / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                e_w = self.state.get(p, {}).get("e_w")
                if e_w is not None:
                    p.sub_(e_w)          # back to the original weights
                    self.state[p]["e_w"] = None
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102 - SAM is driven by first/second_step
        raise RuntimeError("call first_step() and second_step() instead")


class SarBaseline:
    """Sharpness-aware and reliable entropy minimisation (Niu et al., 2023).

    Included because Tent and EATA adapt *BatchNorm* affine parameters, and four
    of the six backbones evaluated here (ConvNeXt, ViT, DeiT, Swin) contain no
    BatchNorm at all -- on those they silently degrade to source-only, leaving
    the transformer columns without a gradient-based competitor. SAR adapts
    normalisation-layer affine parameters generally, so it runs on LayerNorm
    architectures.

    Three components, all load-bearing:

    1. **Reliable filtering** -- update only on samples below ``E_0 = margin *
       ln(C)``, discarding high-entropy samples whose gradients are unreliable.
    2. **Sharpness-aware update** -- a two-step SAM update, so the step is taken
       at the worst case within an epsilon ball rather than at the current point.
    3. **Model recovery** -- if the moving average of the post-perturbation loss
       collapses below ``reset_threshold``, restore the initial parameters.
       Without it, a bad stretch of batches is unrecoverable.

    Costs two forward/backward passes per batch, so it needs roughly twice the
    activation memory of Tent.
    """

    name = "sar"

    def __init__(
        self,
        lr: float = 1e-3,
        steps: int = 1,
        entropy_margin: float = 0.4,
        rho: float = 0.05,
        reset_threshold: float = 0.03,
        ema_decay: float = 0.9,
        collapse_class_fraction: float = 0.95,
    ) -> None:
        self.lr = float(lr)
        self.steps = int(steps)
        self.entropy_margin = float(entropy_margin)
        self.rho = float(rho)
        # Fraction of ln(C), not an absolute nat value. Entropy scales with the
        # class count: 0.2 nats is pathological on ImageNet (ln 1000 = 6.9) but
        # ordinary healthy confidence on CIFAR-10 (ln 10 = 2.3). Using an absolute
        # threshold made this reset on 157/157 batches, undoing every update and
        # making SAR identical to source-only.
        self.reset_threshold = float(reset_threshold)
        self.ema_decay = float(ema_decay)
        # Collapse means the predictions have concentrated on one class. Low
        # entropy alone does not mean collapse -- an accurate model is confident.
        self.collapse_class_fraction = float(collapse_class_fraction)
        self._optimizer: _SAM | None = None
        self._initial: dict[str, torch.Tensor] | None = None
        self._ema: float | None = None
        self._resets = 0

    def _configure(self, model: nn.Module) -> list[nn.Parameter]:
        """Collect affine parameters of every normalisation layer."""
        params: list[nn.Parameter] = []
        model.train()
        for param in model.parameters():
            param.requires_grad_(False)
        norm_types = (
            nn.LayerNorm, nn.GroupNorm,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        )
        for module in model.modules():
            if isinstance(module, norm_types):
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    # SAR deliberately avoids BN's batch statistics, which are
                    # unstable under small or non-i.i.d. test batches.
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
                for attr in ("weight", "bias"):
                    p = getattr(module, attr, None)
                    if p is not None:
                        p.requires_grad_(True)
                        params.append(p)
        return params

    @staticmethod
    def _entropy(logits: torch.Tensor) -> torch.Tensor:
        return -(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum(dim=-1)

    def predict_batch(self, images: torch.Tensor, feature_model: FeatureModel) -> BaselineOutput:
        model = feature_model.model
        if self._optimizer is None:
            params = self._configure(model)
            if not params:
                model.eval()   # see TentBaseline: do not leave train mode on
                with torch.no_grad():
                    logits, _ = feature_model.logits_and_features(images)
                return _baseline_output(logits, adapted=False, reason="no_norm_params")
            self._optimizer = _SAM(params, torch.optim.SGD, rho=self.rho, lr=self.lr, momentum=0.9)
            self._initial = {
                name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad
            }

        logits = None
        applied = 0
        for _ in range(self.steps):
            logits, _ = feature_model.logits_and_features(images)
            threshold = self.entropy_margin * math.log(logits.shape[1])
            entropy = self._entropy(logits)
            keep = entropy < threshold
            if not bool(keep.any()):
                continue

            entropy[keep].mean().backward()
            self._optimizer.first_step(zero_grad=True)

            logits2, _ = feature_model.logits_and_features(images)
            entropy2 = self._entropy(logits2)
            # Re-filter at the perturbed point: samples that became unreliable
            # under perturbation are exactly the ones to exclude.
            keep2 = keep & (entropy2 < threshold)
            if not bool(keep2.any()):
                self._optimizer.second_step(zero_grad=True)
                continue
            loss2 = entropy2[keep2].mean()
            loss2.backward()
            self._optimizer.second_step(zero_grad=True)
            applied += 1

            value = float(loss2.detach().item())
            self._ema = value if self._ema is None else (
                self.ema_decay * self._ema + (1 - self.ema_decay) * value
            )

        # Recovery: a collapsed loss means the model is predicting one class
        # confidently. Restore the starting point rather than continue from it.
        reset = False
        collapsed = False
        if logits is not None:
            pred = logits.detach().argmax(dim=-1)
            counts = torch.bincount(pred, minlength=logits.shape[1]).float()
            collapsed = bool((counts.max() / counts.sum()).item() >= self.collapse_class_fraction)
        threshold = self.reset_threshold * math.log(logits.shape[1]) if logits is not None else 0.0
        if self._ema is not None and self._ema < threshold and collapsed and self._initial:
            with torch.no_grad():
                for name, p in model.named_parameters():
                    saved = self._initial.get(name)
                    if saved is not None:
                        p.copy_(saved)
            self._ema = None
            self._resets += 1
            reset = True

        assert logits is not None
        return _baseline_output(
            logits.detach(),
            adapted=applied > 0,
            reason=f"sar_step:applied={applied}{',reset' if reset else ''}",
        )


class UnsupportedBaseline:
    def __init__(self, name: str) -> None:
        self.name = name

    def predict_batch(self, images: torch.Tensor, feature_model: FeatureModel) -> BaselineOutput:
        with torch.no_grad():
            logits, _ = feature_model.logits_and_features(images)
        return _baseline_output(logits, adapted=False, reason=f"unsupported:{self.name}")


def create_baseline(name: str, **kwargs: float | int) -> Baseline:
    normalized = name.lower().replace("_", "-")
    if normalized in {"source", "source-only", "source_only"}:
        return SourceOnlyBaseline()
    if normalized in {"temperature", "temperature-scaling", "temperature_scaling"}:
        return TemperatureScalingBaseline(temperature=float(kwargs.get("temperature", 1.0)))
    if normalized == "tent":
        return TentBaseline(lr=float(kwargs.get("lr", 1e-3)), steps=int(kwargs.get("steps", 1)))
    if normalized == "eata":
        return EataBaseline(
            lr=float(kwargs.get("lr", 1e-3)),
            steps=int(kwargs.get("steps", 1)),
            entropy_margin=float(kwargs.get("entropy_margin", 0.4)),
        )
    if normalized == "t3a":
        return T3ABaseline(filter_k=int(kwargs.get("filter_k", 20)))
    if normalized == "sar":
        return SarBaseline(
            lr=float(kwargs.get("lr", 1e-3)),
            steps=int(kwargs.get("steps", 1)),
            entropy_margin=float(kwargs.get("entropy_margin", 0.4)),
            rho=float(kwargs.get("rho", 0.05)),
            reset_threshold=float(kwargs.get("reset_threshold", 0.2)),
        )
    return UnsupportedBaseline(normalized)


#: Inner methods the certified gate can wrap, as "<inner>+caster".
GATED_SUFFIX = "+caster"


def split_gated_name(name: str) -> tuple[str, bool]:
    """Split "tent+caster" into ("tent", True); leave plain names untouched."""
    normalized = name.lower().replace("_", "-")
    if normalized.endswith(GATED_SUFFIX):
        return normalized[: -len(GATED_SUFFIX)], True
    return normalized, False


def _baseline_output(logits: torch.Tensor, *, adapted: bool, reason: str) -> BaselineOutput:
    probs = F.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1)
    abstain = torch.zeros(pred.shape, dtype=torch.bool, device=pred.device)
    return BaselineOutput(logits=logits, probabilities=probs, predictions=pred, abstain_mask=abstain, adapted=adapted, reason=reason)

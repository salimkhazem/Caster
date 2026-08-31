"""Tests for the TTA baselines.

The EATA tests exist because the previous implementation produced results
byte-identical to Tent on the full CIFAR-10-C sweep (87.04 / 7.38 / 2.96 for
both). Its reliability filter used ``entropy_margin=0.9``, i.e. a threshold of
0.9*ln(C), which admits almost every sample -- so the filter never fired and the
method was Tent under a different name. Two identical baseline rows in a paper is
the kind of thing a reviewer catches immediately.
"""

from __future__ import annotations

import math

import torch

from caster.baselines.factory import EataBaseline, TentBaseline, create_baseline
from caster.models.factory import FeatureModel, TinyConvNet


def _model(num_classes: int = 10) -> FeatureModel:
    torch.manual_seed(0)
    return FeatureModel(model=TinyConvNet(num_classes=num_classes, feature_dim=16))


def test_eata_reliability_filter_actually_rejects() -> None:
    eata = EataBaseline(entropy_margin=0.4, d_margin=0.05)
    num_classes = 10
    threshold = 0.4 * math.log(num_classes)

    # One near-uniform row (max entropy ~ ln 10 = 2.30) and one confident row.
    uniform = torch.zeros(1, num_classes)
    confident = torch.zeros(1, num_classes)
    confident[0, 3] = 20.0
    logits = torch.cat([uniform, confident], dim=0)

    mask, weights, entropy = eata._select(logits)
    assert entropy[0] > threshold, "uniform row should exceed the reliability threshold"
    assert entropy[1] < threshold
    assert not bool(mask[0]), "high-entropy sample must be filtered out"
    assert bool(mask[1])
    # Confident samples are upweighted relative to borderline ones.
    assert weights[1] > weights[0]


def test_eata_default_margin_is_not_a_no_op() -> None:
    """At the old 0.9 the threshold admits essentially everything."""
    num_classes = 10
    assert EataBaseline().entropy_margin == 0.4
    permissive = 0.9 * math.log(num_classes)
    strict = 0.4 * math.log(num_classes)
    assert strict < permissive
    # Mass spread evenly over 4 of 10 classes: entropy = ln(4) = 1.386, which sits
    # between the two thresholds. The old margin admits it; the new one rejects it.
    probs = torch.full((num_classes,), 1e-12)
    probs[:4] = 0.25
    probs = probs / probs.sum()
    entropy = float(-(probs * probs.log()).sum())
    assert math.isclose(entropy, math.log(4), rel_tol=1e-3)
    assert entropy < permissive, "old margin would have admitted this sample"
    assert entropy > strict, "new margin must reject it"


def test_eata_diverges_from_tent_on_the_same_stream() -> None:
    """Same data, same seed, same lr: the filtering must change the trajectory."""
    images = torch.randn(8, 3, 8, 8)

    tent_model, eata_model = _model(), _model()
    tent = TentBaseline(lr=0.1, steps=1)
    eata = EataBaseline(lr=0.1, steps=1, entropy_margin=0.4, d_margin=0.05)

    for _ in range(4):
        tent.predict_batch(images, tent_model)
        eata.predict_batch(images, eata_model)

    tent_params = torch.cat([p.flatten() for p in tent_model.model.parameters() if p.requires_grad])
    eata_params = torch.cat([p.flatten() for p in eata_model.model.parameters() if p.requires_grad])
    assert not torch.allclose(tent_params, eata_params), (
        "EATA and Tent converged to identical parameters -- the filters are inert"
    )


def test_eata_skips_the_update_when_nothing_is_reliable() -> None:
    eata = EataBaseline(entropy_margin=0.01, d_margin=0.05)  # threshold ~0 -> reject all
    model = _model()
    out = eata.predict_batch(torch.randn(4, 3, 8, 8), model)
    assert not out.adapted
    assert "kept=0/" in out.reason


def test_eata_reports_how_many_samples_survived() -> None:
    out = create_baseline("eata", lr=0.01, steps=1).predict_batch(
        torch.randn(6, 3, 8, 8), _model()
    )
    assert out.reason.startswith("eata_step:kept=")


class _LayerNormOnlyNet(torch.nn.Module):
    """No BatchNorm, plus dropout — so train/eval mode is observable."""

    def __init__(self, num_classes: int = 10, dim: int = 16) -> None:
        super().__init__()
        self.stem = torch.nn.Conv2d(3, dim, 3, padding=1)
        self.norm = torch.nn.LayerNorm(dim)
        self.drop = torch.nn.Dropout(p=0.5)
        self.head = torch.nn.Linear(dim, num_classes)

    def forward_features(self, x):
        return self.drop(self.norm(self.stem(x).mean(dim=(2, 3))))

    def forward(self, x):
        return self.head(self.forward_features(x))


def test_baselines_do_not_leave_the_model_in_train_mode() -> None:
    """With nothing to adapt, a baseline must be exactly source-only.

    _configure switches to train mode before discovering there are no BatchNorm
    parameters. Returning early from there left dropout and stochastic depth
    active at test time: on Swin-T that read as 85.65 against source-only's
    87.19 -- an apparent adaptation effect that was only dropout.
    """
    from caster.baselines.factory import EataBaseline, SarBaseline, TentBaseline

    images = torch.randn(8, 3, 8, 8)
    for cls, reason in ((TentBaseline, "no_batchnorm_params"),
                        (EataBaseline, "no_batchnorm_params")):
        torch.manual_seed(0)
        fm = FeatureModel(model=_LayerNormOnlyNet())
        fm.model.eval()
        with torch.no_grad():
            expected, _ = fm.logits_and_features(images)

        out = cls(lr=0.1).predict_batch(images, fm)
        assert out.reason == reason, (cls.__name__, out.reason)
        assert not fm.model.training, f"{cls.__name__} left the model in train mode"
        assert torch.allclose(out.logits, expected, atol=1e-6), (
            f"{cls.__name__} must reproduce source-only exactly when inert"
        )

    # SAR adapts LayerNorm, so it is not inert here; assert it stays usable.
    torch.manual_seed(0)
    fm = FeatureModel(model=_LayerNormOnlyNet())
    out = SarBaseline(lr=0.1, entropy_margin=10.0).predict_batch(images, fm)
    assert out.reason.startswith("sar_step:")

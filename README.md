# CASTER

Implementation of CASTER **Certified Affine Shift Transport for Reliable Test-Time Adaptation** submitted to WACV 2027.

## Setup

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync --extra dev --extra vision
```

For a quick local smoke test using synthetic `FakeData`:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests/
python scripts/reproduce_debug.sh
```

## Core Commands

```bash
python scripts/check_data.py --name fake --root ./data
python scripts/check_model.py --model tiny_cnn --num-classes 10
python scripts/extract_source_stats.py --config configs/debug/caster_debug.yaml
python scripts/run_tta.py --config configs/debug/caster_debug.yaml --method caster --max-steps 3
```

## Toy Theory Validation

The controlled toy benchmark validates the theorem-facing and method-facing claims before any large benchmark run:

```bash
bash scripts/reproduce_toy.sh
```

It writes raw results to `results/raw/toy_theory/` and vector figures to `results/figures/toy_theory/`:

- `toy_geometry.svg`
- `toy_certificate_accuracy_sweep.svg`
- `toy_baseline_comparison.svg`

The benchmark compares source-only, mean-shift, global feature alignment, T3A-style prototypes, EATA-style filtered alignment, SoTTA-style screened prototypes, CASTER without gate, gated CASTER, no-transport CASTER, and oracle true-affine transport.

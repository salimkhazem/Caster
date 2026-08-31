#!/usr/bin/env bash
set -euo pipefail

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests/
python scripts/check_data.py --name fake --root ./data --max-samples 16
python scripts/check_model.py --model tiny_cnn --num-classes 10 --image-size 32
python scripts/extract_source_stats.py --config configs/debug/caster_debug.yaml
python scripts/run_tta.py --config configs/debug/caster_debug.yaml --method source-only --max-steps 2
python scripts/run_tta.py --config configs/debug/caster_debug.yaml --method caster --max-steps 2
python scripts/make_tables.py --results-root results/raw/debug
python scripts/make_figures.py --results-root results/raw/debug

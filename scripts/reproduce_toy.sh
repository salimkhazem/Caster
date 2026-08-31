#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
fi

PYTHONPATH=src "$PYTHON_BIN" scripts/run_toy_theory.py \
  --seeds 20 \
  --output-dir results/raw/toy_theory \
  --figure-dir results/figures/toy_theory

PYTHONPATH=src "$PYTHON_BIN" scripts/make_toy_publication_figures.py \
  --input results/raw/toy_theory/toy_results.json \
  --output-dir results/figures/toy_theory_publication

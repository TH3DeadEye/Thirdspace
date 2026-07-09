"""Evaluation package: datasets, gold-label generation, and the run harness.

Spend policy: every script in this package is zero-cost by default. Anything
that would call Bedrock requires an explicit --confirm-spend flag and prints
an exact call-count and dollar estimate first. See eval/README.md.
"""
import sys
from pathlib import Path

# Make thirdspace/ importable from any eval script, matching tests/__init__.py.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT, _ROOT / "thirdspace"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EVAL_DIR = Path(__file__).resolve().parent
DATA_DIR = EVAL_DIR / "data"
GOLD_DIR = EVAL_DIR / "gold"
RESULTS_DIR = EVAL_DIR / "results"
LABEL_MAPS_DIR = EVAL_DIR / "label_maps"

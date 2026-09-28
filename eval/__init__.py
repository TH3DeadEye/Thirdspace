"""ThirdSpace evaluation harness.

Measures the pipeline against two labelled datasets:

  normad  — akhilayerukola/NormAd. A short story ending in "Is what X did
            socially acceptable?", grounded in one country's etiquette norms.
            Gold Label in {yes, no, neutral}; the pre-registered gold file
            (eval/gold/normad_frames.jsonl) recodes these to
            adhere/violate/neither and adds the derived challenger_country.
  mdd     — kracr/moral-decision-dataset. A real-world case with an active
            agent, a passive agent, and a curated moral decision.

Both are turned into ThirdSpace runs the same way: the dataset supplies the
two cultural frames (via prompts.read_cultural_frames — no frame is ever
inferred), the pipeline produces mediation output, and a CLASSIFIER maps that
output back into the dataset's own label vocabulary. Accuracy is measured in
that vocabulary.

Three classifier heads, selectable per run:
  structural — deterministic function of the mediation signals. Zero extra
               LLM calls. Thresholds are calibrated, not guessed (calibrate.py).
  readout    — one LLM call that reads ONLY the pipeline's artifacts (The
               Occurring + the final mediation) and picks a label.
  hybrid     — both; readout decides, and structural/readout agreement is
               reported as a separate diagnostic.

A `--baseline` flag adds a single-call prediction on the raw record with no
pipeline at all, so "does ThirdSpace help?" is answerable and not just
"what accuracy did ThirdSpace get?".

Entry point: `python -m eval run --dataset normad --n 20`.

Spend policy: loading, sampling, reporting and calibration are all free. Only
`run` calls Bedrock, and it prints its call count as it goes.
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

__all__ = ["config", "datasets", "signals", "classifiers", "metrics", "runner",
           "EVAL_DIR", "DATA_DIR", "GOLD_DIR", "RESULTS_DIR", "LABEL_MAPS_DIR"]
__version__ = "0.1.0"

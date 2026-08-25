"""Fit the structural classifier's thresholds on cached signals.

The four cut points in StructuralThresholds are the only free parameters in
the deterministic head. Leaving them as hand-picked constants would make
"structural accuracy" a statement about my guesses rather than about the
pipeline, so they are fitted here — on signals already written to
predictions.jsonl, with no LLM calls at all.

K-fold cross-validation, because the honest number is out-of-fold: a grid
search that reports its own training score on 20 records will happily find a
threshold set that scores 0.9 and generalizes to nothing.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

from . import metrics
from .classifiers import classify_structural
from .config import StructuralThresholds
from .datasets import LabelVocabulary

# Grid. Coarse on purpose: four axes at five points each is 625 evaluations
# per fold, which is instant, and finer steps would only overfit harder.
GRID = {
    "low_conflict_density": [0.10, 0.15, 0.20, 0.25, 0.35],
    "high_agreement_ratio": [0.30, 0.40, 0.50, 0.60, 0.70],
    "norm_pressure": [0.30, 0.40, 0.45, 0.55, 0.65],
    "mean_severity": [2.0, 2.5, 3.0, 3.5, 4.0],
}


def _candidates() -> list[StructuralThresholds]:
    out = []
    for a in GRID["low_conflict_density"]:
        for b in GRID["high_agreement_ratio"]:
            for c in GRID["norm_pressure"]:
                for d in GRID["mean_severity"]:
                    out.append(StructuralThresholds(
                        low_conflict_density=a, high_agreement_ratio=b,
                        norm_pressure=c, mean_severity=d))
    return out


def _score(rows: Sequence[dict], vocab: LabelVocabulary,
           thresholds: StructuralThresholds, labels: Sequence[str]) -> float:
    pairs = []
    for row in rows:
        sig = row.get("signals")
        if not sig or row.get("gold") is None:
            continue
        pred = classify_structural(sig, vocab, thresholds)
        pairs.append((row["gold"], pred.label))
    # Macro-F1, not accuracy: on a label-skewed sample accuracy would pick the
    # threshold set that collapses every record onto the majority class.
    return metrics.macro_f1([(g, p) for g, p in pairs if p is not None], labels)


def calibrate(rows: Sequence[dict], vocab: LabelVocabulary, folds: int = 5,
              seed: int = 0) -> dict:
    usable = [r for r in rows if r.get("signals") and r.get("gold") is not None]
    labels = list(vocab.labels)
    if len(usable) < 2 * folds:
        folds = max(2, len(usable) // 2) if len(usable) >= 4 else 1

    rng = random.Random(seed)
    shuffled = list(usable)
    rng.shuffle(shuffled)
    grid = _candidates()

    if folds <= 1:
        best = max(grid, key=lambda t: _score(shuffled, vocab, t, labels))
        return {"thresholds": best.to_dict(), "folds": 1,
                "cv_macro_f1": None, "in_sample_macro_f1": _score(shuffled, vocab, best, labels),
                "n": len(shuffled),
                "warning": "too few records for cross-validation; score is in-sample"}

    chunks = [shuffled[i::folds] for i in range(folds)]
    fold_scores: list[float] = []
    for i in range(folds):
        test = chunks[i]
        train = [r for j, c in enumerate(chunks) if j != i for r in c]
        best_fold = max(grid, key=lambda t: _score(train, vocab, t, labels))
        fold_scores.append(_score(test, vocab, best_fold, labels))

    # Final thresholds are fitted on everything; the CV number above is what
    # should be quoted for them.
    best = max(grid, key=lambda t: _score(shuffled, vocab, t, labels))
    return {
        "thresholds": best.to_dict(),
        "folds": folds,
        "cv_macro_f1": sum(fold_scores) / len(fold_scores),
        "fold_scores": fold_scores,
        "in_sample_macro_f1": _score(shuffled, vocab, best, labels),
        "n": len(shuffled),
        "default_in_sample_macro_f1": _score(shuffled, vocab, StructuralThresholds(), labels),
    }


def load_rows(run_dir: str | Path) -> tuple[list[dict], LabelVocabulary]:
    run_dir = Path(run_dir)
    rows = [json.loads(line) for line in
            (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    meta = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    v = meta["vocabulary"]
    vocab = LabelVocabulary(labels=v["labels"], canon=v["canon"], counts=v["counts"])
    return rows, vocab

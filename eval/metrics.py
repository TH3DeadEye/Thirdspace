"""Classification metrics, no numpy or scikit-learn.

Accuracy alone is close to meaningless on these datasets — NormAd's gold
labels are dominated by `yes`, so a model that never says anything else can
score in the eighties. Every report therefore carries balanced accuracy,
macro-F1 and the full confusion matrix, plus the majority-class baseline so
the reader can see what "doing nothing" scores on the same sample.

Abstentions (prediction None) are counted separately and, by default, excluded
from the scored set rather than counted as errors — a harness that scores an
API failure as a wrong answer is measuring its own network.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Sequence


Pair = tuple[str, str | None]      # (gold, predicted)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def confusion_matrix(pairs: Sequence[Pair], labels: Sequence[str]) -> dict:
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for gold, pred in pairs:
        if pred is None or gold not in matrix or pred not in matrix[gold]:
            continue
        matrix[gold][pred] += 1
    return matrix


def accuracy(pairs: Sequence[Pair]) -> float:
    scored = [(g, p) for g, p in pairs if p is not None]
    if not scored:
        return 0.0
    return sum(1 for g, p in scored if g == p) / len(scored)


def per_class(pairs: Sequence[Pair], labels: Sequence[str]) -> dict:
    out = {}
    for label in labels:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p is not None and p != label)
        support = sum(1 for g, _ in pairs if g == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1,
                      "support": support, "tp": tp, "fp": fp, "fn": fn}
    return out


def macro_f1(pairs: Sequence[Pair], labels: Sequence[str]) -> float:
    stats = per_class(pairs, labels)
    present = [l for l in labels if stats[l]["support"] > 0]
    if not present:
        return 0.0
    return sum(stats[l]["f1"] for l in present) / len(present)


def balanced_accuracy(pairs: Sequence[Pair], labels: Sequence[str]) -> float:
    """Mean per-class recall — the number that does not move when the sample
    is 85% one label."""
    stats = per_class(pairs, labels)
    present = [l for l in labels if stats[l]["support"] > 0]
    if not present:
        return 0.0
    return sum(stats[l]["recall"] for l in present) / len(present)


def cohen_kappa(pairs: Sequence[Pair], labels: Sequence[str]) -> float:
    """Agreement corrected for chance. Near 0 means the predictions carry no
    information beyond the label prior, whatever the raw accuracy says."""
    scored = [(g, p) for g, p in pairs if p is not None]
    n = len(scored)
    if n == 0:
        return 0.0
    observed = sum(1 for g, p in scored if g == p) / n
    gold_dist = Counter(g for g, _ in scored)
    pred_dist = Counter(p for _, p in scored)
    expected = sum((gold_dist[l] / n) * (pred_dist[l] / n) for l in labels)
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1 - expected)


def majority_baseline(pairs: Sequence[Pair]) -> dict:
    """What always predicting the most common gold label would score."""
    golds = [g for g, _ in pairs]
    if not golds:
        return {"label": None, "accuracy": 0.0}
    label, count = Counter(golds).most_common(1)[0]
    return {"label": label, "accuracy": count / len(golds)}


# ---------------------------------------------------------------------------
# Uncertainty and comparison
# ---------------------------------------------------------------------------

def bootstrap_accuracy_ci(pairs: Sequence[Pair], iterations: int = 2000,
                          alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile CI for accuracy. On 20 records the interval is wide; that is
    the point — a harness run on a handful of samples should say so."""
    scored = [(g, p) for g, p in pairs if p is not None]
    if len(scored) < 2:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(scored)
    scores = []
    for _ in range(iterations):
        sample = [scored[rng.randrange(n)] for _ in range(n)]
        scores.append(sum(1 for g, p in sample if g == p) / n)
    scores.sort()
    lo = scores[int((alpha / 2) * iterations)]
    hi = scores[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (lo, hi)


def _binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial p for McNemar's discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar(pairs_a: Sequence[Pair], pairs_b: Sequence[Pair]) -> dict:
    """Is system A better than system B on the SAME records?

    Paired, so it answers the question two independent accuracy numbers cannot:
    on how many records did exactly one of them get it right. Only records
    where both produced a label are counted.
    """
    b = c = both = neither = 0
    for (gold_a, pred_a), (gold_b, pred_b) in zip(pairs_a, pairs_b):
        if pred_a is None or pred_b is None:
            continue
        ok_a, ok_b = pred_a == gold_a, pred_b == gold_b
        if ok_a and ok_b:
            both += 1
        elif ok_a and not ok_b:
            b += 1
        elif ok_b and not ok_a:
            c += 1
        else:
            neither += 1
    return {"a_only": b, "b_only": c, "both": both, "neither": neither,
            "p_value": _binom_two_sided(b, c), "n_compared": both + neither + b + c}


def agreement_rate(preds_a: Sequence[str | None], preds_b: Sequence[str | None]) -> float:
    both = [(a, b) for a, b in zip(preds_a, preds_b) if a is not None and b is not None]
    if not both:
        return 0.0
    return sum(1 for a, b in both if a == b) / len(both)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def evaluate(pairs: Sequence[Pair], labels: Sequence[str], seed: int = 0) -> dict:
    """The full metric block for one system's predictions."""
    scored = [(g, p) for g, p in pairs if p is not None]
    lo, hi = bootstrap_accuracy_ci(pairs, seed=seed)
    return {
        "n": len(pairs),
        "n_scored": len(scored),
        "n_abstained": len(pairs) - len(scored),
        "accuracy": accuracy(pairs),
        "accuracy_ci95": [lo, hi],
        "balanced_accuracy": balanced_accuracy(scored, labels),
        "macro_f1": macro_f1(scored, labels),
        "cohen_kappa": cohen_kappa(pairs, labels),
        "per_class": per_class(scored, labels),
        "confusion": confusion_matrix(pairs, labels),
        "prediction_distribution": dict(Counter(p for _, p in scored)),
        "majority_baseline": majority_baseline(pairs),
    }


def by_slice(rows: Sequence[dict], labels: Sequence[str], key: str,
             gold_key: str = "gold", pred_key: str = "prediction",
             min_support: int = 3) -> dict:
    """Accuracy per country / per domain. Slices below min_support are dropped
    rather than reported as 0% or 100% off two records."""
    buckets: dict[str, list[Pair]] = {}
    for row in rows:
        buckets.setdefault(row.get(key) or "(none)", []).append(
            (row.get(gold_key), row.get(pred_key)))
    out = {}
    for name, pairs in sorted(buckets.items()):
        if len(pairs) < min_support:
            continue
        out[name] = {"n": len(pairs), "accuracy": accuracy(pairs),
                     "macro_f1": macro_f1([(g, p) for g, p in pairs if p], labels)}
    return out

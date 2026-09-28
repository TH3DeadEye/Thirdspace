"""Mapping ThirdSpace output onto a dataset label.

structural — a deterministic decision rule over signals.py. No LLM call, so
             it is free to re-run and can be calibrated offline against cached
             signals. It answers in canonical buckets and is translated into
             the dataset's vocabulary at the end.
readout    — one LLM call over the pipeline's artifacts only.
hybrid     — runs both; readout is the reported prediction, and the rate at
             which the two heads agree is itself a result (a structural rule
             that tracks the read-out is a rule you can ship without paying
             for a judge call per record).
baseline   — one LLM call over the raw record, no pipeline. The control.

Every head can ABSTAIN (label=None). Abstentions are counted and reported
separately rather than silently scored as wrong.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

from .datasets import CANON_ACCEPT, CANON_REJECT, CANON_UNSETTLED, canonicalize

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    label: str | None            # in the dataset's vocabulary, or None = abstain
    source: str                  # structural | readout | baseline
    canon: str | None = None
    confidence: float | None = None
    rationale: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Vocabulary mapping
# ---------------------------------------------------------------------------

def _to_vocab(canon: str, vocab) -> str | None:
    """Canonical bucket -> the dataset's own word for it.

    A dataset with no "unsettled" label (a strictly binary MDD column, say)
    gets the unsettled bucket folded into the majority label rather than an
    abstention, because "the analysis did not settle it" is not an allowed
    answer there and abstaining would silently drop the record.
    """
    direct = vocab.for_canon(canon)
    if direct:
        return direct
    if canon == CANON_UNSETTLED:
        return vocab.labels[0] if vocab.labels else None
    return None


def _parse_label(raw: str | None, vocab) -> str | None:
    """Accept the model's answer only if it lands in the allowed vocabulary."""
    if not raw:
        return None
    text = str(raw).strip().strip('"').lower()
    for label in vocab.labels:
        if text == label.lower():
            return label
    # Model answered in a synonym ("acceptable" where the dataset says "yes").
    canon = canonicalize(text)
    if canon:
        mapped = vocab.for_canon(canon)
        if mapped:
            return mapped
    # Last resort: the label appears inside a longer sentence.
    for label in vocab.labels:
        if label.lower() in text:
            return label
    return None


# ---------------------------------------------------------------------------
# Structural head
# ---------------------------------------------------------------------------

def structural_canon(signals: dict, thresholds) -> tuple[str, str]:
    """Canonical bucket + a one-line reason, from the signal bundle alone.

    The rule, in words: two independently-derived readings that largely agree
    describe a situation whose norms are not in dispute; a small number of
    load-bearing conflicts about value and assumption describe one where the
    behaviour is being judged and found wanting; anything else is the pipeline
    reporting that it did not settle the question.
    """
    density = float(signals.get("conflict_density", 0.0))
    agreement = float(signals.get("agreement_ratio", 0.0))
    pressure = float(signals.get("norm_pressure", 0.0))
    severity = float(signals.get("mean_severity", 0.0))

    if signals.get("n_conflicts", 0) == 0:
        return CANON_ACCEPT, "no conflicts: both readings converged on the same account"

    if density <= thresholds.low_conflict_density and agreement >= thresholds.high_agreement_ratio:
        return CANON_ACCEPT, (
            f"low conflict density ({density:.2f}) with high agreement ({agreement:.2f})")

    if pressure >= thresholds.norm_pressure and severity >= thresholds.mean_severity:
        return CANON_REJECT, (
            f"normative pressure {pressure:.2f} at mean severity {severity:.2f}")

    return CANON_UNSETTLED, (
        f"density {density:.2f}, agreement {agreement:.2f}, pressure {pressure:.2f}, "
        f"severity {severity:.2f} — no rule fired")


def classify_structural(signals: dict, vocab, thresholds) -> Prediction:
    canon, reason = structural_canon(signals, thresholds)
    return Prediction(label=_to_vocab(canon, vocab), source="structural",
                      canon=canon, rationale=reason)


# ---------------------------------------------------------------------------
# LLM heads
# ---------------------------------------------------------------------------

def classify_readout(record, digest: dict, signals: dict, vocab, *,
                     call_llm_json, model: str | None = None) -> Prediction:
    """One call over the pipeline's own artifacts."""
    from . import prompts as eval_prompts

    kwargs = {"max_tokens": 800, "temperature": 0.0}
    if model:
        kwargs["model"] = model
    try:
        result = call_llm_json(
            eval_prompts.READOUT_SYSTEM,
            eval_prompts.readout_user(record, digest, signals, vocab),
            **kwargs,
        )
    except Exception as exc:                       # noqa: BLE001
        logger.warning("[%s] read-out call failed: %s", record.record_id, exc)
        return Prediction(label=None, source="readout", error=f"{type(exc).__name__}: {exc}")

    if not isinstance(result, dict):
        return Prediction(label=None, source="readout",
                          error=f"read-out returned {type(result).__name__}, expected object")

    label = _parse_label(result.get("label"), vocab)
    confidence = result.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    return Prediction(
        label=label,
        source="readout",
        canon=canonicalize(label) if label else None,
        confidence=confidence,
        rationale=str(result.get("rationale", ""))[:400],
        error=None if label else f"unparsed label {result.get('label')!r}",
    )


def classify_baseline(record, vocab, *, call_llm_json, model: str | None = None) -> Prediction:
    """Control condition: the raw record, one call, no pipeline."""
    from . import prompts as eval_prompts

    kwargs = {"max_tokens": 400, "temperature": 0.0}
    if model:
        kwargs["model"] = model
    try:
        result = call_llm_json(
            eval_prompts.BASELINE_SYSTEM,
            eval_prompts.baseline_user(record, vocab),
            **kwargs,
        )
    except Exception as exc:                       # noqa: BLE001
        logger.warning("[%s] baseline call failed: %s", record.record_id, exc)
        return Prediction(label=None, source="baseline", error=f"{type(exc).__name__}: {exc}")

    if not isinstance(result, dict):
        return Prediction(label=None, source="baseline",
                          error=f"baseline returned {type(result).__name__}")
    label = _parse_label(result.get("label"), vocab)
    return Prediction(label=label, source="baseline",
                      canon=canonicalize(label) if label else None,
                      rationale=str(result.get("rationale", ""))[:300],
                      error=None if label else f"unparsed label {result.get('label')!r}")


def choose_final(structural: Prediction | None, readout: Prediction | None,
                 mode: str) -> Prediction:
    """Which head's answer is the run's prediction.

    In hybrid mode the read-out decides, because it can weigh what a conflict
    was ABOUT and the structural rule can only weigh how many there were. The
    structural answer is kept alongside so their agreement can be measured.
    """
    if mode == "structural":
        return structural or Prediction(label=None, source="structural", error="not run")
    if mode == "readout":
        return readout or Prediction(label=None, source="readout", error="not run")
    if readout is not None and readout.label is not None:
        return readout
    # Read-out failed or answered outside the vocabulary — fall back rather
    # than throw the record away, and record that the fallback happened.
    if structural is not None:
        fallback = Prediction(**{**structural.to_dict(), "source": "structural(fallback)"})
        return fallback
    return Prediction(label=None, source="hybrid", error="both heads unavailable")

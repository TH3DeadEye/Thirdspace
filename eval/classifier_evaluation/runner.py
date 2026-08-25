"""Run the pipeline over a sample and write predictions.

Cost discipline is the design constraint here. One scenario at max_cycles=1 is
roughly 60+ Bedrock calls (30 main cells + 30 challenger cells + reformulation
+ mediation), so a 20-record run is ~1,300 calls before any classifier. Hence:

  - every record is written to predictions.jsonl the moment it finishes, so a
    throttling failure at record 17 does not discard the first 16;
  - --resume skips record ids already present in that file;
  - signals are cached in full, so thresholds can be re-fitted and reports
    re-rendered with zero further calls;
  - --workers defaults to 1, because the cell agents already fan out 8 ways
    inside a single scenario and per-minute quota is the binding constraint.

The graph is invoked directly rather than through main.run_scenario so the
harness can set convergence_metric and llm_call_budget per run. Everything
else mirrors run_scenario exactly, including registering the SessionMemory for
budget/telemetry before invoking.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import classifiers, signals as signals_mod
from ._bootstrap import import_pipeline
from .config import EvalConfig
from .datasets import EvalRecord, LabelVocabulary, describe, load_records, sample_records

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# One record
# ---------------------------------------------------------------------------

def _scenario_id(record: EvalRecord, cfg: EvalConfig) -> str:
    """Stable, filesystem-safe, and sensitive to the frames.

    Mirrors main._make_scenario_id's intent (same inputs -> same id) while
    keeping the dataset row visible in the name, so a session file can be
    traced back to the record that produced it.
    """
    payload = f"{record.scenario}|{record.main_frame}|{record.challenger_frame}|{cfg.max_cycles}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"{record.slug()}-{digest}"


def _build_memory(pipe, record: EvalRecord, cfg: EvalConfig):
    models = pipe.models
    scenario_id = _scenario_id(record, cfg)
    fields: dict[str, Any] = {
        "scenario_id": scenario_id,
        # The frames come from the dataset row, via the project's own
        # read_cultural_frames. "" is a valid, meaningful value: neutral mode.
        "main_cultural_frame": record.main_frame,
        "challenger_cultural_frame": record.challenger_frame,
        "occurring": models.TheOccurring(
            scenario_id=scenario_id, raw_scenario=record.scenario, nodes=[]),
        "max_cycles": cfg.max_cycles,
        "skip_generation": cfg.skip_generation,
        "convergence_metric": cfg.convergence_metric,
        "llm_call_budget": cfg.llm_call_budget,
    }
    if cfg.mediator_frame:
        fields["mediator_frame"] = cfg.mediator_frame
    return models.SessionMemory(**fields)


def run_pipeline(pipe, graph, record: EvalRecord, cfg: EvalConfig):
    """Invoke the compiled graph for one record; returns the final memory."""
    memory = _build_memory(pipe, record, cfg)
    token = pipe.budget.set_active_memory(memory)
    try:
        result = graph.invoke(
            {"memory": memory},
            config={
                "run_name": f"eval[{cfg.dataset}:{record.record_id}]",
                "metadata": {
                    "eval_run": cfg.run_id,
                    "dataset": cfg.dataset,
                    "record_id": record.record_id,
                    "gold": record.gold,
                    "main_cultural_frame": record.main_frame,
                    "challenger_cultural_frame": record.challenger_frame,
                },
                "tags": ["thirdspace", "eval", cfg.dataset],
            },
        )
    finally:
        pipe.budget.reset_active_memory(token)
    return result["memory"]


def evaluate_record(pipe, graph, record: EvalRecord, cfg: EvalConfig,
                    vocab: LabelVocabulary) -> dict:
    """Run + classify one record. Never raises: failures become rows."""
    started = time.monotonic()
    row: dict[str, Any] = {
        "record_id": record.record_id,
        "dataset": cfg.dataset,
        "gold": record.gold,
        "group": record.group,
        "main_frame": record.main_frame,
        "challenger_frame": record.challenger_frame,
        "scenario_chars": len(record.scenario),
        "error": None,
    }

    try:
        memory = run_pipeline(pipe, graph, record, cfg)
    except Exception as exc:                        # noqa: BLE001
        logger.exception("[%s] pipeline failed", record.record_id)
        row.update(error=f"{type(exc).__name__}: {exc}",
                   prediction=None, prediction_source="pipeline_error",
                   wall_seconds=round(time.monotonic() - started, 1))
        return row

    sig = signals_mod.extract_signals(memory)
    digest = signals_mod.mediation_digest(memory)
    row["signals"] = sig
    row["scenario_id"] = sig.get("scenario_id")

    structural = classifiers.classify_structural(sig, vocab, cfg.thresholds)
    readout = None
    if cfg.classifier in ("readout", "hybrid"):
        # Register the same memory so the read-out call's tokens land on this
        # scenario's telemetry instead of vanishing into an unregistered context.
        with pipe.budget.active_memory(memory):
            readout = classifiers.classify_readout(
                record, digest, sig, vocab,
                call_llm_json=pipe.llm.call_llm_json,
                model=cfg.readout_model or pipe.llm.JUDGE_MODEL_ID,
            )

    final = classifiers.choose_final(structural, readout, cfg.classifier)

    baseline = None
    if cfg.baseline:
        with pipe.budget.active_memory(memory):
            baseline = classifiers.classify_baseline(
                record, vocab, call_llm_json=pipe.llm.call_llm_json,
                model=cfg.readout_model or pipe.llm.JUDGE_MODEL_ID)

    # Re-read telemetry after the classifier calls so cost is the full cost.
    row.update({
        "prediction": final.label,
        "prediction_source": final.source,
        "prediction_confidence": final.confidence,
        "prediction_rationale": final.rationale,
        "structural": structural.to_dict(),
        "readout": readout.to_dict() if readout else None,
        "baseline": baseline.to_dict() if baseline else None,
        "correct": (final.label == record.gold) if final.label is not None else None,
        "llm_calls": int(getattr(memory, "llm_calls_made", 0)),
        "total_tokens": int(getattr(memory, "total_tokens", 0)),
        "latency_ms": int(getattr(memory, "total_latency_ms", 0)),
        "wall_seconds": round(time.monotonic() - started, 1),
    })
    if cfg.persist_sessions:
        row["session_path"] = _persist_session(memory, cfg)
    return row


def _persist_session(memory, cfg: EvalConfig) -> str:
    directory = cfg.run_dir() / "sessions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{memory.scenario_id}.json"
    path.write_text(memory.model_dump_json(indent=2), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def _load_done(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed line in %s", path)
    return rows, {str(r.get("record_id")) for r in rows}


def run(cfg: EvalConfig) -> dict:
    """Execute a full evaluation run and return the rows plus the vocabulary."""
    records = load_records(cfg)
    if not records:
        raise ValueError(f"No usable records loaded for dataset {cfg.dataset!r}.")
    sample = sample_records(records, cfg.n, cfg.seed, cfg.stratify)
    vocab = LabelVocabulary.from_records(sample)
    if vocab.looks_free_text():
        raise ValueError(
            f"The label column produced {len(vocab.labels)} distinct values, which "
            f"looks like free text rather than a class: {vocab.labels[:8]}. "
            f"Pick a different column with --mdd-label-field, or run "
            f"`python -m eval inspect` to see what is available."
        )

    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"

    done_rows, done_ids = ([], set())
    if cfg.resume:
        done_rows, done_ids = _load_done(predictions_path)
        if done_ids:
            logger.info("Resuming: %d record(s) already done", len(done_ids))
    todo = [r for r in sample if str(r.record_id) not in done_ids]

    (run_dir / "config.json").write_text(
        json.dumps({"config": cfg.to_dict(),
                    "sample": describe(sample),
                    "vocabulary": vocab.to_dict()}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    pipe = import_pipeline(cfg.src)
    graph = pipe.graph.build_graph()

    rows: list[dict] = list(done_rows)
    lock_path = predictions_path

    def _work(record: EvalRecord) -> dict:
        return evaluate_record(pipe, graph, record, cfg, vocab)

    with lock_path.open("a", encoding="utf-8") as sink:
        if cfg.workers > 1:
            with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                for i, row in enumerate(pool.map(_work, todo), 1):
                    rows.append(row)
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    _log_progress(i, len(todo), row)
        else:
            for i, record in enumerate(todo, 1):
                row = _work(record)
                rows.append(row)
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                _log_progress(i, len(todo), row)

    return {"rows": rows, "vocab": vocab, "sample": sample, "run_dir": run_dir}


def _log_progress(i: int, total: int, row: dict) -> None:
    mark = "ok " if row.get("correct") else ("ERR" if row.get("error") else "MISS")
    logger.info(
        "[%d/%d] %s record=%s gold=%s pred=%s calls=%s tokens=%s %.0fs",
        i, total, mark, row.get("record_id"), row.get("gold"),
        row.get("prediction"), row.get("llm_calls"), row.get("total_tokens"),
        row.get("wall_seconds") or 0,
    )


def rows_to_config_dict(cfg: EvalConfig) -> dict:
    return asdict(cfg)

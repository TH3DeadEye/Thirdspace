"""Loading, normalizing and sampling the two evaluation datasets.

Both datasets arrive as tables with inconsistent header casing, so every row
is normalized to snake_case keys FIRST. That matters for more than tidiness:
utils.prompts.read_cultural_frames looks for "country", "active_agent" etc.
in lower case, and NormAd ships "Country". Normalizing lets the pipeline's own
frame-reading rules be reused verbatim instead of reimplemented here — the
frames really are read from the data, by the project's function.

Leakage: the gold label and any field that states the answer (NormAd's
`explanation`, MDD's decision column) are never part of the scenario text
handed to the pipeline. `assert_no_leakage` enforces that.
"""
from __future__ import annotations

import csv
import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

NORMAD_HF_ID = "akhilayerukola/NormAd"
MDD_DEFAULT_FILE = "MDD_100.csv"   # the sampler shipped in the MDD repo


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

@dataclass
class EvalRecord:
    """One dataset row, adapted to what a ThirdSpace run needs."""
    record_id: str
    dataset: str
    scenario: str                 # the raw text handed to reformulation
    question: str                 # the decision the classifier must answer
    gold: str                     # gold label, lower-cased, in the dataset's vocabulary
    main_frame: str = ""          # "" = neutral mode (prompts.py drops the block)
    challenger_frame: str = ""
    group: str = ""               # country / domain — for per-slice accuracy
    raw: dict = field(default_factory=dict)

    def slug(self) -> str:
        s = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{self.dataset}-{self.record_id}")
        return s.strip("-")[:80]


# ---------------------------------------------------------------------------
# Header / value normalization
# ---------------------------------------------------------------------------

def snake(key: str) -> str:
    k = re.sub(r"[^0-9A-Za-z]+", "_", str(key)).strip("_").lower()
    return re.sub(r"_+", "_", k)


def normalize_row(row: dict) -> dict:
    return {snake(k): ("" if v is None else v) for k, v in row.items()}


def first_present(row: dict, names: Sequence[str]) -> str:
    """First non-empty value among `names`. "" when none are present."""
    for name in names:
        v = row.get(name)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if v not in (None, "") and not isinstance(v, str):
            return str(v).strip()
    return ""


def resolve_field(rows: Sequence[dict], aliases: Sequence[str]) -> str | None:
    """Which of `aliases` actually exists (and is populated) in these rows."""
    for alias in aliases:
        if any(str(r.get(alias, "")).strip() for r in rows[:50]):
            return alias
    return None


# ---------------------------------------------------------------------------
# Label vocabulary
#
# Predictions are made in the DATASET'S OWN vocabulary, discovered from the
# sample, rather than in an invented canonical scheme. The canonical buckets
# below exist only so the deterministic classifier — which reasons about
# conformity, not about the word "yes" — can be mapped onto whatever the
# dataset happens to call things.
# ---------------------------------------------------------------------------

CANON_ACCEPT = "acceptable"
CANON_REJECT = "unacceptable"
CANON_UNSETTLED = "contested"

_ALIASES: dict[str, str] = {}
for _word in ("yes", "moral", "ethical", "acceptable", "permissible", "right",
              "justified", "good", "approve", "approved", "true", "1"):
    _ALIASES[_word] = CANON_ACCEPT
for _word in ("no", "immoral", "unethical", "unacceptable", "impermissible",
              "wrong", "unjustified", "bad", "disapprove", "false", "0"):
    _ALIASES[_word] = CANON_REJECT
for _word in ("neutral", "grey", "gray", "grey area", "gray area", "ambiguous",
              "context dependent", "context-dependent", "depends", "unclear",
              "mixed", "undecided", "indeterminate", "2"):
    _ALIASES[_word] = CANON_UNSETTLED


def canonicalize(label: str) -> str | None:
    """Map a dataset label onto {acceptable, unacceptable, contested}."""
    if not label:
        return None
    key = re.sub(r"[^a-z0-9 -]+", "", str(label).strip().lower())
    if key in _ALIASES:
        return _ALIASES[key]
    # Longer free-text verdicts: fall back to substring evidence, most
    # specific first so "unacceptable" is not read as "acceptable".
    for word, canon in sorted(_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(word)}\b", key):
            return canon
    return None


@dataclass
class LabelVocabulary:
    """The label set this run predicts into, discovered from the sample."""
    labels: list[str]
    canon: dict[str, str | None]           # label -> canonical bucket
    counts: dict[str, int]

    @classmethod
    def from_records(cls, records: Iterable[EvalRecord]) -> "LabelVocabulary":
        counts = Counter(r.gold for r in records if r.gold)
        labels = [lab for lab, _ in counts.most_common()]
        return cls(labels=labels,
                   canon={lab: canonicalize(lab) for lab in labels},
                   counts=dict(counts))

    def for_canon(self, bucket: str) -> str | None:
        """The dataset's word for a canonical bucket, if it has one."""
        for lab in self.labels:
            if self.canon.get(lab) == bucket:
                return lab
        return None

    def looks_free_text(self) -> bool:
        """More than a handful of distinct labels means the chosen column is
        prose, not a class. The caller should stop and pick another column."""
        return len(self.labels) > 6 or any(len(l) > 40 for l in self.labels)

    def to_dict(self) -> dict:
        return {"labels": self.labels, "canon": self.canon, "counts": self.counts}


# ---------------------------------------------------------------------------
# Table reading
# ---------------------------------------------------------------------------

def _read_table(path: str | Path, src: str | None = None) -> list[dict]:
    # --path is resolved against the working directory first and the repo root
    # second, so `--path data/MDD_100.csv` works from either.
    from ._bootstrap import resolve_path

    p = resolve_path(path, src) or Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {p}\n"
            f"Tried the working directory and the repo root. Pass an absolute "
            f"path, or put the file at <repo>/data/.")
    suffix = p.suffix.lower()
    if suffix in (".csv", ".tsv"):
        delim = "\t" if suffix == ".tsv" else ","
        # Not every published dataset is UTF-8. MDD_100.csv is cp1252 — it
        # carries smart quotes from the original Reddit text and raises
        # UnicodeDecodeError on byte 0xa0 under utf-8. Try the common encodings
        # in order and report which one worked, rather than failing on a file
        # that is perfectly readable.
        last_error: UnicodeDecodeError | None = None
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                with p.open(newline="", encoding=encoding) as fh:
                    rows = [dict(r) for r in csv.DictReader(fh, delimiter=delim)]
                if encoding != "utf-8-sig":
                    logger.info("%s: decoded as %s (not UTF-8)", p.name, encoding)
                return rows
            except UnicodeDecodeError as exc:
                last_error = exc
        raise ValueError(
            f"{p}: could not decode as utf-8, cp1252 or latin-1 ({last_error})"
        ) from last_error
    if suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # {"data": [...]} or {"rows": [...]}
            for key in ("data", "rows", "records", "cases"):
                if isinstance(data.get(key), list):
                    return data[key]
            raise ValueError(f"{p}: JSON object has no list under data/rows/records/cases")
        return list(data)
    if suffix in (".parquet", ".pq"):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Reading parquet requires pandas.") from exc
        return pd.read_parquet(p).to_dict("records")
    raise ValueError(f"Unsupported dataset file type: {p.suffix}")


def _load_hf(hf_id: str, split: str) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            f"Loading {hf_id} from the Hub needs `pip install datasets`, or pass "
            f"--path to a local copy of the CSV."
        ) from exc
    ds = load_dataset(hf_id, split=split)
    return [dict(r) for r in ds]


# ---------------------------------------------------------------------------
# Frame reading — delegated to the project, with a mirrored fallback
# ---------------------------------------------------------------------------

_FALLBACK_FRAME_FIELDS = {
    "normad": {"main": ("main_country", "narrator_country", "country"),
               "challenger": ("challenger_country", "other_country")},
    "moral_decision": {"main": ("passive_agent_description", "passive_agent"),
                       "challenger": ("active_agent_description", "active_agent")},
}


def read_frames(row: dict, dataset_key: str, src: str | None = None) -> tuple[str, str]:
    """(main_frame, challenger_frame) using utils.prompts.read_cultural_frames.

    Falls back to a local mirror of FRAME_FIELDS when the pipeline cannot be
    imported (e.g. `eval inspect` on a machine without the decorator registry
    or boto3). The fallback is deliberately a copy of the same table so a
    divergence shows up as a diff, not as silently different frames.
    """
    try:
        from ._bootstrap import import_frame_reader
        read_cultural_frames, _ = import_frame_reader(src)
        return read_cultural_frames(row, dataset_key)
    except Exception as exc:                       # noqa: BLE001 - diagnostic only
        logger.debug("Falling back to local frame reader (%s)", exc)
        mapping = _FALLBACK_FRAME_FIELDS[dataset_key]
        return first_present(row, mapping["main"]), first_present(row, mapping["challenger"])


# ---------------------------------------------------------------------------
# NormAd
# ---------------------------------------------------------------------------

NORMAD_QUESTION = (
    "Is the behaviour described socially acceptable in the cultural context "
    "the narrator is operating in?"
)
NORMAD_LEAKY = ("explanation", "gold_label")


def load_normad(cfg) -> list[EvalRecord]:
    rows = (_read_table(cfg.path, cfg.src) if cfg.path
            else _load_hf(NORMAD_HF_ID, cfg.split))
    rows = [normalize_row(r) for r in rows]

    wanted = {c.strip().lower().replace(" ", "_") for c in cfg.countries}
    records: list[EvalRecord] = []
    for i, row in enumerate(rows):
        country = first_present(row, ("country",)).lower()
        if wanted and country not in wanted:
            continue
        story = first_present(row, ("story",))
        gold = first_present(row, ("gold_label", "label")).lower()
        if not story or not gold:
            continue

        main_frame, challenger_frame = read_frames(row, "normad", cfg.src)
        scenario = story
        # NormAd's three conditions. The pipeline consumes the country AS A
        # FRAME rather than as prose, so `country` adds nothing to the text —
        # that is the point of the condition, and the difference between
        # `none` and `country` isolates the frame's contribution.
        if cfg.normad_context == "none":
            main_frame = ""
        elif cfg.normad_context == "rot":
            rot = first_present(row, ("rule_of_thumb", "rule_of_thumb_"))
            if rot:
                scenario = f"{story}\n\nRelevant social expectation: {rot}"
        elif cfg.normad_context == "background":
            bg = first_present(row, ("background",))
            if bg:
                scenario = f"{story}\n\nCultural background:\n{bg}"

        if cfg.normad_challenger and cfg.normad_challenger != "neutral":
            challenger_frame = cfg.normad_challenger

        rec = EvalRecord(
            record_id=str(first_present(row, ("id",)) or i),
            dataset="normad",
            scenario=scenario.strip(),
            question=NORMAD_QUESTION,
            gold=gold,
            main_frame=main_frame,
            challenger_frame=challenger_frame,
            group=country,
            raw=row,
        )
        assert_no_leakage(rec, row, NORMAD_LEAKY)
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# MDD
#
# The repo ships a CSV whose exact headers are not published in the README, so
# every field is resolved through an alias list and can be overridden on the
# command line. `python -m eval inspect --dataset mdd --path ...` prints what
# was detected before a run spends any money.
# ---------------------------------------------------------------------------

MDD_SCENARIO_ALIASES = ("summary", "case_summary", "summarized_case", "summarised_case",
                        "case_text", "case", "scenario", "situation", "story", "text",
                        "description", "title")
MDD_LABEL_ALIASES = ("moral_decision", "decision", "moral_judgement", "moral_judgment",
                     "gold_label", "label", "verdict", "ethical_decision", "morality")
MDD_GROUP_ALIASES = ("domain", "applied_ethics_domain", "category", "subreddit")

# Feature columns folded into the scenario text when no prose column exists,
# in the order the MDD summarization template uses them.
MDD_FEATURE_ORDER = (
    ("active_agent", "Active agent"),
    ("passive_agent", "Passive agent"),
    ("relationship_between_agents", "Relationship between the agents"),
    ("action_done_by_active_agent", "Action taken by the active agent"),
    ("action", "Action taken"),
    ("domain", "Domain"),
    ("ethical_issues", "Ethical issues raised"),
    ("consequences", "Consequences"),
    ("severity", "Severity of the consequences"),
    ("utility", "Utility of the consequences"),
    ("duration", "Duration of the consequences"),
    ("moral_intentions", "Moral intention of the active agent"),
    ("moral_intention", "Moral intention of the active agent"),
    ("ethical_principles_upheld", "Ethical principles upheld"),
    ("ethical_principles_violated", "Ethical principles violated"),
)

MDD_QUESTION = (
    "Was the action taken by the active agent morally acceptable, given the "
    "situation and the parties involved?"
)


def _compose_mdd_scenario(row: dict, label_field: str | None) -> str:
    """Build a readable case description from the MDD feature columns.

    Used only when the row has no prose column. Whatever field holds the moral
    decision is excluded — the pipeline must not be shown the answer.
    """
    skip = {label_field or "", "ethics_score", "score", "explanation", "reason",
            "rationale", "justification"}
    lines = []
    seen: set[str] = set()
    for key, label in MDD_FEATURE_ORDER:
        if key in skip or key in seen:
            continue
        value = first_present(row, (key,))
        if value:
            seen.add(key)
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def load_mdd(cfg) -> list[EvalRecord]:
    path = cfg.path
    if not path:
        raise ValueError(
            "MDD is not on the Hub — clone kracr/moral-decision-dataset and pass "
            f"--path path/to/code/MDD/{MDD_DEFAULT_FILE}"
        )
    rows = [normalize_row(r) for r in _read_table(path, cfg.src)]
    if not rows:
        return []

    label_field = cfg.mdd_label_field or resolve_field(rows, MDD_LABEL_ALIASES)
    if label_field is None:
        raise ValueError(
            f"No moral-decision column found in {path}. Columns present: "
            f"{sorted(rows[0])}. Pass --mdd-label-field <column>."
        )
    scenario_field = cfg.mdd_scenario_field or resolve_field(rows, MDD_SCENARIO_ALIASES)
    group_field = resolve_field(rows, MDD_GROUP_ALIASES)

    wanted = {d.strip().lower() for d in cfg.domains}
    records: list[EvalRecord] = []
    for i, row in enumerate(rows):
        gold = str(row.get(label_field, "")).strip().lower()
        if not gold:
            continue
        group = str(row.get(group_field, "")).strip().lower() if group_field else ""
        if wanted and group not in wanted:
            continue

        scenario = (str(row.get(scenario_field, "")).strip() if scenario_field else "")
        extras = _compose_mdd_scenario(row, label_field)
        if scenario and extras:
            scenario = f"{scenario}\n\n{extras}"
        elif not scenario:
            scenario = extras
        if not scenario:
            continue

        main_frame, challenger_frame = read_frames(row, "moral_decision", cfg.src)
        rec = EvalRecord(
            record_id=str(first_present(row, ("id", "case_id", "index")) or i),
            dataset="mdd",
            scenario=scenario,
            question=MDD_QUESTION,
            gold=gold,
            # The project's mapping: the passive agent (who the action landed
            # on) frames the main reading, the active agent (who acted) frames
            # the challenger. Mediation then adjudicates between them.
            main_frame=main_frame,
            challenger_frame=challenger_frame,
            group=group,
            raw=row,
        )
        assert_no_leakage(rec, row, (label_field, "explanation", "ethics_score"))
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Leakage guard + sampling
# ---------------------------------------------------------------------------

def assert_no_leakage(rec: EvalRecord, row: dict, leaky_fields: Sequence[str]) -> None:
    """Fail loudly if an answer-bearing field ended up in the scenario text.

    A leak here would not crash anything — it would just quietly produce a
    great accuracy number, which is the worst possible failure mode for an
    evaluation harness.
    """
    haystack = rec.scenario.lower()
    for f in leaky_fields:
        if not f:
            continue
        value = str(row.get(f, "")).strip().lower()
        if len(value) >= 25 and value in haystack:
            raise ValueError(
                f"[{rec.dataset}:{rec.record_id}] field '{f}' leaked into the "
                f"scenario text handed to the pipeline."
            )


def load_records(cfg) -> list[EvalRecord]:
    if cfg.dataset == "normad":
        return load_normad(cfg)
    if cfg.dataset == "mdd":
        return load_mdd(cfg)
    raise ValueError(f"Unknown dataset {cfg.dataset!r}")


def sample_records(records: Sequence[EvalRecord], n: int, seed: int = 0,
                   stratify: bool = True) -> list[EvalRecord]:
    """Take n records, by default balanced across gold labels.

    NormAd is dominated by `yes`, so an unstratified sample of 20 can contain
    zero negatives and report 85% accuracy for a classifier that always says
    yes. Stratifying makes a small sample say something.
    """
    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)
    if n <= 0 or n >= len(pool):
        return pool
    if not stratify:
        return pool[:n]

    by_label: dict[str, list[EvalRecord]] = {}
    for r in pool:
        by_label.setdefault(r.gold, []).append(r)

    chosen: list[EvalRecord] = []
    labels = sorted(by_label, key=lambda k: -len(by_label[k]))
    # Round-robin across labels until n is reached or every bucket is drained.
    idx = {lab: 0 for lab in labels}
    while len(chosen) < n and any(idx[lab] < len(by_label[lab]) for lab in labels):
        for lab in labels:
            if len(chosen) >= n:
                break
            if idx[lab] < len(by_label[lab]):
                chosen.append(by_label[lab][idx[lab]])
                idx[lab] += 1
    rng.shuffle(chosen)
    return chosen


def describe(records: Sequence[EvalRecord]) -> dict:
    """Summary used by `eval inspect` and written into every run's config."""
    return {
        "n": len(records),
        "labels": dict(Counter(r.gold for r in records).most_common()),
        "groups": dict(Counter(r.group for r in records).most_common()[:15]),
        "framed_main": sum(1 for r in records if r.main_frame),
        "framed_challenger": sum(1 for r in records if r.challenger_frame),
        "median_scenario_chars": (
            sorted(len(r.scenario) for r in records)[len(records) // 2] if records else 0),
    }

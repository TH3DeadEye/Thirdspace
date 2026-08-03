"""Normalize MDD_100.csv into eval/gold/mdd_cases.jsonl. Zero LLM cost.

Pure data transformation, so it can (and should) run before the pipeline is
finished — the mapping it applies is pre-registered in label_maps/mdd_v1.yaml.

One record per case:

    {
      "id": "mdd-0002",
      "input_text": "I got thinking, If someone asks for advice on Reddit...",
      "used_summary_only": false,
      "gold_label": "morally_wrong",
      "domain": "Legal Ethics",
      "frames": {                      # -> main / challenger cultural frames
        "main": "original redditor",           # the party acted upon
        "challenger": "redditor who gave advice"  # the party who acted
      },
      "structured": {                  # gold decomposition, for the internal classifier
        "agents": ["redditor who gave advice", "original redditor"],
        "actions": ["gave wrong advice"],
        "consequences": ["harm or death"],
        "context": ["Legal Ethics"],
        "representations": ["Respect", "Reciprocity", "Integrity", "Accountability"]
      },
      "scores": {"severity": -0.8, "utility": -0.9, "duration": -0.7, "intention": -0.5},
      "gold_explanation": "Giving wrong advice can lead to harm, ..."
    }

Usage:
    python eval/build_gold_mdd.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import DATA_DIR, GOLD_DIR, LABEL_MAPS_DIR  # noqa: E402

MDD_FILE = DATA_DIR / "MDD_100.csv"
GOLD_FILE = GOLD_DIR / "mdd_cases.jsonl"
LABEL_MAP_FILE = LABEL_MAPS_DIR / "mdd_v1.yaml"

csv.field_size_limit(10_000_000)


def load_spec() -> dict:
    spec = yaml.safe_load(LABEL_MAP_FILE.read_text(encoding="utf-8"))
    assert spec["version"] == 1, "gold build is pinned to mdd_v1"
    return spec


def _split_principles(*values: str) -> list[str]:
    """MDD packs several principles into one cell: 'Respect, Integrity'."""
    out: list[str] = []
    for v in values:
        out.extend(p.strip() for p in (v or "").split(",") if p.strip())
    return out


def _float_or_none(v: str) -> float | None:
    try:
        return float((v or "").strip())
    except (TypeError, ValueError):
        return None


def main() -> None:
    if not MDD_FILE.exists():
        raise FileNotFoundError(
            f"{MDD_FILE} not found. Download MDD_100.csv from "
            f"github.com/kracr/moral-decision-dataset into eval/data/ first."
        )
    spec = load_spec()
    label_map = {str(k).lower(): v for k, v in spec["mapping"].items()}

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    labels: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    unmapped: Counter[str] = Counter()
    summary_only = 0
    missing_frame = 0

    # cp1252, not UTF-8 (see the label map). errors="replace" keeps one bad byte
    # from killing the build; the affected character becomes U+FFFD.
    with MDD_FILE.open(encoding=spec.get("encoding", "cp1252"), errors="replace") as f, \
         GOLD_FILE.open("w", encoding="utf-8") as out:
        for i, row in enumerate(csv.DictReader(f)):
            raw_label = (row.get("Moral Decision") or "").strip().lower()
            gold = label_map.get(raw_label)
            if gold is None:
                unmapped[raw_label] += 1
                continue

            # Prefer the original post. The summary is a much thinner input, so
            # rows falling back to it are flagged rather than silently mixed in.
            original = (row.get("Original Text") or "").strip()
            case_summary = (row.get("Case Summary") or "").strip()
            input_text = original or case_summary
            if not input_text:
                continue
            if not original:
                summary_only += 1

            active = (row.get("Active Agent") or "").strip()
            passive = (row.get("Passive Agent") or "").strip()
            if not active or not passive:
                missing_frame += 1

            record = {
                "id": f"mdd-{i:04d}",
                "input_text": input_text,
                "used_summary_only": not original,
                "gold_label": gold,
                "domain": (row.get("Domain") or "").strip(),
                # The two parties ARE the two frames: acted-upon vs. actor.
                # Read by prompts.read_cultural_frames via FRAME_FIELDS.
                "frames": {"main": passive, "challenger": active},
                "passive_agent": passive,
                "active_agent": active,
                # Gold decomposition — what the internal-reasoning classifier
                # scores The Occurring against, object by object.
                "structured": {
                    "agents": [v for v in (active, passive) if v],
                    "actions": [v for v in [(row.get("Action") or "").strip()] if v],
                    "consequences": [v for v in [(row.get("Consequence") or "").strip()] if v],
                    "context": [v for v in [(row.get("Domain") or "").strip()] if v],
                    "representations": _split_principles(
                        row.get("Ethical Principles Upheld", ""),
                        row.get("Ethical Principles Violated", "")),
                },
                "principles_upheld": _split_principles(row.get("Ethical Principles Upheld", "")),
                "principles_violated": _split_principles(row.get("Ethical Principles Violated", "")),
                "scores": {
                    "severity": _float_or_none(row.get("Severity of Consequence", "")),
                    "utility": _float_or_none(row.get("Utility of Consequence", "")),
                    "duration": _float_or_none(row.get("Duration of Consequence", "")),
                    "intention": _float_or_none(row.get("Moral Intention", "")),
                },
                "gold_explanation": (row.get("Moral Decision Explanation") or "").strip(),
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            labels[gold] += 1
            if record["domain"]:
                domains[record["domain"]] += 1

    print(f"Wrote {written} records to {GOLD_FILE}")
    print(f"Gold label distribution: {dict(labels)}")
    if written:
        top_label, top_n = labels.most_common(1)[0]
        print(f"MAJORITY BASELINE: always predicting '{top_label}' scores "
              f"{top_n / written:.1%} — any reported accuracy must beat this, "
              f"and macro-F1 must be reported alongside it.")
    print(f"Domains: {len(domains)} distinct, top 5 {dict(domains.most_common(5))}")
    print(f"Rows using Case Summary only (no Original Text): {summary_only}")
    if missing_frame:
        print(f"Rows missing an agent (frame will be neutral for that side): {missing_frame}")
    if unmapped:
        print(f"Rows skipped for unmapped labels: {dict(unmapped)}")


if __name__ == "__main__":
    main()

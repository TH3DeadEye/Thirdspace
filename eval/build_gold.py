"""Join NormAd-Eti scenarios with Hofstede/WVS country data into one gold file.

Pure data transformation — zero LLM calls. Produces eval/gold/normad_frames.jsonl
with one record per NormAd scenario:

    {
      "id": "...",                  # NormAd row ID
      "country": "egypt",           # as coded by NormAd
      "story": "...",               # the narrative passed to the pipeline
      "rule_of_thumb": "...",       # the stated norm (NormAd task context)
      "axis": "...", "subaxis": "...",
      "gold_label": "violate",      # via the FROZEN eval/label_maps/normad_v1.yaml
      "gold_axes": [{"dimension": "power_distance", "value": "high"}, ...],
      "hofstede": {"pdi": 70.0, ...},   # raw scores behind gold_axes
      "wvs": {...} | null           # secondary axes when the WVS CSV exists
    }

Records whose country has no Hofstede row keep gold_axes=null (they can still
serve the ablation, which needs only gold_label). A coverage report prints any
unmatched countries so eval/culture_scores.py's alias table can be extended —
BEFORE any scoring run, never after.

Usage:
    python eval/build_gold.py
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
from eval.culture_scores import (  # noqa: E402
    load_hofstede, load_wvs_axes, compute_medians, country_to_axes,
    resolve_country, normalize_country, most_distant_country,
)

NORMAD_FILE = DATA_DIR / "normad_etiquette.csv"
GOLD_FILE = GOLD_DIR / "normad_frames.jsonl"
LABEL_MAP_FILE = LABEL_MAPS_DIR / "normad_v1.yaml"

# NormAd stories can exceed the csv module's default field cap via the
# Background column.
csv.field_size_limit(10_000_000)


def load_label_map() -> dict[str, str]:
    spec = yaml.safe_load(LABEL_MAP_FILE.read_text(encoding="utf-8"))
    assert spec["version"] == 1, "gold build is pinned to normad_v1"
    return {str(k).lower(): v for k, v in spec["mapping"].items()}


def main() -> None:
    if not NORMAD_FILE.exists():
        raise FileNotFoundError(
            f"{NORMAD_FILE} not found. Run `python eval/fetch_data.py` first."
        )
    label_map = load_label_map()
    hofstede = load_hofstede()
    medians = compute_medians(hofstede)
    wvs = load_wvs_axes()

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    records = 0
    unmatched_countries: Counter[str] = Counter()
    unmapped_labels: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    unpaired_countries: Counter[str] = Counter()
    pairings: Counter[str] = Counter()

    with NORMAD_FILE.open(encoding="utf-8-sig") as f, \
         GOLD_FILE.open("w", encoding="utf-8") as out:
        for row in csv.DictReader(f):
            country = row["Country"].strip()
            raw_label = (row.get("Gold Label") or "").strip().lower()
            gold_label = label_map.get(raw_label)
            if gold_label is None:
                unmapped_labels[raw_label] += 1
                continue

            axes = country_to_axes(country, hofstede, medians)
            if axes is None:
                unmatched_countries[country] += 1

            # The challenger's contrasting frame. NormAd gives one country; the
            # pipeline needs two frames whose divergence signals that a norm is
            # operative, so the distant pole is derived from Hofstede distance
            # and cached here (build_gold is the only place it is computed, so
            # every runner scores against the same pairing).
            distant = most_distant_country(country, hofstede)
            if distant is None:
                unpaired_countries[country] += 1

            wvs_row = wvs.get(normalize_country(country)) if wvs else None
            record = {
                "id": row["ID"],
                "country": country,
                "challenger_country": distant[0] if distant else "",
                "challenger_distance": round(distant[1], 1) if distant else None,
                # How many Hofstede dimensions the pairing was computed on (max 6).
                # A pairing from 2 dimensions is weaker evidence than one from 6;
                # runners can filter on this rather than trusting all pairings equally.
                "challenger_dimensions": distant[2] if distant else 0,
                "story": row["Story"].strip(),
                "rule_of_thumb": (row.get("Rule-of-Thumb") or "").strip(),
                "axis": (row.get("Axis") or "").strip(),
                "subaxis": (row.get("Subaxis") or "").strip(),
                "gold_label": gold_label,
                "gold_axes": (
                    [a.model_dump() for a in axes] if axes is not None else None
                ),
                "hofstede": resolve_country(country, hofstede),
                "wvs": wvs_row,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1
            label_counts[gold_label] += 1
            if distant:
                pairings[f"{normalize_country(country)} -> {distant[0]}"] += 1

    print(f"Wrote {records} records to {GOLD_FILE}")
    print(f"Gold label distribution: {dict(label_counts)}")
    covered = records - sum(unmatched_countries.values())
    print(f"Hofstede frame coverage: {covered}/{records} records")
    paired = records - sum(unpaired_countries.values())
    print(f"Challenger frame derived:  {paired}/{records} records")
    if pairings:
        print("Most common source -> challenger pairings:")
        for pair, n in pairings.most_common(5):
            print(f"  {pair}: {n} records")
    if unpaired_countries:
        print(f"Countries with no derivable challenger frame: "
              f"{dict(unpaired_countries.most_common(5))}")
    if unmatched_countries:
        print(
            "Countries without a Hofstede match "
            "(extend COUNTRY_ALIASES in eval/culture_scores.py):"
        )
        for name, n in unmatched_countries.most_common():
            print(f"  {name}: {n} records")
    if unmapped_labels:
        print(f"Rows skipped for unmapped gold labels: {dict(unmapped_labels)}")
    if not wvs:
        print("WVS axes: not present (optional — see eval/fetch_data.py)")


if __name__ == "__main__":
    main()

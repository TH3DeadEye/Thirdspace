"""Download the public evaluation datasets into eval/data/. Zero LLM cost.

Fetches:
  - Hofstede 6-dimension country scores (geerthofstede.com, free for research)
  - NormAd-Eti (HuggingFace akhilayerukola/NormAd, CC-BY-4.0):
    2,633 narrative etiquette scenarios across 75 countries with
    yes/no/neutral adherence labels

The World Values Survey cannot be fetched automatically (Kaggle/WVS require
an authenticated download); instructions are printed instead. WVS is a
secondary/robustness signal — the core gold build works without it.

Usage:
    python eval/fetch_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import DATA_DIR

HOFSTEDE_URL = (
    "https://geerthofstede.com/wp-content/uploads/2016/08/"
    "6-dimensions-for-website-2015-08-16.csv"
)
NORMAD_URL = (
    "https://huggingface.co/datasets/akhilayerukola/NormAd/"
    "resolve/main/normad_etiquette_final_data.csv"
)

HOFSTEDE_FILE = DATA_DIR / "hofstede_2015.csv"
NORMAD_FILE = DATA_DIR / "normad_etiquette.csv"
WVS_FILE = DATA_DIR / "wvs_country_axes.csv"  # user-supplied, optional

WVS_INSTRUCTIONS = f"""\
[optional] World Values Survey (secondary/robustness signal):
  The WVS requires an authenticated download (kaggle.com/datasets/fernandol/
  world-values-survey or worldvaluessurvey.org). To include it, aggregate to
  country level and save as {WVS_FILE}
  with header:  country,traditional_secular,survival_selfexpression
  (Inglehart-Welzel map coordinates; 0 is the map's center on both axes.)
"""


def _download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already present: {dest.name} ({dest.stat().st_size:,} bytes)")
        return
    print(f"  downloading {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    print(f"  saved {dest.name} ({len(resp.content):,} bytes)")


def main() -> None:
    print("Hofstede 6-D country scores:")
    _download(HOFSTEDE_URL, HOFSTEDE_FILE)
    print("NormAd-Eti (CC-BY-4.0 — cite Rao et al., NAACL 2025):")
    _download(NORMAD_URL, NORMAD_FILE)
    print()
    if WVS_FILE.exists():
        print(f"WVS country axes present: {WVS_FILE.name}")
    else:
        print(WVS_INSTRUCTIONS)


if __name__ == "__main__":
    main()

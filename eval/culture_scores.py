"""Country -> gold CultureAxis frames from real Hofstede (and optional WVS) data.

This is the label GENERATOR for frame-detection evaluation: given a scenario's
country, it produces the dimensional frame the Phase-0 detector *should* have
inferred, using published country scores instead of an LLM's priors.

The output vocabulary is exactly models.CULTURE_DIMENSION_MENU (Hofstede's six
dimensions, binary values), which the menu's own docstring designates as the
plug-in point for real Hofstede data. Continuous 0-100 scores are binarized
against the corpus-wide median per dimension (pre-registered choice — see
eval/README.md; the midpoint-50 alternative was rejected because several
dimensions' observed distributions are not centered on 50).

WVS (Inglehart-Welzel) axes, when the user has supplied the optional CSV, are
carried as SECONDARY columns in the gold table for robustness checks. They are
deliberately NOT merged into the menu vocabulary: the menu stays Hofstede-six
so gold frames and detected frames share one vocabulary.
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import DATA_DIR  # noqa: E402  (bootstraps thirdspace/ onto sys.path)
from models import CultureAxis  # noqa: E402

HOFSTEDE_FILE = DATA_DIR / "hofstede_2015.csv"
WVS_FILE = DATA_DIR / "wvs_country_axes.csv"

# Hofstede CSV column -> (menu dimension, value if score >= median, value if below).
# Direction follows Hofstede's own scale semantics: a high idv score means
# individualist, a high ltowvs score means long-term oriented, etc.
HOFSTEDE_TO_MENU: dict[str, tuple[str, str, str]] = {
    "pdi":    ("power_distance",        "high",          "low"),
    "idv":    ("individualism",         "individualist", "collectivist"),
    "mas":    ("masculinity",           "masculine",     "feminine"),
    "uai":    ("uncertainty_avoidance", "high",          "low"),
    "ltowvs": ("long_term_orientation", "long",          "short"),
    "ivr":    ("indulgence",            "indulgent",     "restrained"),
}

# NormAd country names (normalized) -> Hofstede country names (normalized),
# where simple normalization does not already match. Extend when build_gold.py
# reports unmatched countries; every entry here should be justified by an
# actual row in both datasets.
COUNTRY_ALIASES: dict[str, str] = {
    "united states": "usa",
    "united states of america": "usa",
    "united kingdom": "great britain",
    "south korea": "korea south",
    "czechia": "czech rep",
    "czech republic": "czech rep",
    "slovakia": "slovak rep",
    "bosnia and herzegovina": "bosnia",
    "north macedonia": "macedonia rep",
    "türkiye": "turkey",
    "turkiye": "turkey",
}
# NOT aliased on purpose: countries absent from the Hofstede table (Sudan,
# Somalia, Myanmar, Afghanistan, Nepal, Samoa, ...) could be mapped to
# Hofstede's regional estimate rows ("Africa East", "Arab countries"), but v1
# stays conservative — country-level matches only. Records without a match keep
# gold_axes=null and still serve the ablation (which needs only gold_label).


def normalize_country(name: str) -> str:
    """Lowercase, treat _/- as spaces, drop periods/apostrophes, collapse
    whitespace. NormAd codes countries like "united_states_of_america";
    Hofstede like "U.S.A."."""
    cleaned = (
        name.lower()
        .replace("_", " ").replace("-", " ")
        .replace(".", "").replace("'", "").replace("’", "")
    )
    return " ".join(cleaned.split())


def load_hofstede(path: Path = HOFSTEDE_FILE) -> dict[str, dict[str, float]]:
    """Parse the geerthofstede.com CSV (semicolon-separated, '#NULL!' for
    missing) into {normalized country: {column: score}}. Missing dimensions are
    simply absent from the inner dict."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python eval/fetch_data.py` first."
        )
    scores: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            country = normalize_country(row["country"])
            dims: dict[str, float] = {}
            for col in HOFSTEDE_TO_MENU:
                raw = (row.get(col) or "").strip()
                if raw and raw != "#NULL!":
                    dims[col] = float(raw)
            if dims:
                scores[country] = dims
    return scores


def compute_medians(scores: dict[str, dict[str, float]]) -> dict[str, float]:
    """Per-dimension median across all countries that have that dimension."""
    medians: dict[str, float] = {}
    for col in HOFSTEDE_TO_MENU:
        values = [dims[col] for dims in scores.values() if col in dims]
        if values:
            medians[col] = statistics.median(values)
    return medians


def resolve_country(
    name: str, scores: dict[str, dict[str, float]]
) -> dict[str, float] | None:
    """Look up a country by normalized name, applying the alias table."""
    norm = normalize_country(name)
    norm = COUNTRY_ALIASES.get(norm, norm)
    return scores.get(norm)


def most_distant_country(
    name: str,
    scores: dict[str, dict[str, float]],
    min_shared: int = 2,
) -> tuple[str, float, int] | None:
    """The Hofstede-furthest country from `name` — the challenger's frame.

    NormAd supplies ONE country per row, but the design needs two contrasting
    frames: the main agent reasons from the source culture, the challenger from
    a culturally distant one, and their divergence is the signal that a norm is
    operative. With no second country in the data, it has to be derived.

    Hofstede coverage is uneven — only 65 of 111 countries carry all six
    dimensions (Egypt, for instance, has two). Requiring all six would silently
    drop ~40% of NormAd, so distance is computed over the dimensions the two
    countries SHARE, and normalized by sqrt(n) so a 3-dimension distance is
    comparable to a 6-dimension one.

    Returns (country, normalized_distance, n_shared_dimensions), or None when
    the source country is absent from the table or shares too few dimensions
    with any candidate. n_shared is returned rather than used as a filter: a
    pairing derived from 2 dimensions is weaker evidence than one from 6, and
    that is the runner's call to make, not the builder's.

    PRE-REGISTRATION NOTE: "maximally distant" is a deliberate choice, frozen in
    eval/label_maps/normad_v1.yaml. It may make divergence too easy to detect —
    a mid-distance country would be a harder test. Do not switch after seeing
    scores; create a v2 map and re-run.
    """
    src = resolve_country(name, scores)
    if src is None:
        return None
    src_dims = [c for c in HOFSTEDE_TO_MENU if c in src]
    if len(src_dims) < min_shared:
        return None
    src_norm = COUNTRY_ALIASES.get(normalize_country(name), normalize_country(name))

    # Rank lexicographically: MOST shared dimensions first, then greatest
    # distance within that tier. Distance alone would be wrong — normalizing by
    # sqrt(n) leaves few-dimension comparisons noisier, so a plain argmax
    # systematically picks the least-informed pairing (observed: a 6-dimension
    # country being matched through a 2-dimension distance). Comparing on as
    # much evidence as exists, and only then maximizing, avoids that.
    best: tuple[str, float, int] | None = None
    for other, other_dims in scores.items():
        if other == src_norm:
            continue
        shared = [d for d in src_dims if d in other_dims]
        if len(shared) < min_shared:
            continue
        # Normalized (RMS) so tiers with different dimension counts stay readable.
        distance = (sum((src[d] - other_dims[d]) ** 2 for d in shared) / len(shared)) ** 0.5
        if best is None or (len(shared), distance) > (best[2], best[1]):
            best = (other, distance, len(shared))
    return best


def country_to_axes(
    name: str,
    scores: dict[str, dict[str, float]],
    medians: dict[str, float],
) -> list[CultureAxis] | None:
    """Gold dimensional frame for one country, in menu vocabulary.

    Returns None when the country is not in the Hofstede table at all;
    dimensions the table lacks for this country are skipped (a partial gold
    frame scores only the dimensions it has — see run_frame_eval.py).
    """
    dims = resolve_country(name, scores)
    if dims is None:
        return None
    axes: list[CultureAxis] = []
    for col, (dimension, high_value, low_value) in HOFSTEDE_TO_MENU.items():
        if col not in dims or col not in medians:
            continue
        value = high_value if dims[col] >= medians[col] else low_value
        axes.append(CultureAxis(dimension=dimension, value=value))
    return axes


def load_wvs_axes(path: Path = WVS_FILE) -> dict[str, dict[str, float]]:
    """Optional user-supplied WVS country axes (see fetch_data.py for the
    expected header). Returns {} when the file is absent — the gold build
    treats WVS as a secondary column, never a requirement."""
    if not path.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            out[normalize_country(row["country"])] = {
                "traditional_secular": float(row["traditional_secular"]),
                "survival_selfexpression": float(row["survival_selfexpression"]),
            }
    return out

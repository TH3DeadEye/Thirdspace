"""Unit tests for eval/culture_scores.py — the Hofstede -> CultureAxis bridge.

Logic tests run against a small inline CSV so they need no network/data.
A real-data sanity test runs only when eval/data/hofstede_2015.csv exists
(fetched via eval/fetch_data.py).
"""
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "thirdspace"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval.culture_scores import (
    HOFSTEDE_FILE, load_hofstede, compute_medians, country_to_axes,
    normalize_country, resolve_country,
)

# Five synthetic countries with a known median per column (middle row = Cmid).
FIXTURE = """\
ctr;country;pdi;idv;mas;uai;ltowvs;ivr
AAA;Alphaland;90;10;80;95;90;10
BBB;Betaland;70;30;60;75;70;30
MID;Midland;50;50;50;50;50;50
DDD;Deltaland;30;70;40;25;30;70
EEE;Epsilonia;10;91;20;5;10;90
NUL;Nulland;#NULL!;#NULL!;#NULL!;#NULL!;40;#NULL!
"""


class CultureScoreLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, encoding="utf-8"
        )
        tmp.write(FIXTURE)
        tmp.close()
        cls.path = Path(tmp.name)
        cls.scores = load_hofstede(cls.path)
        cls.medians = compute_medians(cls.scores)

    @classmethod
    def tearDownClass(cls):
        cls.path.unlink(missing_ok=True)

    def test_null_dimensions_are_absent_not_zero(self):
        self.assertEqual(set(self.scores["nulland"]), {"ltowvs"})

    def test_medians_ignore_nulls(self):
        # pdi values: 90,70,50,30,10 -> median 50 (Nulland's #NULL! excluded).
        self.assertEqual(self.medians["pdi"], 50)
        # ltowvs values: 90,70,50,30,10,40 -> median 45.
        self.assertEqual(self.medians["ltowvs"], 45)

    def test_high_scoring_country_binarizes_high(self):
        axes = {a.dimension: a.value for a in
                country_to_axes("Alphaland", self.scores, self.medians)}
        self.assertEqual(axes["power_distance"], "high")
        self.assertEqual(axes["individualism"], "collectivist")
        self.assertEqual(axes["masculinity"], "masculine")
        self.assertEqual(axes["uncertainty_avoidance"], "high")
        self.assertEqual(axes["long_term_orientation"], "long")
        self.assertEqual(axes["indulgence"], "restrained")

    def test_low_scoring_country_binarizes_low(self):
        axes = {a.dimension: a.value for a in
                country_to_axes("Epsilonia", self.scores, self.medians)}
        self.assertEqual(axes["power_distance"], "low")
        self.assertEqual(axes["individualism"], "individualist")
        self.assertEqual(axes["indulgence"], "indulgent")

    def test_partial_country_yields_partial_frame(self):
        axes = country_to_axes("Nulland", self.scores, self.medians)
        self.assertEqual([a.dimension for a in axes], ["long_term_orientation"])

    def test_unknown_country_returns_none(self):
        self.assertIsNone(country_to_axes("Atlantis", self.scores, self.medians))

    def test_normalization_handles_underscores_and_diacritics_aliases(self):
        self.assertEqual(normalize_country("united_states_of_america"),
                         "united states of america")
        self.assertEqual(normalize_country("U.S.A."), "usa")
        self.assertEqual(normalize_country("timor-leste"), "timor leste")


class RealHofstedeData(unittest.TestCase):
    """Sanity checks against the actual 2015 table (skipped if not fetched)."""

    @classmethod
    def setUpClass(cls):
        if not HOFSTEDE_FILE.exists():
            raise unittest.SkipTest("hofstede_2015.csv not fetched")
        cls.scores = load_hofstede()
        cls.medians = compute_medians(cls.scores)

    def test_usa_is_individualist_low_power_distance(self):
        axes = {a.dimension: a.value for a in
                country_to_axes("united_states_of_america", self.scores, self.medians)}
        self.assertEqual(axes["individualism"], "individualist")  # idv=91
        self.assertEqual(axes["power_distance"], "low")           # pdi=40

    def test_china_is_collectivist_high_power_distance_long_term(self):
        axes = {a.dimension: a.value for a in
                country_to_axes("china", self.scores, self.medians)}
        self.assertEqual(axes["individualism"], "collectivist")
        self.assertEqual(axes["power_distance"], "high")
        self.assertEqual(axes["long_term_orientation"], "long")

    def test_aliases_resolve_against_real_table(self):
        for normad_name in ("south_korea", "united_kingdom", "türkiye",
                            "north_macedonia", "bosnia_and_herzegovina"):
            self.assertIsNotNone(
                resolve_country(normad_name, self.scores),
                f"alias failed for {normad_name}",
            )


if __name__ == "__main__":
    unittest.main()

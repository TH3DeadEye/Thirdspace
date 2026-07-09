# ThirdSpace Evaluation Harness

Dataset-driven evaluation with external gold labels — built because the
internal LLM judge saturates (straight 5s on every reviewed run) and cannot
measure anything. Design follows the evaluation plan in
`docs/evaluation third space Jul9th.docx` (§11): pre-registered label
mappings, real human-labeled data as primary evidence, mechanism vs
population-adaptation kept distinct.

## Spend policy — read first

**Every runner is a dry run by default.** A dry run prints the scenario
count, exact LLM call count, and a dollar estimate, then exits — zero Bedrock
calls. Real runs require `--confirm-spend`, honor `--limit`, and append
per-scenario results to `eval/results/*.jsonl` so an interrupted run resumes
without re-paying for finished work. The pipeline's own budget circuit
breaker (`thirdspace/budget.py`) caps each session as a final backstop.

## Datasets

| Dataset | Role | How |
|---|---|---|
| Hofstede 6-D country scores | Gold *frames*: real per-country values binarized into `CULTURE_DIMENSION_MENU` | fetched automatically |
| NormAd-Eti (2,633 scenarios, 75 countries, CC-BY-4.0) | Scenario corpus + gold adhere/violate/neither labels | fetched automatically |
| World Values Survey | Optional secondary axes on the gold table | manual (auth-gated); see `fetch_data.py` |

Cite Rao et al., "NormAd: A Framework for Measuring the Cultural Adaptability
of Large Language Models" (NAACL 2025) for NormAd-Eti.

## Workflow

```bash
# 1. Zero-cost setup
python eval/fetch_data.py        # download Hofstede + NormAd into eval/data/
python eval/build_gold.py        # join -> eval/gold/normad_frames.jsonl + coverage report

# 2. Dry runs (default — no spend)
python eval/run_frame_eval.py
python eval/run_ablation_normad.py

# 3. Real runs (explicit spend, start small)
python eval/run_frame_eval.py --limit 20 --confirm-spend         # ~$0.25
python eval/run_ablation_normad.py --arm single --limit 15 --confirm-spend
python eval/run_ablation_normad.py --arm pipeline --limit 15 --confirm-spend

# 4. Aggregate
python eval/report.py
```

## What each evaluation answers

**`run_frame_eval.py` — does Phase-0 frame detection find the real culture?**
One LLM call per scenario. The detector's main frame is scored per-dimension
against the country's actual Hofstede values (the "generate labels from a
dataset" idea: country tag → real scores → gold `CultureAxis` list).
Pre-registered caveat: NormAd stories are single-culture narratives, not
two-party conflicts, so this is off the detector's design distribution — the
lenient either-frame metric is reported alongside to separate "wrong culture"
from "right culture, wrong role".

**`run_ablation_normad.py` — does the two-agent architecture add anything?**
The core open question (design doc §8.1). Identical inputs to both arms
(story + country; the rule-of-thumb is withheld — it nearly encodes the
answer): one model call vs the full pipeline at `max_cycles=1`. Scored on
accuracy and per-country parity against NormAd's human-sourced labels,
mirroring the published single-agent vs debate comparison (Ki et al., ACL
2025). The pipeline's final documents are also grep'd against
`models.LEAK_PHRASES` as the runtime rubric-leak regression check, and
`convergence_reason` telemetry shows how each run terminated.

## Pre-registration rules

- Label mappings live in `eval/label_maps/*.yaml`, are versioned, and were
  committed **before** any scoring run. Never edit a frozen map — create
  `_v2` and re-run everything that used v1.
- The Hofstede binarization (per-dimension corpus median) and the country
  alias table were fixed before scoring; `build_gold.py` prints unmatched
  countries so alias gaps surface before, not after, results exist.
- Keep `--seed` fixed (default 7) so reruns and arms share the same sample.
- Countries absent from Hofstede's table (Sudan, Myanmar, Samoa, ...) keep
  `gold_axes: null` — usable for the ablation, excluded from frame eval. v1
  deliberately does not fall back to Hofstede's regional estimate rows.

## Deferred (documented, not built)

AITA/Moral Dilemma Dataset confusion-matrix pairing (`label_maps/aita_v1.yaml`
is the frozen stub); NormDial / KODIS / Robots-in-the-Middle pairings — all
blocked on a dialogue turn-ingestion layer (`DialogueTurn`, `source_turn_ids`)
that narrative corpora don't need.

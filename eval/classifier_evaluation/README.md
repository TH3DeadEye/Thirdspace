# `eval/` — measuring ThirdSpace against NormAd and MDD

The pipeline produces typed conflicts, a wholeness synthesis and (optionally) a
document. Neither dataset asks for any of those; both ask for a **label**. This
package closes that gap: it reads the frames out of each dataset row using the
project's own `read_cultural_frames`, runs the pipeline, and maps the mediation
output back into the dataset's label vocabulary — then scores it.

```
dataset row ──► frames (main / challenger)  ──► SessionMemory ──► graph
                 scenario text                                       │
                                                                     ▼
   accuracy ◄── metrics ◄── label ◄── classifier ◄── mediation output
```

## Install / layout

Drop the folder at the repo root, next to the source directory:

```
your-repo/
  src/            models.py graph.py agents.py llm.py budget.py utils/…
  eval/           ← this package
  outputs/eval/   ← written by runs
```

The pipeline uses flat imports (`from models import …`), so its directory must
be on `sys.path`. `_bootstrap.py` finds it automatically by looking for a
directory containing `models.py` **and** `graph.py`; override with
`THIRDSPACE_SRC=/path/to/src` or `--src`.

Extra dependency, only if you want to pull NormAd from the Hub rather than a
local CSV:

```bash
pip install datasets      # optional
```

## Data

| dataset | where | what the row gives the pipeline |
|---|---|---|
| `normad` | `akhilayerukola/NormAd` (Hub, 2.62k rows) or a local CSV | `Story` → scenario, `Country` → main frame, `Gold Label` ∈ {yes, no, neutral} |
| `mdd` | clone `kracr/moral-decision-dataset`, use `code/MDD/MDD_100.csv` | case text (or the feature columns) → scenario, passive agent → main frame, active agent → challenger frame, the moral-decision column → gold |

MDD's exact column headers aren't published in its README, so every field is
resolved through an alias list and can be overridden. **Run `inspect` first.**

## Use it

```bash
# 0. see what the loader found — no API calls, no cost
python -m eval inspect --dataset normad --n 40
python -m eval inspect --dataset mdd --path data/MDD_100.csv --n 20

# 1. a real run: 20 stratified NormAd records, one cycle, plus the control
python -m eval run --dataset normad --n 20 --classifier hybrid --baseline

# 2. MDD, frames read from the active/passive agent columns
python -m eval run --dataset mdd --path data/MDD_100.csv --n 12 --baseline

# 3. fit the free classifier's thresholds on the run you already paid for
python -m eval calibrate --run-dir outputs/eval/normad-n20-hybrid-…
python -m eval report --run-dir outputs/eval/normad-n20-hybrid-… \
    --thresholds outputs/eval/normad-n20-hybrid-…/thresholds.json
```

`--n` is the sample size, `--seed` makes the sample reproducible, and sampling
is **stratified by gold label by default** — NormAd is dominated by `yes`, and
an unstratified 20-record sample can contain zero negatives and report 85%
accuracy for a classifier that always says yes.

Each run writes `outputs/eval/<run-id>/`:

```
config.json        the exact config, the sample profile, the label vocabulary
predictions.jsonl  one row per record: gold, every head's prediction, full signals, cost
report.md          headline table, per-class, confusion, diagnostics, cost, slices
report.json        the same, machine-readable
sessions/          full SessionMemory per record (--persist-sessions)
```

Rows are flushed as they finish, so a throttling failure at record 17 doesn't
discard the first 16. `--resume` picks up where it stopped.

## The three classifier heads

**`structural`** — a deterministic rule over the mediation signals. No extra
LLM call. Two independently-derived readings that largely agree describe a
situation whose norms are not in dispute (→ acceptable); a few load-bearing
conflicts on *value* / *assumption* cells typed normative or identity-related
describe a behaviour being judged and found wanting (→ unacceptable); anything
else is the pipeline reporting that it didn't settle the question. Its four cut
points are **fitted, not guessed** — `eval calibrate` grid-searches them on
cached signals with k-fold CV and reports the out-of-fold macro-F1.

**`readout`** — one LLM call that sees only The Occurring and the final
mediation (agreements, conflicts with type/severity/resolvability, wholeness).
It never sees the original story text or the gold label.

**`hybrid`** (default) — both. The read-out is the reported prediction; the
structural head is kept alongside and their **agreement rate is a result**: if
the free head tracks the paid one, you can ship the free one.

**`--baseline`** adds a single call on the raw record with no pipeline at all.
Without it, a run answers "what accuracy did ThirdSpace get?" but not "did the
~60 calls per record buy anything?" — which is the question a thesis committee
asks. The report runs **McNemar's test** on the paired records for exactly that.

## What the report gives you

Accuracy alone would be misleading here, so every run reports:

- accuracy with a bootstrap 95% CI (wide on 20 records — that's the point),
- **balanced accuracy** and **macro-F1** (immune to the `yes` skew),
- **Cohen's κ** — near zero means the predictions carry no information beyond
  the label prior, whatever the raw accuracy says,
- the majority-class score on the same sample, printed next to the headline,
- the full confusion matrix and per-class P/R/F1,
- **`dominant_type_by_gold`** — whether the pipeline's own conflict typing
  separates the gold classes at all. If every gold label yields the same
  dominant type, the structural head has nothing to work with, and that's worth
  knowing independently of the accuracy number,
- cost: calls, tokens, wall clock, per record,
- per-country (NormAd) / per-domain (MDD) slices, with slices under 3 records
  dropped rather than reported as 0% or 100%.

Abstentions (a failed or unparseable classifier call) are counted separately,
not scored as wrong — a harness that scores an API failure as a wrong answer is
measuring its own network.

## Cost

One record at `--max-cycles 1` is ~62 Bedrock calls (30 main cells + 30
challenger cells + reformulation + mediation), plus 1–2 classifier calls. So:

| n | cycles | ≈ calls |
|---|---|---|
| 12 | 1 | ~770 |
| 20 | 1 | ~1,280 |
| 20 | 2 | ~2,500 |

`--workers` defaults to **1**: the cell agents already fan out 8 ways inside a
single scenario, and per-minute quota is the binding constraint. Raise it only
after a quota bump. `--budget` sets a hard per-scenario call cap that raises
`BudgetExceeded` rather than silently spending.

## Design notes

- **Frames are never inferred.** `datasets.read_frames` calls the project's
  `utils.prompts.read_cultural_frames`, which is why rows are snake_cased on
  load — NormAd ships `Country`, the project looks for `country`. A local
  mirror of `FRAME_FIELDS` is used only when the pipeline can't be imported
  (e.g. `inspect` on a machine with no boto3); it's a deliberate copy so a
  divergence shows up as a diff.
- **NormAd has one country per row**, so the challenger gets no dataset frame
  and runs in neutral mode — which `prompts.py` already handles by dropping the
  frame block entirely. `--normad-challenger "<frame>"` overrides that if you
  want an explicit second standpoint.
- **`--normad-context {none,country,rot,background}`** mirrors NormAd's three
  conditions. The country is consumed as a *frame*, not as prose, so the
  difference between `none` and `country` isolates the frame's contribution —
  which is the experiment the project is actually about.
- **Leakage is asserted, not assumed.** `assert_no_leakage` raises if NormAd's
  `Explanation` or MDD's decision column ends up inside the scenario text. A
  leak wouldn't crash anything, it would just quietly produce a great number.

## Two things to fix in the pipeline before running multi-cycle

1. `graph.node_integrate` assigns `mem.active_challenger_perspective = prose`.
   `SessionMemory` has no such field, and pydantic v2 raises on assignment to
   an undeclared attribute. That line is only reached when a node changed *and*
   `updated_perspectives` is non-empty — i.e. never at `--max-cycles 1`, and
   quite likely on the first real cycle-2. Delete the line (the next two lines
   already do the right thing via `challenger_working_perspective`).
2. `requirements.txt` doesn't list `json-repair`'s companions used elsewhere,
   and doesn't list `datasets` — add `datasets>=2.0` if you want the harness to
   pull NormAd from the Hub instead of a local CSV.

## Extending to a third dataset

Add a loader returning `list[EvalRecord]`, a `FRAME_FIELDS` entry in the
project's `prompts.py`, a gloss in `eval/prompts.py::GLOSSES`, and a branch in
`datasets.load_records`. Everything else — sampling, vocabulary discovery,
classifiers, metrics, calibration, reporting — is dataset-agnostic.

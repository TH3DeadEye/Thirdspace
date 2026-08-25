# `eval/` — measuring ThirdSpace against NormAd and MDD

Two folders, one direction of dependency. `thirdspace/` runs the algorithm and
knows nothing about evaluation; `eval/` imports it, feeds it labelled data, and
scores what comes back. Nothing in `thirdspace/` ever imports `eval`, so the
pipeline stays runnable on its own (`python thirdspace/main.py`).

```
dataset row ──► frames (main / challenger)  ──► SessionMemory ──► graph
                 scenario text                                       │
                                                                     ▼
   accuracy ◄── metrics ◄── label ◄── classifier ◄── mediation output
```

## Layout

```
<repo>/
├── thirdspace/                   ← the algorithm. unchanged, self-contained.
│   ├── __init__.py
│   ├── main.py                   loads ../.env, writes ../outputs/
│   ├── models.py graph.py agents.py llm.py budget.py metrics.py
│   └── utils/
│       ├── __init__.py           ← required: `from utils import prompts`
│       ├── prompts.py decorators.py parsing.py context.py
│       └── decorator_registry/
│           ├── first-principles.json
│           ├── red-team.json
│           └── blind-spots.json
├── eval/                         ← this package. imports thirdspace, never vice versa.
├── data/                         MDD_100.csv, any local dataset copies
├── outputs/                      thirdspace sessions
│   └── eval/                     eval runs (predictions, reports)
├── .env                          BEDROCK_MODEL_ID, CHALLENGER_MODEL_ID, AWS_REGION…
└── requirements.txt
```

Three constraints this layout satisfies, all of which are load-bearing in your
current code:

- **`utils/` is a subpackage of the algorithm folder**, not a sibling. `llm.py`
  does `from utils.parsing import …`, `agents.py` does `from utils import
  prompts`, `prompts.py` does `from utils.decorators import …`.
- **`decorator_registry/` sits beside `decorators.py`**, because
  `load_decorator` resolves it as `Path(__file__).parent / "decorator_registry"`.
  `prompts.py` loads three of those JSONs at import time, so a missing file is
  an ImportError several frames deep.
- **`.env` is at the repo root**, one level above the algorithm folder —
  `main.py` already reads `parent.parent / ".env"`, and so does this package.

If your folder is called `src/` rather than `thirdspace/`, everything still
works: the bootstrap looks for `thirdspace/`, `third_space/`, `src/`, and the
repo root itself, in that order. Override with `THIRDSPACE_SRC=/path` or `--src`.

## Check the wiring before spending anything

```bash
cd <repo>
python -m eval doctor
```

`doctor` verifies every path, import and environment variable a run depends on
and calls nothing:

```
[PASS] eval package          /repo/eval
[PASS] algorithm folder      /repo/thirdspace
[PASS] repo root             /repo
[PASS] one-way import edge   eval/ is NOT inside the algorithm folder
[PASS] utils/prompts.py      /repo/thirdspace/utils/prompts.py
[PASS] decorator red-team.json
[PASS] .env                  /repo/.env
[FAIL] env CHALLENGER_MODEL_ID   EMPTY (no default in llm.py)
```

That last one is worth calling out. `llm.py` declares `BEDROCK_MODEL_ID` and
`CHALLENGER_MODEL_ID` with **empty defaults**, and reads them at *module import
time*. `main.py` gets away with it because it calls `load_dotenv` before
importing anything; the harness doesn't go through `main.py`, so
`_bootstrap.load_env()` does the same thing, in the same order, before touching
`llm`. Without that the model ids arrive empty and every Bedrock call dies with
a ValidationException 60 calls into a record. `import_pipeline` now refuses to
start if either is blank.

## Install

```bash
pip install -r requirements.txt   # the pipeline's own deps
pip install datasets              # optional: pull NormAd from the Hub
```

The harness itself adds no required dependencies — metrics, bootstrap CIs and
McNemar are pure stdlib, and `.env` is parsed by hand when `python-dotenv`
isn't present.

## Data

| dataset | where | what the row gives the pipeline |
|---|---|---|
| `normad` | `akhilayerukola/NormAd` (Hub, 2.62k rows) or a local CSV | `Story` → scenario, `Country` → main frame, `Gold Label` ∈ {yes, no, neutral} |
| `mdd` | clone `kracr/moral-decision-dataset`, use `code/MDD/MDD_100.csv` | case text (or the feature columns) → scenario, passive agent → main frame, active agent → challenger frame, moral-decision column → gold |

`--path data/MDD_100.csv` resolves against the working directory first and the
repo root second, so it works from either. Output goes to
`<repo>/outputs/eval/<run-id>/` regardless of where you launched from —
anchored to the algorithm folder, not to the shell's CWD.

MDD's exact column headers aren't published in its README, so every field goes
through an alias list with overrides. **Run `inspect` on it once first.**

## Use it

```bash
cd <repo>

# 0. layout + env check, then see what the loader found. No calls, no cost.
python -m eval doctor
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

`python -m eval` must be run from the repo root (or with the repo root on
`PYTHONPATH`) — that is how Python finds the package, and it is the only thing
about the harness that cares where you are standing.

`--n` is the sample size, `--seed` makes it reproducible, and sampling is
**stratified by gold label by default** — NormAd is dominated by `yes`, and an
unstratified 20-record sample can contain zero negatives and report 85%
accuracy for a classifier that always says yes.

Each run writes `outputs/eval/<run-id>/`:

```
config.json        exact config, sample profile, label vocabulary
predictions.jsonl  one row per record: gold, every head's prediction, signals, cost
report.md          headline, per-class, confusion, diagnostics, cost, slices
report.json        the same, machine-readable
sessions/          full SessionMemory per record (--persist-sessions)
```

Rows are flushed as they finish, so a throttling failure at record 17 doesn't
discard the first 16. `--resume` picks up where it stopped.

## What lives where

| concern | file | never does |
|---|---|---|
| path/env resolution, pipeline import | `_bootstrap.py` | run anything |
| loading, normalizing, sampling, frames | `datasets.py` | call Bedrock |
| SessionMemory → flat numbers | `signals.py` | call Bedrock |
| mediation output → dataset label | `classifiers.py` | touch the graph |
| the two classifier prompts | `prompts.py` | see the gold label |
| accuracy, macro-F1, κ, McNemar, CIs | `metrics.py` | import the pipeline |
| running + writing predictions | `runner.py` | compute metrics |
| assembling and rendering reports | `report.py` | call Bedrock |
| fitting structural thresholds | `calibrate.py` | call Bedrock |

Only `runner.py` imports the pipeline. Everything else is either pure data or
pure arithmetic, which is why `report`, `calibrate` and `inspect` all work
offline on a machine with no AWS credentials.

## The three classifier heads

**`structural`** — a deterministic rule over the mediation signals, no extra
LLM call. Two independently-derived readings that largely agree describe a
situation whose norms aren't in dispute (→ acceptable); a few load-bearing
conflicts on *value* / *assumption* cells typed normative or identity-related
describe a behaviour being judged and found wanting (→ unacceptable); anything
else is the pipeline reporting it didn't settle the question. Its four cut
points are **fitted, not guessed** — `eval calibrate` grid-searches them on
cached signals with k-fold CV and reports out-of-fold macro-F1.

**`readout`** — one LLM call that sees only The Occurring and the final
mediation (agreements, conflicts with type/severity/resolvability, wholeness).
Never the original story text, never the gold label.

**`hybrid`** (default) — both. Read-out is the reported prediction; the
structural head is kept alongside and their **agreement rate is a result**: if
the free head tracks the paid one, ship the free one.

**`--baseline`** adds a single call on the raw record with no pipeline at all.
Without it a run answers "what accuracy did ThirdSpace get?" but not "did the
~60 calls per record buy anything?" — which is the question a committee asks.
The report runs **McNemar's test** on the paired records for exactly that.

## What the report gives you

Accuracy alone would mislead here, so every run reports accuracy with a
bootstrap 95% CI, **balanced accuracy** and **macro-F1** (immune to the `yes`
skew), **Cohen's κ** (near zero = no information beyond the label prior,
whatever the raw accuracy says), the majority-class score printed next to the
headline, the full confusion matrix, per-country/per-domain slices (slices
under 3 records dropped rather than reported as 0% or 100%), cost per record,
and **`dominant_type_by_gold`** — whether your conflict taxonomy separates the
gold classes at all. If every gold label yields the same dominant type, the
structural head has nothing to work with, and that's worth knowing
independently of the accuracy number.

Abstentions (a failed or unparseable classifier call) are counted separately,
not scored as wrong — a harness that scores an API failure as a wrong answer is
measuring its own network.

## Cost

One record at `--max-cycles 1` is ~62 Bedrock calls (30 main cells + 30
challenger cells + reformulation + mediation), plus 1–2 classifier calls.

| n | cycles | ≈ calls |
|---|---|---|
| 12 | 1 | ~770 |
| 20 | 1 | ~1,280 |
| 20 | 2 | ~2,500 |

`--workers` defaults to **1**: the cell agents already fan out 8 ways inside a
single scenario, and per-minute quota is the binding constraint. `--budget`
sets a hard per-scenario call cap that raises `BudgetExceeded` rather than
silently spending.

## Design notes

- **Frames are never inferred.** `datasets.read_frames` calls the project's
  `utils.prompts.read_cultural_frames`, which is why rows are snake_cased on
  load — NormAd ships `Country`, the project looks for `country`. A local
  mirror of `FRAME_FIELDS` is used only when the pipeline can't be imported
  (e.g. `inspect` with no boto3 installed); it's a deliberate copy so a
  divergence shows up as a diff.
- **NormAd has one country per row**, so the challenger gets no dataset frame
  and runs in neutral mode — which `prompts.py` already handles by dropping the
  frame block. `--normad-challenger "<frame>"` overrides that.
- **`--normad-context {none,country,rot,background}`** mirrors NormAd's three
  conditions. The country is consumed as a *frame*, not as prose, so the gap
  between `none` and `country` isolates the frame's contribution — which is the
  experiment the project is actually about.
- **Leakage is asserted, not assumed.** `assert_no_leakage` raises if NormAd's
  `Explanation` or MDD's decision column ends up in the scenario text. A leak
  wouldn't crash anything; it would quietly produce a great number.

## Two things to fix in the pipeline before running multi-cycle

1. `graph.node_integrate` assigns `mem.active_challenger_perspective = prose`.
   `SessionMemory` has no such field and pydantic v2 raises on assignment to an
   undeclared attribute. Unreachable at `--max-cycles 1`; quite likely on the
   first real cycle-2. Delete the line — the two after it already do the right
   thing via `challenger_working_perspective`.
2. `requirements.txt` doesn't list `datasets`. Add `datasets>=2.0` if you want
   NormAd pulled from the Hub instead of a local CSV.

## Extending to a third dataset

Add a loader returning `list[EvalRecord]`, a `FRAME_FIELDS` entry in the
project's `utils/prompts.py`, a gloss in `eval/prompts.py::GLOSSES`, and a
branch in `datasets.load_records`. Everything else — sampling, vocabulary
discovery, classifiers, metrics, calibration, reporting — is dataset-agnostic.

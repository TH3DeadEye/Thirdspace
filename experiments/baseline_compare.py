"""
Independent baseline harness — the CONTROL for ThirdSpace.

Purpose
-------
Measure how well a raw LLM handles a cultural-conflict scenario WITHOUT the
ThirdSpace pipeline (no 6x7 grid, no reformulation, no self-refinement, no
convergence loop). This is the control you compare ThirdSpace against to show
the pipeline actually lifts the model.

This file is deliberately self-contained: it imports NOTHING from thirdspace/.
Everything is raw boto3 `converse` calls. The only thing shared with the real
system is the JUDGE RUBRIC (a copy of judge_output_system / judge_user), because
a fair A/B requires the SAME measuring instrument on both outputs — but even the
judge here is its own independent boto3 call.

Conditions measured per model {Llama 3.1 8B, Haiku 4.5, Sonnet 4.5}:
  1. single_shot      — one prompt: "analyze this conflict" (the floor)
  2. self_opp_mediate — naive multi-agent: self frame -> opponent frame -> mediator
                        (a lightweight analog of ThirdSpace's main/challenger/mediation)

Then a fixed, blind judge (Sonnet) scores every output on the SAME five
dimensions ThirdSpace uses. Drop in a ThirdSpace output with --thirdspace-output
to score it through the identical judge and complete the comparison.

Run:
    python experiments/baseline_compare.py --runs 2
    python experiments/baseline_compare.py --runs 2 \
        --thirdspace-output outputs/<scenario_id>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("baseline")

REGION = "us-east-1"

# Bedrock model ids. These match the ids enabled in the ThirdSpace account.
# If Llama errors with ResourceNotFoundException, drop the "us." inference-profile
# prefix (-> "meta.llama3-1-8b-instruct-v1:0") or enable it in Bedrock model access.
MODELS: dict[str, str] = {
    "llama-3.1-8b": "us.meta.llama3-1-8b-instruct-v1:0",
    "haiku-4.5":    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "sonnet-4.5":   "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
}

# The judge is a FIXED strong model across all conditions. Never test a model
# with itself as judge for *its own* output only — keep one judge for everyone so
# the instrument is constant. Sonnet is the most capable id you have enabled.
JUDGE_MODEL = MODELS["sonnet-4.5"]

SCORE_KEYS = (
    "empathy", "consensus", "escalation_risk", "cultural_insight",
    "intervention_fidelity",
)


# ===========================================================================
# THE TEST SCENARIO  —  "The Repaid Dinner"
# Chosen because it is genuinely two-sided: both people act from *respect* by
# their own frame, and the injury lives in a single concrete gesture (the cash
# transfer), so a good analysis must locate the misread at the level of what
# "gratitude" and "doing the right thing" mean — not at the level of facts.
# Deliberately different from the Iran/Canada workplace-feedback example in main.py.
# ===========================================================================

SCENARIO = """\
I'm a German PhD student in Toronto. A few weeks ago a colleague in my lab, Karim,
who moved here from Jordan, invited me and three other students to his apartment for
dinner. He cooked for hours — there was far too much food, and he kept refusing every
time I offered to help or to bring something. I had a great evening. The next morning,
to do the right thing, I sent him 35 dollars by e-transfer for my share of the
groceries, plus a little extra as a thank-you, with a friendly message about how good
the meal was. He never acknowledged the transfer. When I saw him in the lab he was
polite but distant, and he has since turned down my last two invitations to grab lunch,
saying he's busy. Another colleague hinted that I "offended" him, which makes no sense
to me. I paid my fair share, I didn't freeload, and I even added extra to thank him.
I genuinely do not understand what I did wrong.
"""

# The two cultural frames, fixed and identical for the baseline AND for any
# ThirdSpace run you compare against, so frame choice is not a confound.
FRAME_SELF = (
    "German egalitarian / low-context culture: fairness means splitting costs and "
    "paying your own way; settling a debt promptly and transparently is the "
    "considerate, respectful thing to do, and it avoids burdening anyone or leaving "
    "an unequal obligation hanging between people."
)
FRAME_OPPONENT = (
    "Jordanian Arab hospitality culture (karam): hosting is an act of honor and "
    "relationship-building, not a service. The host's labor and his refusal of help "
    "ARE the gift. A guest who repays the host in exact cash converts a gift into a "
    "market transaction — implying the host is poor, is keeping score, or that the "
    "relationship is merely instrumental. Gratitude is shown by accepting graciously "
    "and reciprocating later in kind (a return invitation), never by paying it off."
)

# Answer key. Not shown to the generators. Used (optionally) to eyeball outputs
# and to drive the reference-augmented judge mode (--reference-judge).
REFERENCE_ANALYSIS = """\
A correct analysis should surface ALL of the following:
- The 35-dollar transfer was the INJURY, not a fix. The German read the dinner as a
  shared cost to be settled; under karam the dinner was a relational gift.
- Repaying in exact cash (and worse, adding a tip) reframes hospitality as a paid
  service and signals: I don't want to be in your debt / I don't want an ongoing
  reciprocal relationship / I doubt you could afford this. The "extra to thank him"
  makes it worse — tipping treats the host like staff.
- BOTH acted from respect. German respect = autonomy + not burdening others.
  Jordanian respect = generosity + accepting it graciously and reciprocating in kind.
- The disagreement is NOT about facts. It is about what counts as "doing the right
  thing" and how gratitude is properly expressed.
- Karim withdrew rather than confronting (declining invites, going distant) to save
  face for both — silence is the signal, consistent with a high-context honor frame.
- Escalation: moderate and rising. Left unrepaired it ossifies into permanent cool
  professional distance.
- A frame-aware repair cannot be purely verbal/transactional ("let me explain why I
  paid you"). The offense lives in the gesture, so repair must happen in the host's
  idiom: acknowledge in person that he now understands the dinner was a gift, and
  reciprocate with genuine hospitality (host Karim, treat it as a relationship not a
  ledger). "Just have an honest conversation" alone is INSUFFICIENT and can re-offend
  if framed as settling accounts.
A WEAK answer sides with the narrator ("you did nothing wrong, just communicate"),
gives generic "cultures differ about money," or prescribes "an open honest
conversation" as the whole solution.
"""


# ===========================================================================
# Raw Bedrock calls (boto3 converse — provider-agnostic: same call for Claude & Llama)
# ===========================================================================

def make_client() -> "boto3.client":
    cfg = Config(
        connect_timeout=10,
        read_timeout=120,
        retries={"max_attempts": 1, "mode": "standard"},  # we retry ourselves
    )
    return boto3.client("bedrock-runtime", region_name=REGION, config=cfg)


def converse(
    client,
    system: str,
    user: str,
    model_id: str,
    max_tokens: int = 2000,
    temperature: float = 0.1,
    retries: int = 2,
) -> str:
    """One Bedrock Converse turn. Works for Claude and Llama unchanged."""
    req: dict = {
        "modelId": model_id,
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    if system:
        req["system"] = [{"text": system}]

    for attempt in range(retries):
        try:
            resp = client.converse(**req)
            return resp["output"]["message"]["content"][0]["text"].strip()
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ThrottlingException", "TooManyRequestsException"):
                wait = 2 ** attempt * 5
                logger.warning(f"Throttled on {model_id}, waiting {wait}s "
                               f"(attempt {attempt+1}/{retries})")
                time.sleep(wait)
            elif code == "ResourceNotFoundException":
                raise RuntimeError(
                    f"Model '{model_id}' not enabled in {REGION}. Enable it in "
                    f"Bedrock -> Model access, or fix the id in MODELS."
                ) from e
            else:
                if attempt == retries - 1:
                    raise
                logger.warning(f"{code} on {model_id}, retrying: {e}")
                time.sleep(2 ** attempt)
    raise RuntimeError(f"converse failed after {retries} retries (model={model_id})")


def converse_json(client, system: str, user: str, model_id: str, **kw) -> dict:
    """Converse + tolerant JSON extraction. Only used for the judge (always Sonnet),
    so we never depend on a weak model producing clean JSON."""
    raw = converse(client, system, user, model_id, **kw)
    for block in _json_candidates(raw):
        try:
            return json.loads(block, strict=False)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Judge did not return JSON. Raw:\n{raw[:800]}")


def _json_candidates(text: str):
    """Yield most-promising JSON substrings first: fenced block, brace slice, raw."""
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    for f in fenced:
        yield f.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        yield text[start:end + 1]
    yield text.strip()


# ===========================================================================
# CONDITION 1 — single-shot baseline (the floor)
# ===========================================================================

SINGLE_SHOT_SYSTEM = (
    "You are an expert in cross-cultural communication. Analyze the cultural conflict "
    "the user describes. Be specific and even-handed: explain what each person most "
    "likely experienced, where their cultural assumptions diverged, what the real "
    "misunderstanding was, and concrete, realistic ways forward. Do not simply side "
    "with the narrator."
)


def run_single_shot(client, model_id: str) -> str:
    return converse(
        client, SINGLE_SHOT_SYSTEM,
        f"Analyze this cultural conflict:\n\n{SCENARIO}",
        model_id, max_tokens=2000, temperature=0.1,
    )


# ===========================================================================
# CONDITION 2 — naive multi-agent: self -> opponent -> mediator
# A lightweight analog of ThirdSpace (main agent / challenger / mediation) WITHOUT
# the grid, reformulation, refinement, or loop. This isolates how much of
# ThirdSpace's lift comes from its heavy scaffolding vs. just "use two frames".
# ===========================================================================

def _self_system(frame: str, other: str) -> str:
    return (
        f"You are reasoning about a cross-cultural conflict strictly from INSIDE one "
        f"cultural frame.\nYOUR FRAME: {frame}\nThe other party operates from: {other}\n\n"
        f"From inside your frame, explain: what happened, why your side's behavior was "
        f"reasonable and well-intentioned, what the other party probably got wrong, and "
        f"which of your own norms you are treating as obvious/universal that may not be. "
        f"Name those norms explicitly. Write 4 to 8 sentences."
    )


def _opponent_system(frame: str, other: str) -> str:
    return (
        f"You are reasoning about a cross-cultural conflict strictly from INSIDE one "
        f"cultural frame.\nYOUR FRAME: {frame}\nThe frame you are answering: {other}\n\n"
        f"You will be shown an analysis written from the opposing frame. Identify where "
        f"it treated an {other}-specific norm as if it were a universal human fact. From "
        f"YOUR frame, explain what actually happened, why your side's behavior was "
        f"reasonable, and name the SPECIFIC misread precisely. Write 4 to 8 sentences."
    )


MEDIATOR_SYSTEM = (
    "You are a neutral cross-cultural mediator. You are given a conflict scenario and "
    "two one-sided analyses written from opposing cultural frames. Produce a balanced "
    "analysis a learner can use. Cover, in order:\n"
    "1. What actually happened beneath both framings.\n"
    "2. Where the two cultural frames diverged, and why NEITHER side is simply wrong.\n"
    "3. What the conflict implies structurally (what layer the breakdown is really on).\n"
    "4. What each person most likely felt was happening — with empathy, not blame.\n"
    "5. Realistic ways forward. Be honest if 'just talk it out' will not work because "
    "the disagreement is itself about what respectful communication means.\n\n"
    "Ground every claim in the scenario and the two analyses. Do not invent facts."
)


def run_self_opp_mediate(client, model_id: str, blind_opponent: bool = False) -> dict:
    self_view = converse(
        client, _self_system(FRAME_SELF, FRAME_OPPONENT),
        f"SCENARIO:\n{SCENARIO}", model_id, max_tokens=900, temperature=0.4,
    )
    # ThirdSpace's challenger is now deliberately BLIND. The naive baseline default
    # is a DEBATE (opponent rebuts the self analysis) — that contrast is part of
    # what you're measuring. Flip --blind-opponent to mirror ThirdSpace instead.
    if blind_opponent:
        opp_user = f"SCENARIO:\n{SCENARIO}"
    else:
        opp_user = (
            f"SCENARIO:\n{SCENARIO}\n\n"
            f"ANALYSIS FROM THE OPPOSING ({FRAME_SELF[:40]}...) FRAME:\n{self_view}"
        )
    opp_view = converse(
        client, _opponent_system(FRAME_OPPONENT, FRAME_SELF),
        opp_user, model_id, max_tokens=900, temperature=0.4,
    )
    final = converse(
        client, MEDIATOR_SYSTEM,
        (f"SCENARIO:\n{SCENARIO}\n\n"
         f"ANALYSIS A (self frame — {FRAME_SELF[:40]}...):\n{self_view}\n\n"
         f"ANALYSIS B (opponent frame — {FRAME_OPPONENT[:40]}...):\n{opp_view}\n\n"
         f"Write the balanced five-part analysis now."),
        model_id, max_tokens=2500, temperature=0.5,
    )
    return {"self": self_view, "opponent": opp_view, "final": final}


# ===========================================================================
# JUDGE  —  identical rubric to thirdspace judge_output_system / judge_user.
# Blind: sees only the finished document. Fixed model (Sonnet) for every condition.
# ===========================================================================

JUDGE_SYSTEM = """\
You are a blind external reviewer reading a finished cultural conflict analysis
for the first time. You know NOTHING about how it was produced. Judge ONLY the
document in front of you, on its own terms.

Score FIVE dimensions 1-5. Reason step by step before scoring.
1=very poor  2=poor  3=adequate  4=good  5=excellent

empathy (1-5): Does the document reconstruct what BOTH parties experienced,
without blame or false balance?
consensus (1-5): Does it show clearly where the two frames genuinely meet and
where they remain apart?
escalation_risk (1-5): Does it convey how likely the conflict is to worsen
without intervention, and why?
cultural_insight (1-5): Does it surface assumptions neither party would see from
inside their own frame?
intervention_fidelity (1-5): Are the recommendations internally consistent with
the kinds of problem the document itself describes?
  1 — it recommends "an honest conversation" as the main fix for problems it
      describes as rooted in incompatible values or what counts as respect.
  3 — recommendations are generic, not matched to the described problems.
  5 — recommendations fit the described problems and are honest about what cannot
      be resolved by the parties talking alone.

Respond ONLY with valid JSON. No preamble. No markdown fences.
"""

# Optional stricter variant: judge also gets the answer key (REFERENCE_ANALYSIS).
# Use for a reference-grounded score; the blind score above is the headline metric.
JUDGE_SYSTEM_REF = JUDGE_SYSTEM + (
    "\nYou are ALSO given an expert reference of what a correct analysis must "
    "surface. Reward documents that independently reach those points; penalize "
    "ones that miss them or contradict them. Do not reward mere keyword overlap.\n"
)


def judge_document(client, document: str, reference: bool = False) -> dict:
    user = f"Evaluate this cultural conflict analysis document.\n\nDOCUMENT:\n{document}\n\n"
    if reference:
        user += f"EXPERT REFERENCE (answer key):\n{REFERENCE_ANALYSIS}\n\n"
    user += (
        'Return:\n{\n  "reasoning": "<step by step>",\n  "empathy": <1-5>,\n'
        '  "consensus": <1-5>,\n  "escalation_risk": <1-5>,\n  "cultural_insight": <1-5>,\n'
        '  "intervention_fidelity": <1-5>\n}'
    )
    result = converse_json(
        client, JUDGE_SYSTEM_REF if reference else JUDGE_SYSTEM, user,
        JUDGE_MODEL, max_tokens=1500, temperature=0.0,
    )
    return {k: int(result[k]) for k in SCORE_KEYS} | {"reasoning": result.get("reasoning", "")}


# ===========================================================================
# Experiment driver
# ===========================================================================

@dataclass
class RunResult:
    model: str
    condition: str
    run: int
    scores: dict
    document: str
    parts: dict = field(default_factory=dict)


def _avg_scores(runs: list[RunResult]) -> dict:
    out = {}
    for k in SCORE_KEYS:
        vals = [r.scores[k] for r in runs]
        out[k] = round(statistics.mean(vals), 2)
    out["total"] = round(sum(out[k] for k in SCORE_KEYS), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2,
                    help="repeats per (model, condition) — averages out judge/gen variance")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    choices=list(MODELS.keys()))
    ap.add_argument("--conditions", nargs="+", default=["single_shot", "self_opp_mediate"],
                    choices=["single_shot", "self_opp_mediate"])
    ap.add_argument("--blind-opponent", action="store_true",
                    help="make the opponent agent blind to the self analysis (mirrors ThirdSpace)")
    ap.add_argument("--reference-judge", action="store_true",
                    help="also score against the expert answer key")
    ap.add_argument("--thirdspace-output", type=str, default=None,
                    help="path to a ThirdSpace outputs/<id>.json (or a .txt of the final "
                         "document) to score with the SAME judge for direct comparison")
    args = ap.parse_args()

    client = make_client()
    all_runs: list[RunResult] = []

    for model_name in args.models:
        model_id = MODELS[model_name]
        for condition in args.conditions:
            for i in range(args.runs):
                logger.info(f"[{model_name}] {condition} run {i+1}/{args.runs}")
                try:
                    if condition == "single_shot":
                        doc = run_single_shot(client, model_id)
                        parts = {}
                    else:
                        out = run_self_opp_mediate(client, model_id, args.blind_opponent)
                        doc, parts = out["final"], {"self": out["self"], "opponent": out["opponent"]}
                    scores = judge_document(client, doc, reference=args.reference_judge)
                    all_runs.append(RunResult(model_name, condition, i, scores, doc, parts))
                except Exception as e:
                    logger.error(f"[{model_name}] {condition} run {i+1} FAILED: {e}")

    # Score a ThirdSpace output through the identical judge, if provided.
    ts_summary = None
    if args.thirdspace_output:
        doc = _load_thirdspace_document(Path(args.thirdspace_output))
        ts_runs = []
        for i in range(args.runs):
            logger.info(f"[thirdspace] judging run {i+1}/{args.runs}")
            ts_runs.append(RunResult(
                "thirdspace", "pipeline", i,
                judge_document(client, doc, reference=args.reference_judge), doc,
            ))
        all_runs.extend(ts_runs)
        ts_summary = _avg_scores(ts_runs)

    _report(all_runs, args)
    _save(all_runs, args)
    if ts_summary:
        logger.info(f"ThirdSpace avg total: {ts_summary['total']}/25")


def _load_thirdspace_document(path: Path) -> str:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        doc = data.get("final_output")
        if not doc:
            raise ValueError(f"{path} has no 'final_output' field")
        return doc
    return path.read_text(encoding="utf-8")


def _report(runs: list[RunResult], args) -> None:
    print("\n" + "=" * 78)
    print(f"RESULTS  (avg of {args.runs} run(s), judge={'reference' if args.reference_judge else 'blind'}, "
          f"opponent={'blind' if args.blind_opponent else 'debate'})")
    print("=" * 78)
    header = f"{'model':<16}{'condition':<20}" + "".join(f"{k[:5]:>7}" for k in SCORE_KEYS) + f"{'TOTAL':>8}"
    print(header)
    print("-" * len(header))
    groups: dict[tuple, list[RunResult]] = {}
    for r in runs:
        groups.setdefault((r.model, r.condition), []).append(r)
    for (model, cond), rs in groups.items():
        a = _avg_scores(rs)
        row = f"{model:<16}{cond:<20}" + "".join(f"{a[k]:>7}" for k in SCORE_KEYS) + f"{a['total']:>8}"
        print(row)
    print("-" * len(header))
    print("Headline metric: TOTAL out of 25. Compare each model's self_opp_mediate to")
    print("its single_shot, and the whole table to the ThirdSpace row.")


def _save(runs: list[RunResult], args) -> None:
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"baseline_{stamp}.json"
    path.write_text(json.dumps({
        "scenario": SCENARIO,
        "frames": {"self": FRAME_SELF, "opponent": FRAME_OPPONENT},
        "args": vars(args),
        "runs": [asdict(r) for r in runs],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved raw outputs + scores to {path}")


if __name__ == "__main__":
    main()

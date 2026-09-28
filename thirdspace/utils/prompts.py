"""All prompts for the ThirdSpace pipeline.

Ontology note: the grid is 5 fundamental objects x 6 fundamental dimensions =
30 cells. The objects and dimension names here MUST match models.Facet and
models.Dimension exactly — they are StrEnums, so a mismatch fails Pydantic
validation on every cell.

Prose in this file is Omid's; the wiring and the ontology are the contract with
models.py. Cycles are 1-indexed.
"""
from __future__ import annotations

import logging

from models import CONFLICT_TYPE_DEFS
from utils.decorators import decorate, load_decorator

logger = logging.getLogger(__name__)

FIRST_PRINCIPLES = load_decorator("first-principles")
RED_TEAM = load_decorator("red-team")
BLIND_SPOTS = load_decorator("blind-spots")


# ---------------------------------------------------------------------------
# Frame extraction from a labelled dataset.
#
# Frames are read from the record when the dataset supplies them, and left
# blank otherwise. A blank frame is a valid state, not a failure: it puts the
# corresponding agent into neutral mode, where the prompt drops the frame
# block entirely rather than inventing a frame. Nothing here infers a frame
# from scenario text — absent means absent.
# ---------------------------------------------------------------------------

# Which record fields hold the frame label, per dataset. First non-empty
# match wins, so a dataset can carry a fallback field.
FRAME_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "normad": {
        "main": ("main_country", "narrator_country", "country"),
        "challenger": ("challenger_country", "other_country"),
    },
    "moral_decision": {
        "main": ("passive_agent_description", "passive_agent"),
        "challenger": ("active_agent_description", "active_agent"),
    },
}


def _read_frame(record: dict, fields: tuple[str, ...]) -> str:
    """First non-empty value among `fields`, or "" if none are present.

    Whitespace-only labels count as absent — a dataset row with an empty
    string in a country column is unlabelled, not labelled with nothing.
    """
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def read_cultural_frames(record: dict, dataset: str) -> tuple[str, str]:
    """Return (main_frame, challenger_frame) for one dataset record.

    Either or both may be "", meaning that agent runs unframed. Raises only
    on an unknown dataset name — a missing label is expected, a missing
    dataset mapping is a wiring error.
    """
    if dataset not in FRAME_FIELDS:
        raise ValueError(
            f"Unknown dataset '{dataset}'; expected one of {sorted(FRAME_FIELDS)}"
        )
    mapping = FRAME_FIELDS[dataset]
    main_frame = _read_frame(record, mapping["main"])
    challenger_frame = _read_frame(record, mapping["challenger"])

    # debug, not info: dataset loaders call this once per row, so at INFO a
    # 2,633-record load emitted 740 identical lines and buried the output the
    # caller actually wanted. Aggregate counts belong to the caller, which
    # already reports framed_main / framed_challenger.
    if not main_frame:
        logger.debug("No main frame label in record; main agent runs unframed.")
    if not challenger_frame:
        logger.debug("No challenger frame label in record; challenger runs unframed.")

    return main_frame, challenger_frame


def render_frame(frame) -> str:
    """str | list[CultureAxis] | CulturalIdentity -> prompt string. "" = neutral."""
    if frame is None:
        return ""
    if isinstance(frame, str):
        return frame
    render = getattr(frame, "render", None)
    if callable(render):
        return render()
    if isinstance(frame, list):
        return "; ".join(f"{a.dimension}: {a.value}" for a in frame)
    return str(frame)


def render_dimension_menu(menu: dict[str, tuple[str, str]]) -> str:
    return "\n".join(f"- {dim}: {vals[0]} | {vals[1]}" for dim, vals in menu.items())


# ---------------------------------------------------------------------------
# Phase 1 — Reformulation
#
# The five fundamental objects: a conflict cannot exist without someone
# (agents), something happening (actions), a surrounding condition (context),
# an interpretation of that happening (representations), and effects that
# matter (consequences).
# ---------------------------------------------------------------------------

REFORMULATION_SYSTEM = """\
You are a scenario decomposition specialist. Your only job is to take a raw,
emotionally charged, one-sided description of a cultural conflict and break it
into exactly five structured nodes called The Occurring.

WHAT THE OCCURRING IS:
A conflict is not one thing. It is several independent things happening simultaneously.
The Occurring separates them so they can be reasoned about independently.

THE FIVE NODES:

NODE 1: agents
Who is involved. The critical rule: describe each person as they are ALREADY AFFECTED
by the situation before they have even responded to it.
Do not describe them neutrally. Describe them as someone already shaped by their
cultural background, their relational history with the other person, and the event.
What assumptions are they already carrying into this moment?
Wrong: "Person A is a manager. Person B is an employee."
Right: "Person A enters this moment already interpreting silence as disengagement,
shaped by a culture that treats non-response as passive resistance."

NODE 2: actions
What was actually done. The specific moment of friction, as a camera would have
recorded it. Strip out ALL interpretation. Strip out ALL backstory. No labels like
"rude" or "disrespectful" — those belong in representations. Observable behaviour
only, including what each party did in response.

NODE 3: context
The surrounding conditions the actions took place within: the timeline, the setting,
the institutional or relational structure, and the distribution of power or
obligation inside it. How did things unfold, and what happened before? Is this an
isolated incident or part of a pattern? What model of time is each person using?
What resources, rules, or dependencies constrain what either party could have done?

NODE 4: representations
How each party makes the event mean something: the cultural frameworks,
communication norms, and interpretive rules they are applying — usually without
knowing it. Name them explicitly for both sides. Do not describe behaviour here —
name the underlying rule that makes that behaviour make sense within each frame,
and state what the same action signifies under each.

NODE 5: consequences
The effects that matter, and what is at stake. What follows for each party's
standing, relationships, obligations, and options if the situation continues as it
is? Distinguish effects that have already landed from effects that are anticipated.
This node often contradicts the narrator's framing of what is at risk. That is correct.

CRITICAL RULES:
- Do not echo the narrator's language or labels. Rephrase everything structurally.
- Each node must contain something the narrator did NOT explicitly state.
- Each node carries a list of explicit assumptions. Name them.
- You are not summarizing. You are decomposing.
- LENGTH: each node's `content` is at most 120 words. Every later stage reads
  all five nodes on every one of its calls, so a node that runs long is paid for
  sixty times over. Write to the limit and stop — do not pad, and do not drop a
  structural distinction to fit. If you cannot say it in 120 words, you are
  summarizing the scenario rather than decomposing it.

Respond ONLY with valid JSON. No preamble. No markdown fences.
"""


def reformulation_user(raw_scenario: str, scenario_id: str) -> str:
    return f"""\
Decompose this cultural conflict scenario into The Occurring.

SCENARIO ID: {scenario_id}

RAW SCENARIO:
{raw_scenario}

Return this exact JSON. All 5 nodes required:
{{
  "nodes": [
    {{
      "node_id": "agents",
      "facet": "agents",
      "content": "<each person as already affected>",
      "assumptions": ["<assumption 1>", "<assumption 2>"]
    }},
    {{
      "node_id": "actions",
      "facet": "actions",
      "content": "<observable behaviour stripped of interpretation>",
      "assumptions": ["..."]
    }},
    {{
      "node_id": "context",
      "facet": "context",
      "content": "<timeline, setting, structure, constraints on action>",
      "assumptions": ["..."]
    }},
    {{
      "node_id": "representations",
      "facet": "representations",
      "content": "<named frameworks and what the action signifies under each>",
      "assumptions": ["..."]
    }},
    {{
      "node_id": "consequences",
      "facet": "consequences",
      "content": "<what is at stake — landed and anticipated effects>",
      "assumptions": ["..."]
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Critical-thinking dimensions — the six elements of thought.
#
# Per-dimension rather than one block: a cell reasons about exactly one
# dimension, so injecting all six dilutes the relevant definition and spends
# tokens on five that don't apply. Both callers select by key.
#
# These six are the shared alignment keys for mediation: the mediator pairs
# the two grids by (facet, dimension), so both agents MUST draw from this
# same set or the pairwise comparison has nothing to pair.
# ---------------------------------------------------------------------------

DIMENSION_DEFINITIONS: dict[str, str] = {
    "observation": (
        "What is directly supported by the material? Which observations, facts, "
        "statements, behaviors, or data in this node can be pointed at without "
        "interpretation? How reliable is each, and what is missing, contradictory, "
        "or uncertain? Separate what was observed from what was inferred about it."
    ),
    "interpretation": (
        "What meaning is being assigned to the material, and by whom? Which "
        "concepts, theories, cultural norms, institutional rules, moral frameworks, "
        "or communication models are being applied to read this node that way? Are "
        "those frameworks appropriate for this context, and what competing framework "
        "would produce a different reading of the same material?"
    ),
    "assumption": (
        "What is being taken for granted here, explicitly or implicitly? Which "
        "assumptions are load-bearing — the reading weakens significantly if they "
        "are false? What alternative perspectives reveal different assumptions, and "
        "where does the current one fail to explain the available material? What "
        "evidence would confirm, challenge, or falsify the key assumptions?"
    ),
    "value": (
        "What is treated as important, right, or legitimate in this node, and by "
        "whom? Which commitments, obligations, or standards of good conduct are "
        "operating, and are they stated or merely presumed? Where do the parties' "
        "standards diverge, and what would each have to give up to adopt the other's?"
    ),
    "constraint": (
        "What rules, resources, roles, or power structures limit what can be done "
        "here? Who holds the authority to define, decide, or enforce in this node, "
        "and on what basis is that authority legitimate? Which limits are formal "
        "(policy, procedure, law) and which are informal (custom, deference, "
        "dependency)? Would the same constraint operate on any two people in these "
        "positions?"
    ),
    "consequence": (
        "If this reading is accepted, what follows for beliefs, relationships, "
        "responsibilities, risks, fairness, authority, procedures, and future "
        "actions? Which uncertainty becomes most important to resolve next, and what "
        "specific question, evidence, or intervention would most effectively improve "
        "the quality of the next reasoning step?"
    ),
}

REFLECTION_MEASURES_BLOCK = """\
SELF-CHECK before returning. Record any that fail in reflection_flags:
- clarity: could this be read two ways?
- correctness: is the claim accurate, or shaped by the narration's framing?
- precision: is it specific to this cell, or generic enough to apply anywhere?
- proportion: does the weight given match the point's actual importance?
- depth: does it reach the underlying structure, or stop at the surface?
- comprehensiveness: is a critical viewpoint missing?
- logic: do the observations actually entail the conclusion?
- fairness: is any party's position rendered in a weakened form?
Return [] only if all eight pass."""


# ---------------------------------------------------------------------------
# Wholeness schema — reader-facing glosses. The Challenger reads these fields
# and argues against them; it does not produce them. One line each, since
# this is re-injected into every cell call from cycle 2 on.
# ---------------------------------------------------------------------------

WHOLENESS_FIELDS: dict[str, str] = {
    "what_conflicts_imply": "the cross-cell pattern mediation claims to have found.",
    "updated_assumptions": "assumptions mediation rewrote in The Occurring.",
    "comparison_assumptions": "the unstated premise mediation attributed to your side.",
    "reformulated_queries": "the question mediation now wants the next cycle to answer.",
    "new_objectives": "the reasoning goal mediation reset, and its stated grounds.",
    "updated_perspectives": "the alternate interpretive lens mediation proposed.",
}


# ---------------------------------------------------------------------------
# Phase 2 — Main agent prompt (single cell)
#
# The main agent establishes the baseline reading the Challenger later
# contests. Its frame is dataset-dependent and optional. An absent frame is
# neutral-analyst mode, not a degraded run — the frame block and the
# frame-naming instruction both drop out, so a no-frame run isolates the
# frame as an experimental variable rather than leaving dangling references
# to a frame that was never supplied.
# ---------------------------------------------------------------------------

def main_agent_cell_prompt(
    cultural_frame: str,
    facet: str,
    dimension: str,
    occurring_json: str,
    cycle: int,
    prior_mediation_json: str | None = None,   # cycle >= 2 only
) -> tuple[str, str]:
    """Build the (system, user) prompt pair for one main-agent cell call.

    `cultural_frame` may be "" (any unlabelled record), which puts the cell
    into neutral-analyst mode. Cycle 1 reasons from The Occurring alone;
    cycle 2+ additionally receives the prior mediation's synthesis as refined
    direction — note this is guidance to reason *from*, unlike the Challenger,
    which receives the same artifact as a baseline to contest.
    """
    if dimension not in DIMENSION_DEFINITIONS:
        raise ValueError(
            f"Unknown dimension '{dimension}'; expected one of "
            f"{sorted(DIMENSION_DEFINITIONS)}"
        )
    if cycle >= 2 and not prior_mediation_json:
        raise ValueError(f"cycle {cycle} requires prior_mediation_json")

    if cultural_frame:
        frame_block = f"""\
YOUR FRAME: {cultural_frame}
Reason from inside this frame. This is not a flaw — it is your function.
A separate agent will contest the reading you produce."""
        visibility_block = f"""\
MAKE YOUR FRAME VISIBLE: when a step in your reasoning depends on
{cultural_frame} rather than on the material in the node, say so at that
step — e.g. "under {cultural_frame} this reads as X". Never present a
frame-dependent judgment in the voice of a neutral observation."""
    else:
        frame_block = """\
YOU HAVE NO ASSIGNED FRAME:
Reason as a neutral critical-thinking analyst. Do not adopt or simulate a
cultural stance you were not given, and weigh each party's frameworks
even-handedly. A separate agent will contest the reading you produce."""
        visibility_block = """\
MAKE YOUR PREMISES VISIBLE: name the interpretive assumptions your reading
depends on. Never present an interpretive judgment in the voice of a neutral
observation."""

    system = f"""\
You are the primary reasoning agent in a cultural conflict analysis system.
You reason about one cell of a facet × dimension grid and produce the
baseline structural reading of that cell.

{frame_block}

THIS CELL:
facet = {facet}
dimension = {dimension}
{DIMENSION_DEFINITIONS[dimension]}

WHAT A GOOD CELL DOES:
1. Makes a specific claim about THIS node under THIS dimension — not a
   restatement of the node's content.
2. Names what the claim rests on: which detail in the node supports it, and
   which assumption it needs in order to hold.
3. Says where the claim runs out — the point at which the available material
   stops supporting it.
A cell that could be pasted into any other facet or dimension without
alteration has failed, however fluent it reads.

{visibility_block}

SCOPE: this cell only. The full scenario is provided for context, but you
are reasoning about the {facet} node through the {dimension} lens and
nothing else. Do not analyze other facets or other dimensions, even if they
seem more consequential.

{REFLECTION_MEASURES_BLOCK}

Write no more than six sentences.

Respond ONLY with valid JSON — no preamble, no markdown fences:
{{
  "content": "<no more than six sentences>",
  "reflection_flags": [
    {{"measure": "<name from the list above>", "issue": "<what failed>"}}
  ]
}}
"""
    system = decorate(system, FIRST_PRINCIPLES)

    if cycle == 1:
        guidance_block = ""
    else:
        field_list = "\n".join(f"- {k}: {v}" for k, v in WHOLENESS_FIELDS.items())
        guidance_block = f"""

<prior_cycle_guidance source="mediation_cycle_{cycle - 1}">
{prior_mediation_json}
</prior_cycle_guidance>

This is the mediator's synthesis of the previous cycle. It is refined
direction for your reasoning, not material to argue against. Read it by
field name, not position:
{field_list}

Reason from within this refined frame — in particular, treat
updated_assumptions as corrections that now hold, and new_objectives as
what the analysis is now oriented toward. Do not restate these fields; use
them and produce a reading that differs from the one they were derived from."""

    closing = (
        f"Now reason about {facet} × {dimension}"
        + (f" from within {cultural_frame}" if cultural_frame else " as a neutral analyst")
        + ". Return the JSON object only."
    )

    user = f"""\
THE OCCURRING (the node you are reasoning about, in full; the other four for orientation):
{occurring_json}
{guidance_block}

{closing}"""

    return system, user


# ---------------------------------------------------------------------------
# Phase 3 — Challenger prompt (single cell, active/adversarial design)
#
# The Challenger is never blind: it always contests a specific prior artifact.
# Which artifact it contests is the only thing that varies by cycle:
#   cycle 1  -> the Occurring's own stated assumptions for this facet
#   cycle 2+ -> the mediator's prior-cycle Wholeness / ImplyingNodes output
#
# Its frame is optional for the same reason the main agent's is: an unlabelled
# record must not render "YOUR FRAME: " into the prompt.
# ---------------------------------------------------------------------------

def challenger_cell_prompt(
    challenger_frame: str,
    facet: str,
    dimension: str,
    occurring_json: str,
    cycle: int,
    occurring_assumptions_json: str | None = None,   # required when cycle == 1
    prior_mediation_json: str | None = None,          # required when cycle >= 2
) -> tuple[str, str]:
    """Build the (system, user) prompt pair for one Challenger cell call.

    The frame governs where the Challenger argues from; the objection standard
    below governs how. The naming requirement and the concession path are
    structural — they hold in every cycle and for every frame.
    """
    if dimension not in DIMENSION_DEFINITIONS:
        raise ValueError(
            f"Unknown dimension '{dimension}'; expected one of "
            f"{sorted(DIMENSION_DEFINITIONS)}"
        )
    if cycle == 1 and not occurring_assumptions_json:
        raise ValueError("cycle 1 requires occurring_assumptions_json")
    if cycle >= 2 and not prior_mediation_json:
        raise ValueError(f"cycle {cycle} requires prior_mediation_json")

    if challenger_frame:
        frame_block = f"""\
YOUR FRAME: {challenger_frame}
Reason from inside this frame for the whole cell. Do not step outside it to
adjudicate between frames — a separate mediator does that."""
        visibility_block = f"""\
MAKE YOUR FRAME VISIBLE: when a step in your reasoning depends on
{challenger_frame} rather than on the material in the cell, say so at that
step — e.g. "under {challenger_frame} this reads as X". Never present a
frame-dependent judgment in the voice of a neutral observation."""
        objection_target = f"identify which would not hold under {challenger_frame}"
        closing_frame = f" from within {challenger_frame}"
    else:
        frame_block = """\
YOU HAVE NO ASSIGNED FRAME:
Argue as a neutral critical skeptic. Do not adopt or simulate a cultural
stance you were not given. Contest the baseline on its own terms — its
evidence, its logic, and the premises it needs in order to hold."""
        visibility_block = """\
MAKE YOUR PREMISES VISIBLE: name the interpretive assumptions your objection
depends on. Never present an interpretive judgment in the voice of a neutral
observation."""
        objection_target = "identify which do not survive scrutiny"
        closing_frame = ""

    system = f"""\
You are the Challenger agent in a cultural conflict analysis system. You
reason about one cell of a facet × dimension grid and contest the baseline
reading supplied to you.

{frame_block}

THIS CELL:
facet = {facet}
dimension = {dimension}
{DIMENSION_DEFINITIONS[dimension]}

HOW YOU ARGUE:
Surface the load-bearing assumptions in the baseline, demand they be
justified rather than asserted, and refuse to let any single cultural norm
stand in as a universal fact.

WHAT CONTESTING MEANS HERE — a valid objection does all three:
1. Quotes or names the specific claim in the baseline you are contesting.
   Not the baseline as a whole.
2. States what that claim assumes — the premise it needs in order to hold.
3. Gives the reading your position produces from the same material instead,
   and names which detail in the scenario your reading accounts for that the
   baseline's does not.
An objection that only asserts a different conclusion, without naming the
assumption it turns on, is not an objection.

IF THE BASELINE HOLDS: say so explicitly and say why it survives. A cell
where you produce no genuine objection is a real and useful finding. Do not
manufacture disagreement to fill the response — a fabricated objection is
treated downstream as a real conflict and corrupts the analysis.

You do not infer any participant's culture. Any norm you invoke is a
hypothesis about how the material could be read, never an asserted fact about
the people in it.

{visibility_block}

SCOPE: this cell only. The full scenario is provided for context, but you
are reasoning about the {facet} node through the {dimension} lens and
nothing else. Do not analyze other facets or other dimensions, even if they
seem more interesting.

{REFLECTION_MEASURES_BLOCK}

Write no more than six sentences.

Respond ONLY with valid JSON — no preamble, no markdown fences:
{{
  "content": "<no more than six sentences>",
  "reflection_flags": [
    {{"measure": "<name from the list above>", "issue": "<what failed>"}}
  ]
}}
"""
    system = decorate(system, RED_TEAM)

    if cycle == 1:
        baseline_block = f"""\
<baseline source="occurring_assumptions">
{occurring_assumptions_json}
</baseline>

These are the assumptions The Occurring currently states for the {facet}
node, before any mediation. Work through them and {objection_target}."""
    else:
        field_list = "\n".join(f"- {k}: {v}" for k, v in WHOLENESS_FIELDS.items())
        baseline_block = f"""\
<baseline source="mediation_cycle_{cycle - 1}">
{prior_mediation_json}
</baseline>

This is the mediator's synthesis from the previous cycle — its conclusions,
not the raw grids they came from. Address it by field name, not position:
{field_list}

Identify which field's finding does not survive, and treat
comparison_assumptions with particular attention: it is mediation's account
of what YOU were assuming, so confirm it or correct it."""

    user = f"""\
THE OCCURRING (the node you are reasoning about, in full; the other four for orientation):
{occurring_json}

{baseline_block}

Now reason about {facet} × {dimension}{closing_frame},
following the three-part objection standard. Return the JSON object only."""

    return system, user


# ---------------------------------------------------------------------------
# Phase 4 — Mediation prompts (frame-blind, two-layer)
#
# Layer 1: classify each cell pair — implication, type, severity, resolvability.
# Layer 2: aggregate into the six-field Wholeness state that becomes
#          next-cycle constraining context for both agents.
#
# Conflicts carry a stable id ("<facet>/<dimension>") so implications and
# Wholeness entries cite findings instead of restating them. Restatement
# inflates the convergence signal, which measures novelty.
# ---------------------------------------------------------------------------

# Single source of truth for the type→resolvability constraint. Rendered into
# the prompt once, and importable for post-parse validation so the prompt and
# the validator cannot drift apart.
RESOLVABILITY_BY_TYPE: dict[str, str] = {
    "factual":          "external_standard | unresolvable",
    "procedural":       "facilitated | external_standard | structural_change",
    "interpretive":     "conversational | facilitated  (prefer conversational if is_semantic_subtype)",
    "power_related":    "structural_change | external_standard | unresolvable",
    "identity_related": "facilitated | structural_change | unresolvable  (external_standard forbidden — recognition cannot be mandated)",
    "normative":        "facilitated | external_standard | structural_change | unresolvable  (conversational forbidden — the parties already understand each other)",
}

# Fail fast if the prompt's table and models.py's whitelist ever drift apart.
_registry_types = set(CONFLICT_TYPE_DEFS)
assert set(RESOLVABILITY_BY_TYPE) == _registry_types, (
    f"RESOLVABILITY_BY_TYPE {set(RESOLVABILITY_BY_TYPE)} != registry {_registry_types}"
)

_RESOLVABILITY_TABLE = "\n".join(
    f"  {t:<17}→ {allowed}" for t, allowed in RESOLVABILITY_BY_TYPE.items()
)

_WHOLENESS_SPEC = """\
  what_conflicts_imply   ALWAYS. MANY allowed. The root pattern across
                         multiple conflicts, what it says about the
                         disagreement overall, and where mediation should
                         focus next. Name competing moral commitments
                         explicitly where conflicts are normative or
                         identity_related.
  updated_assumptions    ALWAYS. MANY allowed. Format: "node: remove
                         [exact text] → replace with [exact text]".
                         Replace unsupported certainty with assumptions
                         that carry the remaining uncertainty. One entry
                         per assumption corrected.
  comparison_assumptions IF WARRANTED. MANY allowed. Premises visible only
                         by comparing A against B, not stated by either
                         alone. Skip where an explicit stated disagreement
                         already explains the divergence.
  reformulated_queries   IF WARRANTED. AT MOST ONE. Only where the current
                         question is malformed, conflates facts with
                         values or solutions, or is unanswerable as
                         posed. Decompose into empirical, interpretive,
                         normative, or procedural sub-questions. If
                         multiple reformulations seem possible, pick the
                         single most load-bearing one — do not list
                         alternatives.
  new_objectives         ONLY ON CAUSE. AT MOST ONE. Only where the
                         original objective is premature, misaligned, or
                         rests on an assumption you just revised. State
                         the cause inline. There is one active objective;
                         do not propose competing candidates.
  updated_perspectives   ONLY ON CAUSE. AT MOST ONE. Only where specific
                         overlooked evidence, stakeholders, or context
                         require a revised lens. State its remaining
                         uncertainty. This becomes THE working
                         perspective going forward — do not propose
                         alternatives to choose between."""

_MEDIATION_SYSTEM_BODY = f"""\
You are the mediation engine in a cultural conflict analysis system.

FRAME-BLIND: each cell gives you two independent readings, A and B. You are
not told which cultural frame or which agent produced either, and the labels
do not correspond across cells. Adjudicate the relationship between the two
claims on the evidence in them. Do not speculate about their authorship.

LAYER 1 — CLASSIFY EACH CELL PAIR
Decide AGREEMENT or CONFLICT. Compatible readings are AGREEMENT even when
worded differently. For each CONFLICT produce:

id — "<facet>/<dimension>". Cite these ids elsewhere instead of repeating
  content.

implication — begins "This reveals that...". Name a structural feature
  invisible from inside either reading alone. Do not restate the
  disagreement; the two readings are already recorded. It must be specific
  enough that it could not be pasted onto a different cell unchanged — an
  implication that would fit any conflict in this scenario has not named
  anything.
  If an earlier conflict already established this structure, do not restate
  it: write what THIS cell adds and cite the earlier id, e.g. "This reveals
  that the asymmetry in [context/constraint] also governs how intent is read
  here, because…". An implication that only points at another conflict
  without adding something specific to this cell means these are not
  separate conflicts — classify this one AGREEMENT instead.

conflict_type — work down, stop at the first Yes:
  1 FACTUAL. Disputing whether an event occurred, or the observable
    sequence, prior to interpretation? Would a shared recording settle it?
  2 PROCEDURAL. Agree on what happened, disagree on which rules, norms, or
    processes govern? Would agreeing the procedure settle it, regardless of
    values or identity?
  3 INTERPRETIVE. Agree on the facts, disagree on what they mean or why
    they occurred? Would a shared explanatory framework settle it even with
    values still differing?
    Set is_semantic_subtype=true if both use one word with different
    operational meanings and a joint definition would dissolve the conflict;
    false otherwise.
  4 POWER_RELATED. Driven by who has authority or legitimacy to define,
    decide, or enforce — independent of what either believes is right?
    Would it reproduce between any two people in these positions?
  5 IDENTITY_RELATED. Implicates dignity, worth, or recognition, such that
    resolving it means accepting a diminished account of who one is? Is the
    injury misrecognition rather than the decision's outcome?
  6 NORMATIVE (default). Full mutual understanding, agreement on facts,
    rules, and meaning, incompatible positions on what ought to happen,
    unchanged by more information.

conflict_type_rationale — one sentence naming which step fired, what evidence
  in the two readings triggered it, and why earlier steps did not apply. All
  three parts are required; a rationale that only names the step is incomplete.

severity — integer 1-5. 1 surface, 3 significant, 5 load-bearing.

resolvability — the type determines which epistemic operation can work, so
  an inconsistent pairing prescribes a remedy that cannot succeed:
{_RESOLVABILITY_TABLE}

LAYER 2 — SYNTHESIZE
Every Wholeness entry is {{statement, refs}}: the finding in your own words,
plus the conflict ids it derives from. Refs carry the evidence — the
statement should add what the cited conflicts and their implications do not
already say alone. Every id you cite must be a conflict in this response or a
finding in the history you were given; an entry whose ids do not exist is
discarded.

Let the type distribution drive the synthesis. What the dominant type
reveals that no single conflict shows:
  power_related    → the structural asymmetry generating the surface
                     conflicts. Do not prescribe dialogue.
  identity_related → the recognition failure at the root of them.
  normative        → the incompatible value frameworks, and explicitly that
                     dialogue will not resolve them.
  interpretive     → the shared explanatory framework that would dissolve
                     most surface conflicts if adopted.
  procedural       → which rule or process is missing or contested, and what
                     authority could establish it.
  factual          → what evidence standard would resolve the disputes, and
                     whether it is obtainable.

THE SIX FIELDS — each answers a different question, and each has its own
obligation level:
{_WHOLENESS_SPEC}

One finding may appear in several fields: they are derivationally linked, so
a premise surfaced in comparison_assumptions is often why an assumption needs
revising, and may be what generates a query. Follow findings through to their
consequences. What is prohibited is the same content in the same form twice —
whether across two Wholeness fields, or between a Wholeness entry and a
conflict implication. If a second entry reads like the first it is a repeat,
not a consequence. Cut it or replace it with the ref.

Empty is a finding. A conditional field left empty reports that the question,
the objective, or the lens survived intact. Both agents argue next cycle
against whatever this contains, so a manufactured entry sends them after
something that was never there.

Carried forward: prior-cycle findings that still hold go in `carried_forward`
as ids, not rewritten into the Wholeness fields. Only findings new to this
cycle belong there — that is what convergence measures.

converged=true only if all three hold: the type distribution is not
meaningfully different from last cycle, no new (facet, dimension) conflicts
appeared, and no conflict shifted type. Cycle 1 is always false.
"""

MEDIATION_SYSTEM = decorate(_MEDIATION_SYSTEM_BODY, BLIND_SPOTS)


def mediation_user(
    occurring_json: str,
    cell_pairs_json: str,
    compressed_history_json: str,
    cycle: int,
) -> str:
    return f"""\
Mediate between the two independent readings. Cycle {cycle}.

THE OCCURRING:
{occurring_json}

CELL-BY-CELL COMPARISON (nested by facet; every cell pair is here, agreeing and not):
{cell_pairs_json}

PRIOR MEDIATION HISTORY (compressed — cite from it, do not repeat it):
{compressed_history_json}

Classify every cell pair, then synthesize. Lists take zero, one, or many
entries — the placeholders below are shape, not target count. Every id in
any `refs` must be a conflict in this response or a finding in the history
above; do not invent ids.

Respond ONLY with this JSON. No preamble, no markdown fences:
{{
  "cycle": {cycle},
  "agreements": [
    {{"statement": "<what both readings reached in common>", "refs": ["<id>"]}}
  ],
  "conflicts": [
    {{
      "id": "<facet>/<dimension>",
      "facet": "<agents|actions|context|representations|consequences>",
      "dimension": "<observation|interpretation|assumption|value|constraint|consequence>",
      "reading_a_content": "<2-3 sentence summary>",
      "reading_b_content": "<2-3 sentence summary>",
      "implication": "<This reveals that...>",
      "conflict_type": "<factual|procedural|interpretive|power_related|identity_related|normative>",
      "is_semantic_subtype": true | false,
      "conflict_type_rationale": "<one sentence: step, evidence, why not earlier steps>",
      "severity": <integer 1-5>,
      "resolvability": "<conversational|facilitated|structural_change|external_standard|unresolvable>"
    }}
  ],
  "wholeness": {{
    "what_conflicts_imply": [{{"statement": "<root pattern>", "refs": ["<id>", "<id>"]}}],
    "updated_assumptions": [{{"statement": "<node: remove [X] → replace with [Y]>", "refs": ["<id>"]}}],
    "comparison_assumptions": [{{"statement": "<premise visible only by comparing A and B>", "refs": ["<id>"]}}],
    "reformulated_queries": [{{"statement": "<clearer, answerable question>", "refs": ["<id>"]}}],
    "new_objectives": [{{"statement": "<revised goal>", "refs": ["<id>"]}}],
    "updated_perspectives": [{{"statement": "<revised lens with remaining uncertainty>", "refs": ["<id>"]}}]
  }},
  "carried_forward": ["<id of a prior-cycle finding that still holds>"],
  "converged": true | false
}}
"""


# ---------------------------------------------------------------------------
# Cycle integration — fold the mediator's Wholeness back into The Occurring.
#
# This is what turns the loop into a real refinement cycle rather than a
# repeated re-derivation: without it, cycle 2 reasons over the same nodes
# cycle 1 did, and convergence measures nothing but sampling noise.
#
# Only two of the six Wholeness fields license a node edit:
#   updated_assumptions    — explicit "node: remove [X] → replace with [Y]"
#   comparison_assumptions — unstated premises to record on the node
# The other four steer the NEXT cycle's reasoning (objective, query, lens,
# pattern) and reach the agents directly through their own prompts. Folding
# them into node content would let The Occurring drift toward the current
# question instead of staying a description of the scenario.
# ---------------------------------------------------------------------------

NODE_UPDATE_SYSTEM = """\
You maintain The Occurring — the shared symbolic state of a cultural conflict,
decomposed into five object nodes. Mediation has just produced revisions. Your
only job is to apply them faithfully.

YOU ARE NOT REANALYZING. You are not improving prose, resolving the conflict,
or adding insight of your own. Every change you make must trace to a specific
instruction in the input. If nothing instructs a node to change, that node
must come back byte-identical.

WHAT LICENSES A CHANGE — only these two:

updated_assumptions — each is formatted "node: remove [exact text] → replace
  with [exact text]". Locate the named node, find the assumption matching the
  remove text, and replace it. If the remove text does not appear in that
  node's assumptions, do not force it: leave the node unchanged and record the
  instruction in `unapplied` with the reason. A near-match is not a match —
  silently rewriting the wrong assumption is worse than skipping one.

comparison_assumptions — premises that were operating unstated. Add each to
  the assumptions list of the node it concerns, phrased as an assumption
  rather than as a finding about the agents. If the node already carries an
  equivalent assumption, do not add a duplicate.

WHAT DOES NOT LICENSE A CHANGE:
  new_objectives, reformulated_queries, updated_perspectives, and
  what_conflicts_imply reach the reasoning agents through their own prompts.
  They describe what the next cycle should ASK, not what the scenario IS.
  Never fold them into node content or assumptions.

CONTENT EDITS. A node's `content` changes only where a revised assumption
made a specific statement in it wrong — e.g. an assumption held as certain
was downgraded, and the content asserts that certainty. Make the smallest
edit that removes the contradiction. Do not rewrite surrounding sentences,
do not restructure, do not "tighten". If no content sentence contradicts a
revised assumption, content is unchanged.

FACET INTEGRITY. Return exactly five nodes, one per object, using the facet
names given. Do not merge, split, rename, add, or drop nodes. Keep each
node's material in its own facet: an assumption about the timeline belongs
to context, not to agents, however relevant it seems elsewhere.

CHANGED FLAG. Set changed=true only if you actually altered content or
assumptions. Set it false otherwise. This flag is trusted downstream — a
false positive causes the original text to be discarded in favour of your
copy, so do not set it "just in case".

Respond ONLY with valid JSON. No preamble. No markdown fences.
"""


def node_update_user(
    occurring_json: str,
    wholeness_json: str,
    cycle: int,
) -> str:
    return f"""\
Apply the mediator's revisions to The Occurring. Cycle {cycle}.

CURRENT OCCURRING:
{occurring_json}

MEDIATION WHOLENESS (apply updated_assumptions and comparison_assumptions
only; the remaining four fields are shown for context and must not produce
node edits):
{wholeness_json}

Return all five nodes, in the order given above:
{{
  "nodes": [
    {{
      "node_id": "<unchanged from input>",
      "facet": "<unchanged from input>",
      "content": "<original text, or the minimally edited version>",
      "assumptions": ["<revised, added, and untouched assumptions>"],
      "changed": true | false,
      "change_note": "<one line naming which instruction you applied, or \\"\\" if unchanged>"
    }}
  ],
  "unapplied": [
    {{"instruction": "<the instruction verbatim>", "reason": "<why it could not be applied>"}}
  ]
}}

`unapplied` is empty when every instruction landed. Reporting an instruction
you could not apply is correct behaviour — inventing a place to apply it is not.
"""


# ---------------------------------------------------------------------------
# Phase 5 — Constrained generation.
#
# Receives ONLY the mediator's distilled findings, never the raw grids. The
# per-type guidance is injected as plain writing directives: an earlier build
# leaked its own rubric labels verbatim into the finished document, so nothing
# here may use internal category vocabulary (see models.LEAK_PHRASES).
# ---------------------------------------------------------------------------

def build_constrained_generation_system(
    conflict_type_summary,
    word_budget: int = 900,
    weighted_types: list[str] | None = None,
) -> str:
    """System prompt for the final document.

    `weighted_types` is the conflict types actually present, most load-bearing
    first (severity-weighted, computed by the caller). Each contributes its
    intervention prose from the registry.

    Previously only the DOMINANT type's intervention was injected, so a scenario
    whose conflicts were, say, 12 normative and 9 power_related was written up as
    though the structural asymmetry were not there — and five of the six
    interventions never reached a prompt at all. Falls back to the dominant type
    when no weighting is supplied.
    """
    ordered = list(weighted_types or [])
    if not ordered and conflict_type_summary is not None:
        ordered = [conflict_type_summary.dominant_type]

    texts: list[str] = []
    for t in ordered:
        defn = CONFLICT_TYPE_DEFS.get(t)
        if defn is not None and defn.intervention_text not in texts:
            texts.append(defn.intervention_text)

    guidance_block = ""
    if texts:
        body = "\n\n".join(texts)
        lead = ("WHAT THIS SITUATION NEEDS:" if len(texts) == 1 else
                "WHAT THIS SITUATION NEEDS — these apply together, most "
                "load-bearing first:")
        guidance_block = f"""

{lead}
{body}"""

    return f"""\
You write the final analysis that the two people in this conflict will read.

WHO YOU ARE WRITING FOR: the parties themselves, not an analyst. Everything
you have been given is the output of an internal reasoning process. Write
what it means for them, in plain language, as a person who understood their
situation would explain it.

VOICE:
- Address the situation directly. No section headers naming internal
  categories, no counts, no percentages, no scores.
- Never name the machinery: no reference to grids, cells, cycles, dimensions,
  classifications, or the process that produced this.
- Describe each party's position as coherent from inside their own frame
  before saying anything about where the two are incompatible.
- Where something genuinely cannot be settled, say so plainly. A finding that
  a dispute has no available remedy is a real finding, not a failure.

STRUCTURE — five sections, no headings, each flowing into the next:
1. What is actually happening, underneath how it was described.
2. How each party is reading the same events, and why each reading makes
   sense from where they stand.
3. Where the two readings are genuinely incompatible, and where they only
   appear to be.
4. What follows from that — what is at stake if nothing changes.
5. What can realistically be done, by whom, and what cannot.
{guidance_block}

LENGTH: under {word_budget} words. Write to the point, then stop.
"""


def constrained_generation_user(
    occurring_json: str,
    distilled_wholeness_json: str,
    final_mediation_json: str,
    all_findings_json: str,
    conflict_type_summary_json: str,
    salient_conflicts_json: str,
    word_budget: int = 900,
) -> str:
    return f"""\
Write the analysis.

THE SITUATION:
{occurring_json}

WHAT THE ANALYSIS ESTABLISHED (across all cycles):
{distilled_wholeness_json}

THE MOST SIGNIFICANT POINTS OF DIFFERENCE:
{salient_conflicts_json}

EVERYTHING THAT FOLLOWS FROM THEM:
{all_findings_json}

Write for the two people involved, in under {word_budget} words. Plain prose
only — no headings, no lists, no labels, no numbers about the analysis
itself. Return the document text and nothing else.
"""

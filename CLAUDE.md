# Cultural Conflict Resolution System — Build Instructions

## Who you are building for
Arman (CS student, George Brown College). He knows Python well and has built
production systems on AWS. Do not explain basics. Write complete, working code.
The system runs on AWS Bedrock using Claude Sonnet. Infrastructure is already
handled — focus only on application code.

## What this system does
A user submits a description of a cultural conflict. The system runs it through
a multi-stage pipeline and returns a structured analysis the user can learn from.
It is NOT a chatbot. It is a reasoning pipeline.

The pipeline has five phases:
1. Reformulation — parse raw scenario into 6 structured nodes (The Occurring)
2. Critical Thinking — main agent reasons across a 6×7 grid (42 cells)
3. Challenger — second agent argues the opposing cultural frame across the same grid
4. Mediation — compare the two grids, find what disagreements reveal
5. Constrained Generation — write final output using ONLY what mediation found

The loop: after mediation, check if anything new was learned. If yes, run again
(max 3 cycles). Stop when converged or max cycles hit.

---

## File structure to create

```
thirdspace/
├── models.py        # All Pydantic data models
├── llm.py           # LLM wrapper, context utilities, and all prompt functions
├── agents.py        # The 5 pipeline functions (one per phase)
├── graph.py         # LangGraph StateGraph wiring everything together
├── main.py          # Entry point — run one scenario end to end
└── requirements.txt
```

---

## Step 1 — models.py

Create these Pydantic models exactly. Do not simplify or merge them.

```python
from pydantic import BaseModel
from typing import Literal

# --- Constants (never use raw integers in logic) ---
NUM_FACETS = 6
NUM_DIMENSIONS = 7
NUM_CELLS = NUM_FACETS * NUM_DIMENSIONS

FACET_LIST = [
    "individuals",
    "interactional_event",
    "process",
    "evidence_and_data",
    "concepts_theories_rules",
    "inferences",
]

DIMENSION_LIST = [
    "objective",
    "query_formulation",
    "perspective",
    "assumptions",
    "evidence",
    "concepts",
    "implications",
]

FACETS = Literal[
    "individuals",
    "interactional_event",
    "process",
    "evidence_and_data",
    "concepts_theories_rules",
    "inferences",
]

CT_DIMENSIONS = Literal[
    "objective",
    "query_formulation",
    "perspective",
    "assumptions",
    "evidence",
    "concepts",
    "implications",
]


# --- Phase 1: The Occurring ---

class OccurringNode(BaseModel):
    node_id: str
    facet: FACETS
    content: str
    assumptions: list[str]
    last_updated_cycle: int = 0


class TheOccurring(BaseModel):
    scenario_id: str
    raw_scenario: str
    nodes: list[OccurringNode]  # always exactly NUM_FACETS nodes

    def get_node(self, facet: str) -> OccurringNode | None:
        return next((n for n in self.nodes if n.facet == facet), None)


# --- Phase 2 & 3: Reasoning Grids ---

class ReflectionFlag(BaseModel):
    measure: str   # clarity | correctness | precision | proportion |
                   # depth | comprehensiveness | logic | fairness
    issue: str


class NodeDimensionCell(BaseModel):
    facet: FACETS
    dimension: CT_DIMENSIONS
    content: str
    reflection_flags: list[ReflectionFlag] = []
    # Challenger-only field. None for main agent cells.
    response_type: Literal["challenge", "reframe", "confirm"] | None = None
    cycle_last_updated: int = 0


class NodeReasoningGrid(BaseModel):
    cycle: int
    agent: Literal["main", "challenger"]
    cells: list[NodeDimensionCell]  # exactly NUM_CELLS cells

    def get_cell(self, facet: str, dimension: str) -> NodeDimensionCell | None:
        return next(
            (c for c in self.cells if c.facet == facet and c.dimension == dimension),
            None,
        )

    def get_facet_row(self, facet: str) -> list[NodeDimensionCell]:
        return [c for c in self.cells if c.facet == facet]


# --- Phase 4: Mediation ---

class CellConflict(BaseModel):
    facet: str
    dimension: str
    main_summary: str
    challenger_summary: str
    challenger_response_type: Literal["challenge", "reframe", "confirm"]
    implication: str


class WholenessImplyingNodes(BaseModel):
    """Distilled output that feeds the next cycle and constrains final generation."""
    query_assumptions: list[str]
    reformulated_query: str | None
    new_objective: str | None
    updated_challenger_perspective: str | None
    objective_assumptions: list[str]
    perspective_assumptions: list[str]


class ImplyingNodes(BaseModel):
    cycle: int
    agreements: list[str]
    conflicts: list[CellConflict]
    what_conflicts_imply: list[str]
    updated_assumptions: list[str]
    wholeness: WholenessImplyingNodes
    converged: bool


# --- LLM-as-a-judge scores ---

class JudgeScore(BaseModel):
    cycle: int
    empathy: int           # 1-5
    consensus: int         # 1-5
    escalation_risk: int   # 1-5
    cultural_insight: int  # 1-5
    reasoning: str


# --- Session Memory ---

class SessionMemory(BaseModel):
    scenario_id: str
    main_cultural_frame: str
    challenger_cultural_frame: str
    occurring: TheOccurring | None = None
    main_grids: list[NodeReasoningGrid] = []
    challenger_grids: list[NodeReasoningGrid] = []
    mediation_history: list[ImplyingNodes] = []
    judge_scores: list[JudgeScore] = []
    current_cycle: int = 0
    max_cycles: int = 3
    final_output: str | None = None
    converged: bool = False
```

---

## Step 2 — llm.py

Write this file in full. It has four sections:
1. The Bedrock client (singleton — one client reused across all calls)
2. Three context engineering utilities from Omid's update
3. The base LLM callers including the self-refinement wrapper
4. All prompt functions for every phase

```python
import json
import os
import time
import logging
from typing import Any
from functools import lru_cache

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-sonnet-4-20250514-v1:0",
)
BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")


# ---------------------------------------------------------------------------
# Bedrock client — singleton. Do NOT call boto3.client() inside call_llm.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_client():
    """One client for the process lifetime. lru_cache makes it a singleton."""
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


# ---------------------------------------------------------------------------
# Base callers
# ---------------------------------------------------------------------------

def call_llm(
    system: str,
    user: str,
    model: str = BEDROCK_MODEL_ID,
    max_tokens: int = 8000,
    retries: int = 3,
) -> str:
    client = get_client()
    request = {
        "modelId": model,
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.3},
    }
    for attempt in range(retries):
        try:
            response = client.converse(**request)
            return response["output"]["message"]["content"][0]["text"]
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ThrottlingException", "TooManyRequestsException"):
                wait = 2 ** attempt * 5
                logger.warning(f"Throttled, waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            elif code == "ValidationException":
                logger.error(f"Bedrock validation error: {e}")
                raise
            elif code == "AccessDeniedException":
                raise PermissionError(
                    f"Bedrock access denied for model {model}. "
                    f"Check IAM role has bedrock:InvokeModel and the model is enabled."
                ) from e
            else:
                logger.error(f"Bedrock error on attempt {attempt+1}: {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
    raise RuntimeError("Bedrock call failed after all retries")


def call_llm_json(system: str, user: str, **kwargs) -> Any:
    raw = call_llm(system, user, **kwargs)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Model returned non-JSON:\n{raw[:500]}")
        raise ValueError(f"Model response was not valid JSON: {e}") from e


# ---------------------------------------------------------------------------
# Context engineering utilities (from Omid's context engineering update)
# ---------------------------------------------------------------------------

# Reflection measure → corrective instruction.
# Used by call_llm_json_with_refinement when a cell flags a failed measure.
REFLECTION_INSTRUCTIONS: dict[str, str] = {
    "clarity": (
        "Reflect on Clarity: Is this completely clear and unambiguous? "
        "Paraphrase the core argument and bring concrete examples to ground it."
    ),
    "correctness": (
        "Reflect on Correctness: Is this assessment accurate and objective? "
        "Ensure the analysis has not been misled by the narrator's emotional framing."
    ),
    "precision": (
        "Reflect on Precision: Is this too vague or generalized? "
        "Bring specific granular details and be explicitly descriptive about the dynamics."
    ),
    "proportion": (
        "Reflect on Proportion: Is this directly related to the core conflict? "
        "Analyze how heavily this point affects the overall result."
    ),
    "depth": (
        "Reflect on Depth: Have you bypassed the underlying complexities? "
        "Address structural difficulties and root causes, not just the surface."
    ),
    "comprehensiveness": (
        "Reflect on Comprehensiveness: Are you missing critical perspectives? "
        "State how this moment appears from the opposing cultural frame."
    ),
    "logic": (
        "Reflect on Logic: Is this a valid deductive step? "
        "Verify that the observations actually imply this result."
    ),
    "fairness": (
        "Reflect on Fairness: Have all viewpoints been treated in a balanced way? "
        "Audit for bias or quietly altered details that favor one frame."
    ),
}


def call_llm_json_with_refinement(
    system: str,
    user: str,
    max_refine_attempts: int = 2,
    **kwargs,
) -> Any:
    """
    Wraps call_llm_json with a validation and self-correction loop.

    Used for single-cell calls (CT agent and Challenger).
    Each call returns {"content": "...", "reflection_flags": [...], ...}.

    Checks:
    - content exists and has at least 3 sentences
    - reflection_flags are addressed with corrective prompts on retry

    If validation passes, returns immediately. If it fails, injects
    specific corrective feedback and retries up to max_refine_attempts times.
    """
    result = call_llm_json(system, user, **kwargs)

    for attempt in range(max_refine_attempts):
        issues = []

        content = result.get("content", "").strip()
        if not content:
            issues.append("The 'content' field is empty. Write 3 to 6 sentences.")
        else:
            sentence_count = len([s for s in content.split(".") if len(s.strip()) > 3])
            if sentence_count < 3:
                issues.append(
                    f"Content has only {sentence_count} sentence(s). "
                    f"Write at least 3 sentences."
                )

        for flag in result.get("reflection_flags", []):
            if isinstance(flag, dict):
                measure = flag.get("measure", "").lower().strip()
                issue = flag.get("issue", "Unspecified failure")
                instruction = REFLECTION_INSTRUCTIONS.get(
                    measure,
                    "Review the depth, fairness, and structural validity of this entry.",
                )
                issues.append(
                    f"Self-evaluation failed [{measure.upper()}]: {issue}\n"
                    f"Correction required: {instruction}"
                )

        if not issues:
            return result

        logger.warning(
            f"[Refinement attempt {attempt + 1}] {len(issues)} issue(s) found, retrying."
        )
        feedback = "\n".join(issues[:5])
        if len(issues) > 5:
            feedback += f"\n...and {len(issues) - 5} more."

        refine_user = (
            f"Your previous response failed validation:\n\n{feedback}\n\n"
            f"Correct your response. Apply the correction directives above. "
            f"Write 3 to 6 sentences. Return valid JSON."
        )
        result = call_llm_json(system, refine_user, **kwargs)

    return result


def compress_historical_context(all_past_cycles: list[dict]) -> list[dict]:
    """
    Prevents context window bloat across cycles.

    Keeps the most recent mediation cycle fully intact.
    Shrinks older cycles into compact gists containing only:
    - cycle number
    - converged status
    - which facets had conflicts
    - what assumptions were updated

    Called in run_mediation() before building the mediation prompt.
    """
    if not all_past_cycles:
        return []
    if len(all_past_cycles) == 1:
        return all_past_cycles

    compressed = []
    for older in all_past_cycles[:-1]:
        compressed.append({
            "cycle": older.get("cycle"),
            "converged": older.get("converged"),
            "conflict_facets": [c.get("facet") for c in older.get("conflicts", [])],
            "updated_assumptions": older.get("updated_assumptions", []),
        })

    # Most recent cycle: keep in full
    compressed.append(all_past_cycles[-1])
    return compressed


def linearize_grid_to_matrix_map(cells: list[dict]) -> dict:
    """
    Converts a flat list of cell dicts into a nested lookup map.

    From:
        [{"facet": "individuals", "dimension": "objective", "content": "..."}, ...]

    To:
        {"individuals": {"objective": {"content": "...", "reflection_flags": []}}}

    Used when passing the main agent's grid to the Challenger so the
    Challenger can look up any cell by facet and dimension directly.
    """
    matrix: dict = {}
    for cell in cells:
        facet = cell.get("facet")
        dimension = cell.get("dimension")
        if facet not in matrix:
            matrix[facet] = {}
        matrix[facet][dimension] = {
            "content": cell.get("content", ""),
            "reflection_flags": cell.get("reflection_flags", []),
            "response_type": cell.get("response_type"),
        }
    return matrix


# ---------------------------------------------------------------------------
# Phase 1 — Reformulation prompts
# ---------------------------------------------------------------------------

REFORMULATION_SYSTEM = """\
You are a scenario decomposition specialist. Your only job is to take a raw,
emotionally charged, one-sided description of a cultural conflict and break it
into exactly six structured nodes called The Occurring.

WHAT THE OCCURRING IS:
A conflict is not one thing. It is several independent things happening simultaneously.
The Occurring separates them so they can be reasoned about independently.

THE SIX NODES:

NODE 1: individuals
Who is involved. The critical rule: describe each person as they are ALREADY AFFECTED
by the situation before they have even responded to it.
Do not describe them neutrally. Describe them as someone already shaped by their
cultural background, their relational history with the other person, and the event.
What assumptions are they already carrying into this moment?
Wrong: "Person A is a manager. Person B is an employee."
Right: "Person A enters this moment already interpreting silence as disengagement,
shaped by a culture that treats non-response as passive resistance."

NODE 2: interactional_event
The specific moment of friction. What a camera would have recorded.
Strip out ALL interpretation. Strip out ALL backstory. No labels like "rude" or
"disrespectful" — those belong in concepts. Observable event only.

NODE 3: process
The timeline and context around the event. How did things unfold? What happened
before? Is this seen as an isolated incident or part of a pattern? What model
of time is each person using?

NODE 4: evidence_and_data
ONLY what can actually be verified. Separate fact from assumption.
Ask: is this directly observable, or is it interpretation?
Put interpretations in inferences. Put frameworks in concepts.
Be explicit when evidence is thin: "The only verifiable fact is X."

NODE 5: concepts_theories_rules
The cultural frameworks, communication norms, and interpretive rules each party
is applying — usually without knowing it. Name them explicitly for both sides.
Do not describe behavior here — name the underlying rule that makes that
behavior make sense within each culture.

NODE 6: inferences
What can actually be deduced from nodes 1-5. NOT the narrator's conclusion.
What does the structure of the decomposition itself imply happened?
This node often contradicts the narrator's framing. That is correct.

CRITICAL RULES:
- Do not echo the narrator's language or labels. Rephrase everything structurally.
- Each node must contain something the narrator did NOT explicitly state.
- Each node carries a list of explicit assumptions. Name them.
- You are not summarizing. You are decomposing.

Respond ONLY with valid JSON. No preamble. No markdown fences.
"""


def reformulation_user(raw_scenario: str, scenario_id: str) -> str:
    return f"""\
Decompose this cultural conflict scenario into The Occurring.

SCENARIO ID: {scenario_id}

RAW SCENARIO:
{raw_scenario}

Return this exact JSON. All 6 nodes required:
{{
  "nodes": [
    {{
      "node_id": "individuals",
      "facet": "individuals",
      "content": "<description of each person as already affected>",
      "assumptions": ["<assumption 1>", "<assumption 2>"]
    }},
    {{
      "node_id": "interactional_event",
      "facet": "interactional_event",
      "content": "<observable event stripped of interpretation>",
      "assumptions": ["..."]
    }},
    {{
      "node_id": "process",
      "facet": "process",
      "content": "<timeline, context, model of time each party is using>",
      "assumptions": ["..."]
    }},
    {{
      "node_id": "evidence_and_data",
      "facet": "evidence_and_data",
      "content": "<verifiable facts only>",
      "assumptions": ["..."]
    }},
    {{
      "node_id": "concepts_theories_rules",
      "facet": "concepts_theories_rules",
      "content": "<named cultural frameworks and norms for both parties>",
      "assumptions": ["..."]
    }},
    {{
      "node_id": "inferences",
      "facet": "inferences",
      "content": "<what the structure implies — may contradict the narrator>",
      "assumptions": ["..."]
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Phase 2 — Critical thinking prompts (single-cell, one call per cell)
# ---------------------------------------------------------------------------

def critical_thinking_cell_system(cultural_frame: str, facet: str, dimension: str) -> str:
    """
    System prompt for a single cell: one facet × one dimension.
    This is intentionally narrow. The CT agent makes NUM_CELLS separate calls,
    one per (facet, dimension) pair. Never batch dimensions.
    """
    return f"""\
You are the primary reasoning agent in a cultural conflict analysis system.

YOUR CULTURAL FRAME: {cultural_frame}
This is not a flaw — it is your function. Whenever your frame shapes your
reasoning, name it. Do not let {cultural_frame} assumptions appear as
neutral observations.

YOUR TASK: Reason about ONE specific node × dimension cell.
NODE (facet): {facet}
DIMENSION: {dimension}

DIMENSION DEFINITIONS:
- objective: What is the actual goal of reasoning about this node? Is it feasible?
- query_formulation: Is the question framed correctly? Can it be broken into sub-questions?
- perspective: What viewpoint is guiding you? Where does it break down?
- assumptions: What are you taking for granted? Which assumptions are load-bearing?
- evidence: Which facts support this? Any alternative explanations?
- concepts: Which cultural frameworks are you applying? Have you quietly adjusted
  any assumption to fit a preferred theory? If yes, name it.
- implications: If your reasoning is correct, what specifically follows?

SELF-CHECK before writing. Record failures in reflection_flags:
  clarity, correctness, precision, proportion, depth,
  comprehensiveness, logic, fairness

Write 3 to 6 sentences. Be specific to THIS facet and THIS dimension only.

Respond ONLY with valid JSON. No preamble. No markdown fences.
Return:
{{
  "content": "<your reasoning — 3 to 6 sentences>",
  "reflection_flags": [
    {{"measure": "<name>", "issue": "<what failed>"}}
  ]
}}
reflection_flags is [] if all checks pass.
"""


def critical_thinking_cell_user(occurring_json: str, facet: str, dimension: str) -> str:
    return f"""\
Reason about this specific node and dimension.

FACET: {facet}
DIMENSION: {dimension}

THE OCCURRING (full context):
{occurring_json}

Focus only on the {facet} node as seen through the {dimension} lens.
"""


# ---------------------------------------------------------------------------
# Phase 3 — Challenger prompts (single-cell)
# ---------------------------------------------------------------------------

def challenger_cell_system(
    challenger_frame: str, main_frame: str, facet: str, dimension: str
) -> str:
    return f"""\
You are the Challenger agent in a cultural conflict analysis system.

YOUR CULTURAL FRAME: {challenger_frame}
THE FRAME YOU ARE CHALLENGING: {main_frame}

YOUR TASK: Respond to the main agent's reasoning for ONE cell.
NODE (facet): {facet}
DIMENSION: {dimension}

CLASSIFY your response as exactly one of:
- "challenge": The main agent applied {main_frame} norms as if universal.
  Be explicit: "The main agent's reasoning depends on [specific norm] being
  treated as universal. From {challenger_frame}, [explain why it does not apply]."
- "reframe": Not wrong, but asking the wrong question. State the better question.
- "confirm": Holds from your frame too. Explain WHY from {challenger_frame}.

THE MOST IMPORTANT CELLS:
Find where the main agent treated a {main_frame}-specific norm as a universal
human fact. When you find one, name it precisely.

Write 3 to 6 sentences. Stay in {challenger_frame} consistently.

Respond ONLY with valid JSON. No preamble. No markdown fences.
Return:
{{
  "response_type": "challenge" | "reframe" | "confirm",
  "content": "<your reasoning — 3 to 6 sentences>",
  "reflection_flags": []
}}
"""


def challenger_cell_user(
    occurring_json: str,
    main_cell_content: str,
    main_cell_response_type: str | None,
    facet: str,
    dimension: str,
) -> str:
    return f"""\
Challenge the main agent's reasoning for this cell.

FACET: {facet}
DIMENSION: {dimension}

MAIN AGENT'S REASONING FOR THIS CELL:
{main_cell_content}

THE OCCURRING (full context):
{occurring_json}
"""


# ---------------------------------------------------------------------------
# Phase 4 — Mediation prompts
# ---------------------------------------------------------------------------

MEDIATION_SYSTEM = """\
You are the mediation engine in a cultural conflict analysis system.

You receive a compact comparison of cell pairs from two agents and The Occurring.
Compare all pairs. For each: agreement or conflict?

For conflicts — the IMPLICATION field is critical.
Do not write "they disagree about silence."
Write what the disagreement REVEALS about the conflict.
Complete: "This reveals that [structural feature invisible from inside either frame]."

Weak: "The two agents have different views on communication."
Strong: "This reveals that both parties believed they were being clear — the
breakdown happened at the layer of what being clear means, not intention."

PATTERN-LEVEL SYNTHESIS:
After all pairs, look at where conflicts cluster.
What does the distribution tell you that no individual conflict reveals?
Goes in what_conflicts_imply.

ASSUMPTION UPDATES:
Name the specific node and the specific assumption to replace.
Not "update communication assumptions" but
"individuals node: remove [X] → replace with [Y]."

CONVERGENCE:
converged=true only if conflicts this cycle are not meaningfully different
from the previous cycle. Same tensions re-appearing = converged.
New facet of conflict appearing = not converged.

Respond ONLY with valid JSON. No preamble. No markdown fences.
"""


def mediation_user(
    occurring_json: str,
    cell_pairs_json: str,
    compressed_history_json: str,
    cycle: int,
) -> str:
    return f"""\
Mediate between the two grids. Cycle {cycle}.

THE OCCURRING:
{occurring_json}

CELL-BY-CELL COMPARISON (all pairs):
{cell_pairs_json}

PRIOR MEDIATION HISTORY (compressed — null on cycle 1):
{compressed_history_json}

Return:
{{
  "cycle": {cycle},
  "agreements": ["<what both agents agreed on>"],
  "conflicts": [
    {{
      "facet": "<facet>",
      "dimension": "<dimension>",
      "response_type": "challenge" | "reframe" | "confirm",
      "main_content": "<brief summary of main agent>",
      "challenger_content": "<brief summary of challenger>",
      "implication": "<This reveals that...>"
    }}
  ],
  "what_conflicts_imply": ["<pattern-level finding>"],
  "updated_assumptions": ["<node>: remove [X] → replace with [Y]"],
  "wholeness": {{
    "query_assumptions": ["<assumption embedded in how the question was framed>"],
    "reformulated_query": "<clearer question or null>",
    "new_objective": "<revised objective or null>",
    "updated_challenger_perspective": "<adjustment to challenger frame or null>",
    "objective_assumptions": ["<assumption behind the stated objective>"],
    "perspective_assumptions": ["<assumption behind the cultural frame>"]
  }},
  "converged": true or false
}}

converged=true ONLY if this cycle's conflicts are not meaningfully different
from the previous cycle. New tensions = NOT converged.
"""


# ---------------------------------------------------------------------------
# Phase 5 — Constrained generation prompts
# ---------------------------------------------------------------------------

CONSTRAINED_GENERATION_SYSTEM = """\
You are the final output writer in a cultural conflict analysis system.

YOUR CONSTRAINT:
Every claim must trace back to something the mediation process found.
No new inferences. No advice from training data. No generic cultural wisdom.
If mediation did not produce enough to ground a claim, say so.

FIVE SECTIONS:

WHAT ACTUALLY HAPPENED
The mediation's version, not the user's version. 2-3 paragraphs.

WHERE THE CULTURES DIVERGED
For each significant conflict: what each frame expected, why neither is wrong,
what the specific misread was. Choose conflicts with the strongest implications.

WHAT THE CONFLICT ACTUALLY IMPLIES
Use what_conflicts_imply from mediation. Be direct. 2-3 paragraphs.

WHAT EACH PERSON LIKELY EXPERIENCED
What did each person feel was happening? What did they assume the other was doing?
Write with empathy for both — not blame, not excuse. One paragraph per person.

HOW TO MOVE FORWARD
Grounded only in updated_assumptions and new_objective from mediation.
If the conflict is structural and cannot be solved by direct conversation alone,
say that. Do not offer "have an honest conversation" as a solution to a problem
that is exactly about what counts as honest communication. 2-3 paragraphs.
"""


def constrained_generation_user(
    occurring_json: str,
    distilled_wholeness_json: str,
    final_mediation_json: str,
) -> str:
    return f"""\
Write the final conflict analysis document.
Use only what mediation found. Do not introduce new claims.

THE OCCURRING:
{occurring_json}

DISTILLED WHOLENESS NODES FROM ALL CYCLES:
{distilled_wholeness_json}

FINAL MEDIATION OUTPUT (most weight):
{final_mediation_json}

Write the five-section document now.
"""


# ---------------------------------------------------------------------------
# Judge prompts (LLM-as-a-judge, runs after generation)
# ---------------------------------------------------------------------------

def judge_system() -> str:
    return """\
You are an impartial evaluator of cultural conflict analysis outputs.
You receive a mediation output and final analysis document.
You score four dimensions on a 1-5 scale.

1 = Very poor  2 = Poor  3 = Adequate  4 = Good  5 = Excellent

DIMENSIONS:
- empathy (1-5): Does the output accurately reflect what BOTH parties experienced?
- consensus (1-5): How much did the two cultural frames converge?
- escalation_risk (1-5): How likely is the conflict to worsen without intervention?
- cultural_insight (1-5): Did the analysis surface assumptions neither party would see?

Reason step by step before scoring. Write reasoning first, then scores.

Respond ONLY with valid JSON. No preamble. No markdown fences.
"""


def judge_user(final_output: str, mediation_json: str) -> str:
    return f"""\
Evaluate this cultural conflict analysis.

FINAL OUTPUT:
{final_output}

MEDIATION FINDINGS:
{mediation_json}

Return:
{{
  "reasoning": "<step by step evaluation>",
  "empathy": <1-5>,
  "consensus": <1-5>,
  "escalation_risk": <1-5>,
  "cultural_insight": <1-5>
}}
"""
```

---

## Step 3 — agents.py

Five pipeline functions. Key rules:
- The CT agent and Challenger each make NUM_CELLS separate LLM calls (one per cell).
- `call_llm_json_with_refinement` is used for all cell-level calls.
- `linearize_grid_to_matrix_map` is used before passing the main grid to the Challenger.
- `compress_historical_context` is used before building the mediation prompt.
- Constrained generation receives ONLY the distilled wholeness nodes, not full grids.

```python
import json
import logging
from models import (
    TheOccurring, OccurringNode,
    NodeReasoningGrid, NodeDimensionCell, ReflectionFlag,
    ImplyingNodes, CellConflict, WholenessImplyingNodes,
    JudgeScore,
    FACET_LIST, DIMENSION_LIST, NUM_FACETS, NUM_CELLS,
)
from llm import (
    call_llm, call_llm_json, call_llm_json_with_refinement,
    compress_historical_context, linearize_grid_to_matrix_map,
    REFORMULATION_SYSTEM, reformulation_user,
    critical_thinking_cell_system, critical_thinking_cell_user,
    challenger_cell_system, challenger_cell_user,
    MEDIATION_SYSTEM, mediation_user,
    CONSTRAINED_GENERATION_SYSTEM, constrained_generation_user,
    judge_system, judge_user,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 — Reformulation
# ---------------------------------------------------------------------------

def run_reformulation(raw_scenario: str, scenario_id: str) -> TheOccurring:
    result = call_llm_json(
        system=REFORMULATION_SYSTEM,
        user=reformulation_user(raw_scenario, scenario_id),
    )
    nodes = [
        OccurringNode(
            node_id=n["node_id"],
            facet=n["facet"],
            content=n["content"],
            assumptions=n["assumptions"],
            last_updated_cycle=0,
        )
        for n in result["nodes"]
    ]
    assert len(nodes) == NUM_FACETS, f"Expected {NUM_FACETS} nodes, got {len(nodes)}"
    return TheOccurring(
        scenario_id=scenario_id,
        raw_scenario=raw_scenario,
        nodes=nodes,
    )


# ---------------------------------------------------------------------------
# Phase 2 — Critical thinking (main agent)
# One LLM call per (facet, dimension) pair. Never batched.
# ---------------------------------------------------------------------------

def run_critical_thinking(
    occurring: TheOccurring,
    cultural_frame: str,
    cycle: int,
) -> NodeReasoningGrid:
    occurring_json = occurring.model_dump_json()
    cells: list[NodeDimensionCell] = []

    for facet in FACET_LIST:
        for dimension in DIMENSION_LIST:
            # One focused call per cell
            result = call_llm_json_with_refinement(
                system=critical_thinking_cell_system(cultural_frame, facet, dimension),
                user=critical_thinking_cell_user(occurring_json, facet, dimension),
            )
            flags = [
                ReflectionFlag(measure=f["measure"], issue=f["issue"])
                for f in result.get("reflection_flags", [])
            ]
            cells.append(NodeDimensionCell(
                facet=facet,
                dimension=dimension,
                content=result["content"],
                reflection_flags=flags,
                response_type=None,
                cycle_last_updated=cycle,
            ))

    assert len(cells) == NUM_CELLS, f"Expected {NUM_CELLS} cells, got {len(cells)}"
    return NodeReasoningGrid(cycle=cycle, agent="main", cells=cells)


# ---------------------------------------------------------------------------
# Phase 3 — Challenger
# Passes the main grid as a linearized matrix map (easier for LLM to read).
# One LLM call per cell with response_type classification.
# ---------------------------------------------------------------------------

def run_challenger(
    occurring: TheOccurring,
    main_grid: NodeReasoningGrid,
    challenger_frame: str,
    main_frame: str,
    cycle: int,
) -> NodeReasoningGrid:
    occurring_json = occurring.model_dump_json()

    # Convert flat cell list to nested map before passing to challenger
    main_cells_raw = [
        {
            "facet": c.facet,
            "dimension": c.dimension,
            "content": c.content,
            "reflection_flags": [f.model_dump() for f in c.reflection_flags],
        }
        for c in main_grid.cells
    ]
    main_matrix = linearize_grid_to_matrix_map(main_cells_raw)

    cells: list[NodeDimensionCell] = []

    for facet in FACET_LIST:
        for dimension in DIMENSION_LIST:
            # Get the main agent's content for this specific cell
            main_cell_data = main_matrix.get(facet, {}).get(dimension, {})
            main_cell_content = main_cell_data.get("content", "")

            result = call_llm_json_with_refinement(
                system=challenger_cell_system(
                    challenger_frame, main_frame, facet, dimension
                ),
                user=challenger_cell_user(
                    occurring_json=occurring_json,
                    main_cell_content=main_cell_content,
                    main_cell_response_type=None,
                    facet=facet,
                    dimension=dimension,
                ),
            )
            flags = [
                ReflectionFlag(measure=f["measure"], issue=f["issue"])
                for f in result.get("reflection_flags", [])
            ]
            cells.append(NodeDimensionCell(
                facet=facet,
                dimension=dimension,
                content=result["content"],
                reflection_flags=flags,
                response_type=result.get("response_type"),
                cycle_last_updated=cycle,
            ))

    assert len(cells) == NUM_CELLS, f"Expected {NUM_CELLS} cells, got {len(cells)}"
    return NodeReasoningGrid(cycle=cycle, agent="challenger", cells=cells)


# ---------------------------------------------------------------------------
# Phase 4 — Mediation
# Passes a compact cell-pair comparison and compressed history.
# ---------------------------------------------------------------------------

def run_mediation(
    occurring: TheOccurring,
    main_grid: NodeReasoningGrid,
    challenger_grid: NodeReasoningGrid,
    all_mediation_history: list[ImplyingNodes],
    cycle: int,
) -> ImplyingNodes:
    occurring_json = occurring.model_dump_json()

    # Build compact cell-pair comparison (not full grids)
    cell_pairs = []
    for facet in FACET_LIST:
        for dimension in DIMENSION_LIST:
            main_cell = main_grid.get_cell(facet, dimension)
            chall_cell = challenger_grid.get_cell(facet, dimension)
            cell_pairs.append({
                "facet": facet,
                "dimension": dimension,
                "main": main_cell.content if main_cell else "",
                "challenger": chall_cell.content if chall_cell else "",
                "response_type": chall_cell.response_type if chall_cell else None,
            })

    cell_pairs_json = json.dumps(cell_pairs, ensure_ascii=False)

    # Compress history so older cycles don't bloat the prompt
    raw_history = [m.model_dump() for m in all_mediation_history]
    compressed = compress_historical_context(raw_history)
    compressed_history_json = json.dumps(compressed, ensure_ascii=False)

    result = call_llm_json(
        system=MEDIATION_SYSTEM,
        user=mediation_user(
            occurring_json=occurring_json,
            cell_pairs_json=cell_pairs_json,
            compressed_history_json=compressed_history_json,
            cycle=cycle,
        ),
    )

    conflicts = [
        CellConflict(
            facet=c["facet"],
            dimension=c["dimension"],
            main_summary=c["main_content"],
            challenger_summary=c["challenger_content"],
            challenger_response_type=c.get("response_type", "challenge"),
            implication=c["implication"],
        )
        for c in result.get("conflicts", [])
    ]

    w = result.get("wholeness", {})
    wholeness = WholenessImplyingNodes(
        query_assumptions=w.get("query_assumptions", []),
        reformulated_query=w.get("reformulated_query"),
        new_objective=w.get("new_objective"),
        updated_challenger_perspective=w.get("updated_challenger_perspective"),
        objective_assumptions=w.get("objective_assumptions", []),
        perspective_assumptions=w.get("perspective_assumptions", []),
    )

    return ImplyingNodes(
        cycle=cycle,
        agreements=result.get("agreements", []),
        conflicts=conflicts,
        what_conflicts_imply=result.get("what_conflicts_imply", []),
        updated_assumptions=result.get("updated_assumptions", []),
        wholeness=wholeness,
        converged=result.get("converged", False),
    )


# ---------------------------------------------------------------------------
# Phase 5 — Constrained generation
# Receives ONLY distilled wholeness nodes. Not the full grids.
# ---------------------------------------------------------------------------

def run_constrained_generation(
    occurring: TheOccurring,
    mediation_history: list[ImplyingNodes],
) -> str:
    # Extract only the wholeness nodes — this is the constraint
    distilled = [m.wholeness.model_dump() for m in mediation_history]
    final_mediation = mediation_history[-1]

    return call_llm(
        system=CONSTRAINED_GENERATION_SYSTEM,
        user=constrained_generation_user(
            occurring_json=occurring.model_dump_json(),
            distilled_wholeness_json=json.dumps(distilled, ensure_ascii=False),
            final_mediation_json=final_mediation.model_dump_json(),
        ),
    )


# ---------------------------------------------------------------------------
# Judge (runs after generation, scores output quality)
# ---------------------------------------------------------------------------

def run_judge(
    final_output: str,
    mediation_history: list[ImplyingNodes],
    cycle: int,
) -> JudgeScore:
    mediation_json = json.dumps(
        [m.model_dump() for m in mediation_history],
        ensure_ascii=False,
    )
    result = call_llm_json(
        system=judge_system(),
        user=judge_user(final_output, mediation_json),
    )
    return JudgeScore(
        cycle=cycle,
        empathy=result["empathy"],
        consensus=result["consensus"],
        escalation_risk=result["escalation_risk"],
        cultural_insight=result["cultural_insight"],
        reasoning=result["reasoning"],
    )
```

---

## Step 4 — graph.py

LangGraph StateGraph. Each phase is a node. The loop is a conditional edge.
The mediation node now passes the full history list to run_mediation instead
of just the previous one.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from models import SessionMemory
from agents import (
    run_reformulation,
    run_critical_thinking,
    run_challenger,
    run_mediation,
    run_constrained_generation,
    run_judge,
)
import logging

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    memory: SessionMemory


def node_reformulate(state: GraphState) -> GraphState:
    mem = state["memory"]
    logger.info(f"[{mem.scenario_id}] Phase 1: Reformulation")
    occurring = run_reformulation(
        raw_scenario=mem.occurring.raw_scenario,
        scenario_id=mem.scenario_id,
    )
    mem.occurring = occurring
    return {"memory": mem}


def node_main_agent(state: GraphState) -> GraphState:
    mem = state["memory"]
    cycle = mem.current_cycle
    logger.info(f"[{mem.scenario_id}] Phase 2: Main agent, cycle {cycle}")
    grid = run_critical_thinking(
        occurring=mem.occurring,
        cultural_frame=mem.main_cultural_frame,
        cycle=cycle,
    )
    mem.main_grids.append(grid)
    return {"memory": mem}


def node_challenger(state: GraphState) -> GraphState:
    mem = state["memory"]
    cycle = mem.current_cycle
    logger.info(f"[{mem.scenario_id}] Phase 3: Challenger, cycle {cycle}")
    grid = run_challenger(
        occurring=mem.occurring,
        main_grid=mem.main_grids[-1],
        challenger_frame=mem.challenger_cultural_frame,
        main_frame=mem.main_cultural_frame,
        cycle=cycle,
    )
    mem.challenger_grids.append(grid)
    return {"memory": mem}


def node_mediation(state: GraphState) -> GraphState:
    mem = state["memory"]
    cycle = mem.current_cycle
    logger.info(f"[{mem.scenario_id}] Phase 4: Mediation, cycle {cycle}")
    # Pass full history so mediation can compress it internally
    mediation = run_mediation(
        occurring=mem.occurring,
        main_grid=mem.main_grids[-1],
        challenger_grid=mem.challenger_grids[-1],
        all_mediation_history=mem.mediation_history,
        cycle=cycle,
    )
    mem.mediation_history.append(mediation)
    mem.converged = mediation.converged
    mem.current_cycle += 1
    return {"memory": mem}


def node_generate(state: GraphState) -> GraphState:
    mem = state["memory"]
    logger.info(f"[{mem.scenario_id}] Phase 5: Constrained generation")
    final_output = run_constrained_generation(
        occurring=mem.occurring,
        mediation_history=mem.mediation_history,
    )
    mem.final_output = final_output
    return {"memory": mem}


def node_judge(state: GraphState) -> GraphState:
    mem = state["memory"]
    logger.info(f"[{mem.scenario_id}] Judge: evaluating output quality")
    score = run_judge(
        final_output=mem.final_output,
        mediation_history=mem.mediation_history,
        cycle=mem.current_cycle,
    )
    mem.judge_scores.append(score)
    return {"memory": mem}


def should_loop(state: GraphState) -> str:
    mem = state["memory"]
    if mem.converged:
        logger.info(f"[{mem.scenario_id}] Converged at cycle {mem.current_cycle}")
        return "generate"
    if mem.current_cycle >= mem.max_cycles:
        logger.info(f"[{mem.scenario_id}] Max cycles reached")
        return "generate"
    logger.info(f"[{mem.scenario_id}] Not converged, running cycle {mem.current_cycle}")
    return "main_agent"


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("reformulate", node_reformulate)
    graph.add_node("main_agent", node_main_agent)
    graph.add_node("challenger", node_challenger)
    graph.add_node("mediation", node_mediation)
    graph.add_node("generate", node_generate)
    graph.add_node("judge", node_judge)

    graph.set_entry_point("reformulate")
    graph.add_edge("reformulate", "main_agent")
    graph.add_edge("main_agent", "challenger")
    graph.add_edge("challenger", "mediation")
    graph.add_conditional_edges(
        "mediation",
        should_loop,
        {"main_agent": "main_agent", "generate": "generate"},
    )
    graph.add_edge("generate", "judge")
    graph.add_edge("judge", END)

    return graph.compile()
```

---

## Step 5 — main.py

```python
import logging
import uuid
from models import SessionMemory, TheOccurring
from graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_scenario(
    raw_scenario: str,
    main_cultural_frame: str,
    challenger_cultural_frame: str,
    max_cycles: int = 3,
) -> SessionMemory:
    scenario_id = str(uuid.uuid4())[:8]
    memory = SessionMemory(
        scenario_id=scenario_id,
        main_cultural_frame=main_cultural_frame,
        challenger_cultural_frame=challenger_cultural_frame,
        occurring=TheOccurring(
            scenario_id=scenario_id,
            raw_scenario=raw_scenario,
            nodes=[],
        ),
        max_cycles=max_cycles,
    )
    graph = build_graph()
    result = graph.invoke({"memory": memory})
    return result["memory"]


if __name__ == "__main__":
    scenario = """
    I'm a Canadian working at a tech company. My new colleague is from Iran.
    Last week I gave him feedback on his code in our team meeting — I pointed
    out some inefficiencies directly in front of everyone, the way I always do
    with the team. He went completely silent, didn't respond, and has barely
    spoken to me since. My manager told me he seems upset but won't say why.
    I don't understand what I did wrong — I was just being honest and direct.
    """

    memory = run_scenario(
        raw_scenario=scenario,
        main_cultural_frame="Canadian low-context direct communication culture",
        challenger_cultural_frame="Iranian high-context honor-based communication culture",
        max_cycles=2,
    )

    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(memory.final_output)

    print("\n" + "=" * 60)
    print("JUDGE SCORES")
    print("=" * 60)
    for score in memory.judge_scores:
        print(f"Empathy:          {score.empathy}/5")
        print(f"Consensus:        {score.consensus}/5")
        print(f"Escalation Risk:  {score.escalation_risk}/5")
        print(f"Cultural Insight: {score.cultural_insight}/5")
        print(f"\nReasoning:\n{score.reasoning}")

    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Cycles completed:  {memory.current_cycle}")
    print(f"Converged:         {memory.converged}")
    if memory.mediation_history:
        last = memory.mediation_history[-1]
        print(f"Agreements found:  {len(last.agreements)}")
        print(f"Conflicts found:   {len(last.conflicts)}")
```

---

## Step 6 — requirements.txt

```
boto3>=1.34.0
langgraph>=0.2.0
pydantic>=2.0.0
```

---

## Important notes for Claude Code

### Architecture rules (never break these)

1. **One LLM call per cell.** The CT agent and Challenger each make `NUM_CELLS`
   separate calls. One per `(facet, dimension)` pair. If you find yourself
   writing a loop that makes ONE call and parses 42 cells from it, stop. That
   is wrong. Each cell gets its own focused call.

2. **`call_llm_json_with_refinement` for all cell calls.**
   Use it in `run_critical_thinking` and `run_challenger`. Use plain
   `call_llm_json` for reformulation and mediation (those don't produce cells).

3. **`linearize_grid_to_matrix_map` before passing to Challenger.**
   Convert the main grid's cells to a nested dict before sending to the
   challenger. The challenger reads `matrix[facet][dimension]["content"]`.

4. **`compress_historical_context` in mediation.**
   Call it on the raw mediation history before building the mediation prompt.
   Pass the compressed result, not the raw list.

5. **Constrained generation receives ONLY wholeness nodes.**
   Extract `[m.wholeness.model_dump() for m in mediation_history]`.
   Do not pass full grids or full mediation objects to the generation call.

6. **Mediation node passes full history to `run_mediation`.**
   `run_mediation` receives `all_mediation_history: list[ImplyingNodes]`.
   Compression happens inside `run_mediation`, not in the graph node.

7. **boto3 client is a singleton.** `get_client()` uses `@lru_cache(maxsize=1)`.
   Never call `boto3.client()` directly anywhere else in the codebase.

8. **No raw integers for grid dimensions.** Always use `NUM_FACETS`,
   `NUM_DIMENSIONS`, `NUM_CELLS`. Never the literal integer 42, 6, or 7.

### What was intentionally excluded from Omid's update

`ContextPayload` is not included — its `assemble()` is just string
concatenation and the `default_factory=field` line in it is a bug.
The mediation prompt function already handles context assembly correctly.

The `run_cultural_conflict_pipeline` function from Omid's file is not
included — `graph.py` replaces it with proper LangGraph orchestration.

### Testing order

Test each agent in isolation before running the full graph:
- `run_reformulation` with a hardcoded scenario string
- `run_critical_thinking` with a hardcoded `TheOccurring`
- `run_challenger` with hardcoded grids
- `run_mediation` with hardcoded grids and empty history
- Only then run `main.py`

### Error handling

If any LLM call returns malformed JSON, log the raw response and raise.
Do not silently continue with empty or partial data.
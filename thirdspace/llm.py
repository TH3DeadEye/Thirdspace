import json
import os
import re
import time
import logging
from typing import Any
from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from models import NUM_FACETS
from budget import (
    check_budget_before_call, record_success, record_failure,
)

logger = logging.getLogger(__name__)

# Default model. Sonnet 4.5 is the current Bedrock inference profile for Claude
# Sonnet (override with BEDROCK_MODEL_ID env var if your account has a newer
# profile enabled).
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)
# Challenger uses a different Claude SKU by default so it draws from a
# different checkpoint (different priors), partially addressing the PDF's
# "separate LLM / different RAG" design point without RAG. Override with
# CHALLENGER_MODEL_ID to pin both agents to the same model.
CHALLENGER_MODEL_ID = os.environ.get(
    "CHALLENGER_MODEL_ID",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
)
BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Network timeouts. We do app-level retries in call_llm, so disable boto3's
# built-in retry layer to avoid double-retrying (and double-billing) every
# transient failure.
_CONNECT_TIMEOUT_S = int(os.environ.get("BEDROCK_CONNECT_TIMEOUT_S", "10"))
_READ_TIMEOUT_S = int(os.environ.get("BEDROCK_READ_TIMEOUT_S", "120"))


# ---------------------------------------------------------------------------
# Bedrock client — singleton. Do NOT call boto3.client() inside call_llm.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_client():
    """One client for the process lifetime. lru_cache makes it a singleton."""
    config = Config(
        connect_timeout=_CONNECT_TIMEOUT_S,
        read_timeout=_READ_TIMEOUT_S,
        retries={"max_attempts": 1, "mode": "standard"},
    )
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION, config=config)


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
    # Budget guard: refuse to issue the call if the scenario is out of budget.
    check_budget_before_call()

    client = get_client()
    request = {
        "modelId": model,
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": 0.3},
    }
    for attempt in range(retries):
        t0 = time.monotonic()
        try:
            response = client.converse(**request)
            latency_ms = int((time.monotonic() - t0) * 1000)
            record_success(latency_ms)
            return response["output"]["message"]["content"][0]["text"]
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ThrottlingException", "TooManyRequestsException"):
                wait = 2 ** attempt * 5
                logger.warning(f"Throttled, waiting {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            elif code == "ValidationException":
                logger.error(f"Bedrock validation error: {e}")
                record_failure()
                raise
            elif code == "AccessDeniedException":
                record_failure()
                raise PermissionError(
                    f"Bedrock access denied for model {model}. "
                    f"Check IAM role has bedrock:InvokeModel and the model is enabled."
                ) from e
            else:
                logger.error(f"Bedrock error on attempt {attempt+1}: {e}")
                if attempt == retries - 1:
                    record_failure()
                    raise
                time.sleep(2 ** attempt)
    record_failure()
    raise RuntimeError("Bedrock call failed after all retries")


def _bracket_balanced_slice(text: str) -> str | None:
    """Return the substring from the first '{' or '[' to its matching close.

    Returns None if no opening bracket exists or no balanced slice was found
    (caller can then fall back to a direct json.loads on the trimmed text).
    """
    start = next((i for i, ch in enumerate(text) if ch in "{["), None)
    if start is None:
        return None

    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    # Unbalanced: return everything from the first bracket so the caller can
    # still attempt a parse (json.loads will raise its own diagnostic error).
    return text[start:]


def _candidate_blocks(raw: str) -> list[str]:
    """Enumerate candidate JSON substrings from a raw model response.

    Order of attempts:
      1. Every fenced code block (```json ...``` or generic ```...```).
      2. The bracket-balanced slice of the full text (handles prose wrappers
         like "Sure, here is the JSON:\n{...}").
      3. The raw stripped text itself (handles bare JSON with no prose).
    """
    text = raw.strip()
    blocks: list[str] = []

    # All fenced blocks, in order.
    for match in re.finditer(
        r"```(?:json|JSON)?[ \t]*\r?\n?(.*?)```", text, re.DOTALL
    ):
        block = match.group(1).strip()
        if block:
            blocks.append(block)

    # Bracket-balanced slice of the full text.
    balanced = _bracket_balanced_slice(text)
    if balanced:
        blocks.append(balanced)

    # Trimmed text as the very last fallback.
    blocks.append(text)
    return blocks


def _extract_json(raw: str) -> str:
    """Pull a JSON value out of a model response.

    Scans every fenced code block, then attempts a bracket-balanced slice of
    the whole response, then falls back to the raw text. Returns the first
    candidate that parses; if nothing parses, returns the most promising
    candidate so the caller's json.loads produces a useful error message.
    """
    candidates = _candidate_blocks(raw)
    for block in candidates:
        try:
            json.loads(block)
            return block
        except json.JSONDecodeError:
            continue
    # No candidate parsed cleanly. Return the first (most promising) one;
    # call_llm_json will surface the JSONDecodeError as a ValueError.
    return candidates[0] if candidates else raw.strip()


def call_llm_json(system: str, user: str, **kwargs) -> Any:
    raw = call_llm(system, user, **kwargs)

    # First, try the extractor's best candidate.
    cleaned = _extract_json(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: try every candidate (some Bedrock responses include extra
    # commentary after a fenced block; the extractor already prefers parseable
    # blocks but we retry explicitly for diagnostics).
    for block in _candidate_blocks(raw):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue

    # Last resort: direct json.loads on the trimmed text.
    try:
        return json.loads(raw.strip())
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


# Abbreviation-aware sentence splitter. Splits on `.`, `!`, `?` followed by
# whitespace or EOL, but refuses to split inside common abbreviations
# ("e.g.", "i.e.", or a capitalized-lowercase-period sequence like "Dr.").
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<![A-Z][a-z]\.)(?<![eE]\.g)(?<![iI]\.e)[.!?](?:\s|$)"
)


def _count_sentences(content: str) -> int:
    """Count sentences in content, abbreviation-aware.

    Floors to 1 if content is non-empty so a single sentence without trailing
    punctuation still counts as one sentence.
    """
    if not content:
        return 0
    parts = [s for s in _SENTENCE_SPLIT_RE.split(content) if len(s.strip()) > 3]
    return max(len(parts), 1)


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

    If validation passes, returns immediately. If it fails, injects specific
    corrective feedback and retries up to max_refine_attempts times.

    The returned dict has an extra non-LLM key `_refinement_meta` with:
      {"attempts": int, "reasons": list[str]}
    so callers (the cell builders in agents.py) can persist telemetry on the
    cell. `attempts` is 0 when the first response passed validation.
    """
    result = call_llm_json(system, user, **kwargs)
    # Cell prompts always ask for a JSON object; if the model returns a list
    # or scalar, fail loudly here instead of later when we try to attach
    # `_refinement_meta` or call `.get()`.
    if not isinstance(result, dict):
        raise ValueError(
            f"call_llm_json_with_refinement expects a JSON object but got "
            f"{type(result).__name__}: {str(result)[:200]}"
        )
    attempts = 0
    reasons: list[str] = []

    for attempt in range(max_refine_attempts):
        issues = []

        content = result.get("content", "").strip()
        if not content:
            issues.append("The 'content' field is empty. Write 3 to 6 sentences.")
        else:
            sentence_count = _count_sentences(content)
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
            result["_refinement_meta"] = {"attempts": attempts, "reasons": reasons}
            return result

        attempts += 1
        # Capture short, structured reasons for telemetry (the full prompt
        # feedback is too verbose to persist on every cell).
        for issue in issues:
            first_line = issue.splitlines()[0]
            reasons.append(first_line)

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

    result["_refinement_meta"] = {"attempts": attempts, "reasons": reasons}
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
    challenger_frame: str,
    main_frame: str,
    challenger_style: str,
    facet: str,
    dimension: str,
) -> str:
    return f"""\
You are the Challenger agent in a cultural conflict analysis system.

YOUR CULTURAL FRAME: {challenger_frame}
THE FRAME YOU ARE CHALLENGING: {main_frame}

YOUR ARGUMENTATION STYLE (set by the mediator, independent of culture):
{challenger_style}

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
# Cycle integration — node updating from implying nodes
# ---------------------------------------------------------------------------

NODE_UPDATE_SYSTEM = """\
You update The Occurring between reasoning cycles.

You receive the current nodes and the implying nodes that mediation produced:
reformulated query, revised objective, updated assumptions, and auxiliary query
assumptions. Fold these findings into the nodes so the next cycle reasons over
sharper inputs instead of the original raw framing.

RULES:
- Only change a node when an implying node actually bears on it. Leave the rest
  byte-for-byte identical.
- Apply assumption updates literally: if mediation says "individuals node:
  remove [X] -> replace with [Y]", do exactly that in that node's assumptions.
- Do not invent new content. Everything you write must trace to an implying node.
- Keep each node's content in the same structural style it already has.
- For every node you touch, set "changed": true. For untouched nodes set false.

Respond ONLY with valid JSON. No preamble. No markdown fences.
"""


def node_update_user(
    occurring_json: str,
    updated_assumptions_json: str,
    wholeness_json: str,
    cycle: int,
) -> str:
    return f"""\
Update The Occurring for cycle {cycle} using the implying nodes below.

CURRENT OCCURRING:
{occurring_json}

UPDATED ASSUMPTIONS (node: remove [X] -> replace with [Y]):
{updated_assumptions_json}

IMPLYING NODES (reformulated query, new objective, query/objective assumptions):
{wholeness_json}

Return this exact JSON. All {NUM_FACETS} nodes required, one per facet, in this order:
{{
  "nodes": [
    {{
      "node_id": "individuals",
      "facet": "individuals",
      "content": "<updated or unchanged content>",
      "assumptions": ["<assumption>", "..."],
      "changed": true
    }},
    {{
      "node_id": "interactional_event",
      "facet": "interactional_event",
      "content": "<updated or unchanged content>",
      "assumptions": ["..."],
      "changed": false
    }},
    {{
      "node_id": "process",
      "facet": "process",
      "content": "<updated or unchanged content>",
      "assumptions": ["..."],
      "changed": false
    }},
    {{
      "node_id": "evidence_and_data",
      "facet": "evidence_and_data",
      "content": "<updated or unchanged content>",
      "assumptions": ["..."],
      "changed": false
    }},
    {{
      "node_id": "concepts_theories_rules",
      "facet": "concepts_theories_rules",
      "content": "<updated or unchanged content>",
      "assumptions": ["..."],
      "changed": false
    }},
    {{
      "node_id": "inferences",
      "facet": "inferences",
      "content": "<updated or unchanged content>",
      "assumptions": ["..."],
      "changed": false
    }}
  ]
}}

Set "changed": true only on nodes you actually rewrote; false on untouched nodes.
For untouched nodes, copy content and assumptions verbatim from the input.
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
    all_findings_json: str,
    evaluation_feedback: str | None = None,
) -> str:
    feedback_block = ""
    if evaluation_feedback:
        feedback_block = f"""\
EVALUATION FEEDBACK TO ADDRESS (from the judge on your previous draft):
{evaluation_feedback}
Revise the document to fix exactly these issues. Do not introduce new claims.

"""
    return f"""\
Write the final conflict analysis document.
Use only what mediation found. Do not introduce new claims.

{feedback_block}THE OCCURRING:
{occurring_json}

IMPLICATIONS GATHERED ACROSS THE ENTIRE MEDIATION CYCLE:
{all_findings_json}

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


def judge_cycle_user(mediation_json: str, cycle: int) -> str:
    """In-loop evaluation: no final output exists yet, score the mediation only."""
    return f"""\
Evaluate the mediation findings for cycle {cycle}. There is no final document yet,
so judge the quality of the mediation reasoning itself on the same four dimensions.

MEDIATION FINDINGS:
{mediation_json}

Return:
{{
  "reasoning": "<step by step evaluation of this cycle's mediation>",
  "empathy": <1-5>,
  "consensus": <1-5>,
  "escalation_risk": <1-5>,
  "cultural_insight": <1-5>
}}
"""

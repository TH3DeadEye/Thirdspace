import json
import os
import time
import logging
from typing import Any

# AWS SDK
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

#setting up AWS model and region
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-sonnet-4-20250514-v1:0"
)

BEDROCK_REGION = os.environ.get("AWS_REGION", "us-east-1")

#create a AWS client 
def get_client():
    return boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)


#calling bedrock model 
def call_llm(
    system: str,
    user: str,
    model: str = BEDROCK_MODEL_ID,
    max_tokens: int = 8000,
    retries: int = 3,
) -> str:
    client = get_client()
    #Bedrock converse API expects a specific request format
    request = {
        "modelId": model,
        "system": [{"text": system}],
        "messages": [
            {"role": "user", "content": [{"text": user}]}
        ],
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": 0.3,   # lower = more consistent structured output
        },
    }
    #rate limiting and error handeling
    for attempt in range(retries):
        try:
            response = client.converse(**request)
            return response["output"]["message"]["content"][0]["text"]

        except ClientError as e:
            code = e.response["Error"]["Code"]
            
            if code in ("ThrottlingException", "TooManyRequestsException"):
                wait = 2 ** attempt * 5
                logger.warning(
                    f"Bedrock throttled, waiting {wait}s "
                    f"(attempt {attempt+1}/{retries})"
                )
                time.sleep(wait)

            elif code == "ValidationException":
                # Bad request, no point retrying
                logger.error(f"Bedrock validation error: {e}")
                raise

            elif code == "AccessDeniedException":
                raise PermissionError(
                    f"Bedrock access denied for model {model}. "
                    "Check IAM role has bedrock:InvokeModel permission "
                    "and the model is enabled in your Bedrock console."
                ) from e

            else:
                logger.error(f"Bedrock error on attempt {attempt+1}: {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

    raise RuntimeError("Bedrock call failed after all retries")

#helper function to call LLM and parse JSON output
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


# PHASE 1 — REFORMULATION

REFORMULATION_SYSTEM = """\
You are a scenario decomposition specialist. Your only job is to take a raw, \
emotionally charged, one-sided description of a cultural conflict and break it \
into exactly six structured nodes called The Occurring.

WHAT THE OCCURRING IS:
A conflict is not one thing. It is several independent things happening simultaneously.
The Occurring separates them so they can be reasoned about independently.

THE SIX NODES — definitions and what each one must contain:

NODE 1: individuals
Who is involved. The critical rule: describe each person as they are ALREADY AFFECTED
by the situation before they have even responded to it.
This means: do not describe them neutrally. Describe them as someone already shaped by
their cultural background, their relational history with the other person, and the
event itself. What assumptions are they already carrying into this moment?
Example of wrong framing: "Person A is a manager. Person B is an employee."
Example of right framing: "Person A enters this moment already interpreting silence
as a sign of disengagement, shaped by a professional culture that treats
non-response as passive resistance."

NODE 2: interactional_event
The specific moment of friction. The concrete thing that happened.
Strip out ALL interpretation. Strip out ALL backstory. Strip out ALL labels like
"rude", "disrespectful", "weird", "aggressive" — those belong in concepts node.
Just the observable event. What would a camera have recorded?

NODE 3: process
The timeline and context AROUND the event.
How did things unfold? What happened before? Is this being experienced as an
isolated incident or as part of a pattern? What is each person's model of time
here — is this a one-off moment or a continuation of something?
Cultural conflicts often hinge on process disagreements MORE than the event itself.

NODE 4: evidence_and_data
ONLY what can actually be verified. This node is about separating fact from assumption.
For every claim in the scenario, ask: is this something that was directly observable,
or is it an interpretation? Put interpretations in inferences. Put frameworks in concepts.
This node gets only the verifiable residue.
Be explicit when evidence is thin: "The only verifiable fact is X. Everything else
in the scenario is interpretation."

NODE 5: concepts_theories_rules
The cultural frameworks, communication norms, hierarchy assumptions, and interpretive
rules that each party is applying — usually without knowing it.
Name them explicitly for both sides. Do not describe behavior here — name the underlying
rule that makes that behavior make sense within each culture.
Example: not "Person A expected a response" but "Person A is operating under a
low-context communication norm where explicit acknowledgment signals receipt."

NODE 6: inferences
What can actually be deduced from nodes 1-5.
NOT what the narrator believes happened. NOT the narrator's conclusion.
What does the STRUCTURE of the decomposition itself imply happened?
This node often contradicts the narrator's framing. That is correct.

CRITICAL RULES:
- Do not echo the narrator's language or labels. Rephrase everything structurally.
- Each node must contain something the narrator did NOT explicitly state.
- Each node carries a list of explicit assumptions. Name them.
- Thin scenarios still get all 6 nodes. Use the structure to surface what is missing.
- You are not summarizing. You are decomposing.

Respond ONLY with valid JSON. No preamble. No markdown fences.
"""

#helper functio to rewrite the user raw scanerio into the reformulation prompt with instructions
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
      "content": "<description of each person as already affected — not neutral>",
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
      "content": "<verifiable facts only — everything else named as assumption>",
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



# PHASE 2 — MAIN AGENT

def critical_thinking_system(cultural_frame: str) -> str:
    return f"""\
You are the primary reasoning agent in a cultural conflict analysis system.

YOUR CULTURAL FRAME: {cultural_frame}

You will receive The Occurring — six structured nodes describing a cultural conflict.
For each node, you must reason across seven critical thinking dimensions.
This produces a 6x7 grid of 42 cells. Every cell must be filled.

YOUR CULTURAL FRAME RULE:
You reason from {cultural_frame}. This is not a flaw — it is your function.
But you must be explicit about it. Whenever your cultural frame is shaping
your reasoning, name it. Do not let {cultural_frame} assumptions appear as
neutral observations. Mark them as what they are.
Example of wrong: "The silence was a failure to communicate."
Example of right: "From {cultural_frame}, silence in response to a direct
question signals disengagement. This interpretation depends on a low-context
communication norm where explicit acknowledgment is expected."

THE SEVEN DIMENSIONS — what to write in each cell:

OBJECTIVE
What is the actual goal of reasoning about THIS specific node?
Is that goal reasonable and answerable given what we know?
Does your reasoning stay focused on that goal, or does it drift?

QUERY_FORMULATION
Is the question framed correctly for this node?
Can you break it into smaller sub-questions?
Does this node have one defensible answer, or multiple reasonable interpretations?

PERSPECTIVE
What viewpoint is guiding your reasoning about this node?
Name the intuitions behind that viewpoint.
Where does YOUR perspective break down or have blind spots?

ASSUMPTIONS
What are you taking for granted when reasoning about this node?
Which assumptions are load-bearing — if wrong, your whole interpretation collapses?
Be specific: "I am assuming that..." not "one might assume that..."

EVIDENCE
Which facts from the scenario actually support this reasoning?
Is there another explanation that fits the same evidence equally well?
Rate evidence quality honestly: strong, moderate, weak, or absent.

CONCEPTS
Which cultural frameworks or norms are you applying to this node?
Name them precisely.
CRITICAL CHECK: Have you quietly adjusted any assumption to make this node fit
a preferred framework? If yes, name it explicitly — this is the most common
failure mode and you must catch it yourself.

IMPLICATIONS
If your reasoning about this node is correct, what specifically follows?
What would need to be true about the situation for your reasoning to be right?

SELF-EVALUATION — run before writing each cell:
  clarity, correctness, precision, proportion, depth,
  comprehensiveness, logic, fairness
If a cell fails one measure, record it in reflection_flags.

FORMATTING:
- Every cell: minimum 3 sentences, maximum 6.
- reflection_flags is [] if the cell passes all measures.
  If it fails: {{"measure": "name", "issue": "what specifically failed"}}

Respond ONLY with valid JSON. No preamble. No markdown fences.
"""
#helper function to analyze the scenario with the 6 nodes and 7 deementsions and rewrite it into the critical thinking prompt with instructions
def critical_thinking_user(occurring_json: str, cycle: int) -> str:
    return f"""\
Apply critical thinking across all 6 nodes. Cycle {cycle}.

THE OCCURRING:
{occurring_json}

Return exactly 42 cells — one for every facet × dimension combination.
Facets: individuals, interactional_event, process, evidence_and_data,
        concepts_theories_rules, inferences
Dimensions: objective, query_formulation, perspective, assumptions,
            evidence, concepts, implications

{{
  "cells": [
    {{
      "facet": "<facet name>",
      "dimension": "<dimension name>",
      "content": "<your reasoning — 3 to 6 sentences>",
      "reflection_flags": []
    }},
    ...42 total
  ]
}}
"""


# PHASE 3 — CHALLENGER AGENT

def challenger_system(challenger_cultural_frame: str, main_cultural_frame: str) -> str:
    return f"""\
You are the Challenger agent in a cultural conflict analysis system.

YOUR CULTURAL FRAME: {challenger_cultural_frame}
THE FRAME YOU ARE CHALLENGING: {main_cultural_frame}

You will receive The Occurring and the main agent's 42-cell grid.
Produce an identical 42-cell grid from {challenger_cultural_frame}.
For every cell, engage with what the main agent said.

THE MOST IMPORTANT CELLS:
Find cells where the main agent treated a norm specific to {main_cultural_frame}
as if it were a universal human fact. When you find one, be explicit:
"The main agent's reasoning depends on [specific norm from {main_cultural_frame}]
being treated as universal. From {challenger_cultural_frame}, [explain what
this norm looks like and why it does not apply here]."

THREE RESPONSE TYPES per cell:
TYPE 1 — CHALLENGE: the main agent applied {main_cultural_frame} norms incorrectly.
TYPE 2 — REFRAME: not wrong, but asking the wrong question. The better question is X.
TYPE 3 — CONFIRM: holds from your frame too. Explain WHY from {challenger_cultural_frame}.

RULES:
- Do not hedge into "both perspectives have merit" without specific grounding.
- Maintain {challenger_cultural_frame} consistently across all 42 cells.
- Every cell: 3 to 6 sentences. Same reflection_flags structure.

Respond ONLY with valid JSON. No preamble. No markdown fences.
"""
#return the challenger prompt with the main agent's grid and the occurring for the challenger to analyze and respond to
def challenger_user(occurring_json: str, main_grid_json: str, cycle: int) -> str:
    return f"""\
Challenge the main agent's grid from your cultural frame. Cycle {cycle}.

THE OCCURRING:
{occurring_json}

MAIN AGENT'S GRID:
{main_grid_json}

Return 42 cells. Same JSON structure as the main agent.
"""



# PHASE 4 — MEDIATION

MEDIATION_SYSTEM = """\
You are the mediation engine in a cultural conflict analysis system.

You receive two 42-cell grids and The Occurring they were both reasoning about.
Compare all 42 cell pairs. For each: agreement or conflict?

For conflicts — the IMPLICATION field is critical:
Do not write "they disagree about silence."
Write what the disagreement REVEALS about the conflict.
Complete this sentence: "This reveals that [structural feature of the conflict
that is invisible from inside either cultural frame alone]."

Weak implication (do not write): "The two agents have different views on communication."
Strong implication (write this): "This reveals that both parties believed they were
being clear — the breakdown happened at the layer of what being clear means,
not at the layer of intention or information."

PATTERN-LEVEL SYNTHESIS:
After all 42 pairs, look at where conflicts cluster.
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
#return the mediator's response with the main agent's grid, the challenger grid, the occurring and the previous mediation output for the mediator to analyze and respond to
def mediation_user(
    occurring_json: str,
    main_grid_json: str,
    challenger_grid_json: str,
    previous_mediation_json: str,
    cycle: int,
) -> str:
    return f"""\
Mediate between the two grids. Cycle {cycle}.

THE OCCURRING:
{occurring_json}

MAIN AGENT GRID:
{main_grid_json}

CHALLENGER GRID:
{challenger_grid_json}

PREVIOUS MEDIATION (null on cycle 1):
{previous_mediation_json}

Return:
{{
  "cycle": {cycle},
  "agreements": ["<what both agents agreed on>"],
  "conflicts": [
    {{
      "facet": "<facet>",
      "dimension": "<dimension>",
      "main_content": "<brief summary of main agent>",
      "challenger_content": "<brief summary of challenger>",
      "implication": "<what this disagreement REVEALS — not just what they disagree on>"
    }}
  ],
  "what_conflicts_imply": ["<pattern-level finding>"],
  "updated_assumptions": ["<node>: remove [X] → replace with [Y]"],
  "new_objective": "<revised framing or null>",
  "reformulated_query": "<clearer question or null>",
  "converged": true or false
}}
"""


# PHASE 5 — CONSTRAINED GENERATION

CONSTRAINED_GENERATION_SYSTEM = """\
You are the final output writer in a cultural conflict analysis system.

YOUR CONSTRAINT:
Every claim must trace back to something the mediation process found.
No new inferences. No advice from training data. No generic cultural wisdom.
If mediation did not produce enough to ground a specific claim, say so.

FIVE SECTIONS:

WHAT ACTUALLY HAPPENED
The mediation's version, not the user's version.
Describe the conflict as structural decomposition and two rounds of cultural
reasoning understood it. Be direct where the user's framing was incomplete.
2-3 paragraphs.

WHERE THE CULTURES DIVERGED
For each significant conflict from mediation:
- What each cultural frame expected at that moment
- Why neither expectation is wrong — just different
- What the specific misread was
Choose the conflicts with the strongest implications. Quality over quantity.

WHAT THE CONFLICT ACTUALLY IMPLIES
Use what_conflicts_imply from mediation.
These are the structural features neither party can see from inside their own frame.
Be direct. Do not soften findings into "both sides have a point."
2-3 paragraphs.

WHAT EACH PERSON LIKELY EXPERIENCED
Based on the individuals node and conflict findings:
What did each person feel was happening to them?
What did they assume the other person was doing, and why?
Write with empathy for both — not blame, not excuse.
One paragraph per person.

HOW TO MOVE FORWARD
Grounded only in updated_assumptions and new_objective from mediation.
If the conflict is structural and cannot be bridged by direct conversation alone,
say that — do not offer "have an honest conversation" as a solution to a problem
that is exactly about what counts as honest communication.
2-3 paragraphs.

Write in plain language for someone trying to understand a real situation.
Be direct. Uncomfortable truths stated clearly are more useful than comfortable generalities.
"""
#returns the final output using the all the agent's outputs and the occurring for the final output writer to analyze and respond to
def constrained_generation_user(
    occurring_json: str,
    all_mediation_json: str,
    final_mediation_json: str,
) -> str:
    return f"""\
Write the final conflict analysis document.
Use only what mediation found. Do not introduce new claims.

THE OCCURRING:
{occurring_json}

ALL MEDIATION CYCLES:
{all_mediation_json}

FINAL MEDIATION OUTPUT (most weight):
{final_mediation_json}

Write the five-section document now.
"""
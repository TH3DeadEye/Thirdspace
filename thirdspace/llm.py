import json
import os
import time
import logging
from typing import Any
from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from budget import (
    check_budget_before_call, record_success, record_failure,
)
from utils.parsing import _candidate_blocks, _count_sentences

logger = logging.getLogger(__name__)
#main agent LLM model 
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)

#challenger agent LLM model (it should be different from main)
CHALLENGER_MODEL_ID = os.environ.get(
    "CHALLENGER_MODEL_ID",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
)

#judge agent LLM model
JUDGE_MODEL_ID = os.environ.get(
    "JUDGE_MODEL_ID",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)

#embedding model (convergence similarity between cycles). Titan Text Embeddings
# V2 is invoked via invoke_model, not converse.
EMBED_MODEL_ID = os.environ.get(
    "BEDROCK_EMBED_MODEL_ID",
    "amazon.titan-embed-text-v2:0",
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

def _annotate_llm_run(
    model: str,
    usage: dict,
    temperature: float,
    max_tokens: int,
    latency_ms: int,
) -> None:
    """Attach model identity and token usage to the active LangSmith run.

    No-op when tracing is disabled (get_current_run_tree() returns None), so
    call_llm can call it unconditionally. ls_model_name / ls_provider let
    LangSmith map Bedrock token counts onto its cost table; usage_metadata
    turns the run into a first-class LLM call with input/output token counts.
    Telemetry must never break a real call, so failures are swallowed.
    """
    run = get_current_run_tree()
    if run is None:
        return
    try:
        run.set(
            metadata={
                "ls_provider": "amazon_bedrock",
                "ls_model_name": model,
                "ls_temperature": temperature,
                "ls_max_tokens": max_tokens,
                "latency_ms": latency_ms,
            },
            usage_metadata={
                "input_tokens": usage.get("inputTokens", 0),
                "output_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            },
        )
    except Exception:
        logger.debug("LangSmith usage annotation failed", exc_info=True)


@traceable(name="bedrock_call", run_type="llm")
def call_llm(
    system: str,
    user: str,
    model: str = BEDROCK_MODEL_ID,
    max_tokens: int = 8000,
    retries: int = 3,
    temperature:float = 0.3,
) -> str:
    # Budget guard: refuse to issue the call if the scenario is out of budget.
    check_budget_before_call()

    client = get_client()
    request = {
        "modelId": model,
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
    }
    for attempt in range(retries):
        t0 = time.monotonic()
        try:
            response = client.converse(**request)
            latency_ms = int((time.monotonic() - t0) * 1000)
            text = response["output"]["message"]["content"][0]["text"]
            if not text.strip():
                logger.warning(
                    f"Model returned empty response on attempt {attempt+1}/{retries}. "
                    f"Retrying."
                )
                time.sleep(2 ** attempt)
                continue
            if response.get("stopReason") == "max_tokens":
                # Truncated mid-output: the JSON is almost certainly incomplete
                # and will fail to parse downstream. Surface it here so the
                # cause is obvious rather than a cryptic JSON error.
                logger.warning(
                    f"Response hit maxTokens={max_tokens} and was truncated; "
                    f"downstream JSON parsing may fail. Raise max_tokens if this recurs."
                )
            record_success(latency_ms)
            _annotate_llm_run(
                model, response.get("usage", {}), temperature, max_tokens, latency_ms
            )
            return text
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
            elif code == "ResourceNotFoundException":
                record_failure()
                raise PermissionError(
                    f"Model '{model}' not enabled in your AWS account. "
                    f"Go to AWS Console → Bedrock → Model access → enable Anthropic Claude."
                ) from e
            else:
                logger.error(f"Bedrock error on attempt {attempt+1}: {e}")
                if attempt == retries - 1:
                    record_failure()
                    raise
                time.sleep(2 ** attempt)
    record_failure()
    raise RuntimeError(
        f"Bedrock call failed after {retries} retries "
        f"(model={model}). Last response was empty or all attempts errored."
    )


@traceable(name="bedrock_embed", run_type="embedding")
def embed_text(
    text: str,
    model: str = EMBED_MODEL_ID,
    retries: int = 3,
) -> list[float]:
    """Embed a single string via Bedrock and return the dense vector.

    Used by the convergence metric to compare the semantic content of two
    cycles' findings. Reuses the singleton bedrock-runtime client and the same
    budget / circuit-breaker hooks as call_llm so embedding calls count toward
    the scenario budget. Titan Text Embeddings V2 is invoked via invoke_model
    (converse does not support embedding models).
    """
    check_budget_before_call()

    client = get_client()
    body = json.dumps({"inputText": text})
    for attempt in range(retries):
        t0 = time.monotonic()
        try:
            response = client.invoke_model(modelId=model, body=body)
            latency_ms = int((time.monotonic() - t0) * 1000)
            payload = json.loads(response["body"].read())
            embedding = payload.get("embedding")
            if not embedding:
                logger.warning(
                    f"Embedding model returned no vector on attempt "
                    f"{attempt+1}/{retries}. Retrying."
                )
                time.sleep(2 ** attempt)
                continue
            record_success(latency_ms)
            return embedding
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ThrottlingException", "TooManyRequestsException"):
                wait = 2 ** attempt * 5
                logger.warning(
                    f"Embedding throttled, waiting {wait}s "
                    f"(attempt {attempt+1}/{retries})"
                )
                time.sleep(wait)
            elif code == "ValidationException":
                logger.error(f"Bedrock embedding validation error: {e}")
                record_failure()
                raise
            elif code == "AccessDeniedException":
                record_failure()
                raise PermissionError(
                    f"Bedrock access denied for embedding model {model}. "
                    f"Check IAM role has bedrock:InvokeModel and the model is enabled."
                ) from e
            elif code == "ResourceNotFoundException":
                record_failure()
                raise PermissionError(
                    f"Embedding model '{model}' not enabled in your AWS account. "
                    f"Go to AWS Console → Bedrock → Model access → enable Amazon Titan."
                ) from e
            else:
                logger.error(f"Bedrock embedding error on attempt {attempt+1}: {e}")
                if attempt == retries - 1:
                    record_failure()
                    raise
                time.sleep(2 ** attempt)
    record_failure()
    raise RuntimeError(
        f"Bedrock embedding call failed after {retries} retries (model={model})."
    )


def call_llm_json(system: str, user: str, **kwargs) -> Any:
    raw = call_llm(system, user, **kwargs)

    # _candidate_blocks yields the most-promising substrings first (fenced
    # blocks, then a bracket-balanced slice, then the raw text). strict=False
    # tolerates the literal newlines/tabs Claude routinely emits inside string
    # values — e.g. multi-line numbered lists in a cell's "content". Strict
    # JSON rejects those as control characters, which was the cause of spurious
    # "not valid JSON" failures.
    first_error: json.JSONDecodeError | None = None
    for block in _candidate_blocks(raw):
        try:
            return json.loads(block, strict=False)
        except json.JSONDecodeError as e:
            if first_error is None:
                # Keep the first (best) candidate's error for diagnostics,
                # rather than the trailing raw-text "Expecting value" artifact.
                first_error = e
            continue

    logger.error(
        "Model response was not valid JSON (length=%d chars). If the snippet "
        "below looks cut off, the call likely hit maxTokens and was truncated.\n%s",
        len(raw), raw[:1500],
    )
    raise ValueError(
        f"Model response was not valid JSON: {first_error}"
    ) from first_error


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


@traceable(run_type="chain", name="cell_refinement")
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

import json
import os
import random
import time
import logging
from typing import Any
from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from json_repair import repair_json
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from budget import (
    check_budget_before_call, record_success, record_failure,
)
from utils.parsing import _candidate_blocks, _count_sentences

logger = logging.getLogger(__name__)
#main agent LLM model. 
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    
)

#challenger agent LLM model (it should be different from main)
CHALLENGER_MODEL_ID = os.environ.get(
    "CHALLENGER_MODEL_ID",
    
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
    retries: int = 6,
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
            usage = response.get("usage", {})
            record_success(latency_ms, usage)
            _annotate_llm_run(
                model, usage, temperature, max_tokens, latency_ms
            )
            return text
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ThrottlingException", "TooManyRequestsException"):
                # Exponential backoff capped at 60s, plus full jitter. The cell
                # agents fire many calls at once; without jitter their retries
                # re-synchronize and slam the same per-minute quota together.
                wait = min(60, 2 ** attempt * 5) + random.uniform(0, 5)
                logger.warning(
                    f"Throttled, waiting {wait:.1f}s (attempt {attempt+1}/{retries})"
                )
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
            record_success(
                latency_ms,
                {"inputTokens": payload.get("inputTextTokenCount", 0) or 0},
            )
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

    # Strict parsing failed on every candidate. Fall back to json_repair, a
    # tolerant parser that fixes the malformations Claude actually produces but
    # strict JSON cannot: unescaped inner quotes (e.g. echoing a phrase like
    # "green light" inside a content string), trailing commas, and truncated
    # tails. We only reach here on genuinely broken output, so a heuristic
    # recovery is strictly better than crashing the whole run.
    for block in _candidate_blocks(raw):
        try:
            repaired = repair_json(block, return_objects=True)
        except Exception:
            continue
        # Our prompts always return an object or array; a non-empty dict/list
        # means json_repair recovered something usable. Anything else (""/{}/[])
        # is a failed recovery — try the next candidate.
        if isinstance(repaired, (dict, list)) and repaired:
            logger.warning(
                "Strict JSON parse failed (%s); recovered with json_repair.",
                first_error,
            )
            return repaired

    logger.error(
        "Model response was not valid JSON and json_repair could not recover it "
        "(length=%d chars). First 1500 chars:\n%s",
        len(raw), raw[:1500],
    )
    raise ValueError(
        f"Model response was not valid JSON: {first_error}"
    ) from first_error


# ---------------------------------------------------------------------------
# Context engineering utilities (from Omid's context engineering update)
# ---------------------------------------------------------------------------

# Refinement retry cap. Retries fire only on structural defects (empty or
# too-short content), which are rare — reflection_flags are telemetry, not
# failures, so they never trigger a retry. Tune via env if needed.
MAX_REFINE_ATTEMPTS = int(os.environ.get("THIRDSPACE_REFINE_ATTEMPTS", "2"))


def _structural_issues(result: dict) -> list[str]:
    """Objective defects that justify paying for a retry call.

    Deliberately narrow: an honest model asked to self-criticize will always
    find something, so populated reflection_flags are NOT treated as failures
    (run-june-10-1: 126/126 cells retried on their own honesty, 49% of the
    run's calls, and 155 cells still carried flags afterwards). Flags are kept
    on the result as downstream context instead.
    """
    content = str(result.get("content", "") or "").strip()
    if not content:
        return ["The 'content' field is empty. Write 3 to 6 sentences."]
    sentence_count = _count_sentences(content)
    if sentence_count < 3:
        return [
            f"Content has only {sentence_count} sentence(s). "
            f"Write at least 3 sentences."
        ]
    return []


@traceable(run_type="chain", name="cell_refinement")
def call_llm_json_with_refinement(
    system: str,
    user: str,
    max_refine_attempts: int = MAX_REFINE_ATTEMPTS,
    **kwargs,
) -> Any:
    """
    Wraps call_llm_json with a structural validation and retry loop.

    Used for single-cell calls (CT agent and Challenger).
    Each call returns {"content": "...", "reflection_flags": [...], ...}.

    Retries ONLY on objective structural defects (empty / too-short content —
    see _structural_issues). reflection_flags are self-reported nuance, not
    defects: they stay on the result untouched and never trigger a retry.

    The retry prompt is context-complete: it contains the ORIGINAL user prompt
    (The Occurring etc.), the model's previous response, and the feedback —
    a bare "fix it" retry regenerates from the system prompt alone, with no
    scenario in front of the model.

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

    for _ in range(max_refine_attempts):
        issues = _structural_issues(result)
        if not issues:
            break

        attempts += 1
        reasons.extend(issues)
        logger.warning(
            f"[Refinement attempt {attempts}] {len(issues)} structural issue(s), retrying."
        )
        feedback = "\n".join(issues)

        refine_user = (
            f"{user}\n\n"
            f"--- YOUR PREVIOUS RESPONSE ---\n"
            f"{json.dumps(result, ensure_ascii=False)}\n\n"
            f"--- VALIDATION FEEDBACK ---\n"
            f"{feedback}\n\n"
            f"Rewrite your previous response so it fixes the feedback while "
            f"keeping everything that was already correct. Write 3 to 6 "
            f"sentences. Return valid JSON in the same format."
        )
        result = call_llm_json(system, refine_user, **kwargs)
        if not isinstance(result, dict):
            raise ValueError(
                f"call_llm_json_with_refinement expects a JSON object but got "
                f"{type(result).__name__}: {str(result)[:200]}"
            )

    result["_refinement_meta"] = {"attempts": attempts, "reasons": reasons}
    return result

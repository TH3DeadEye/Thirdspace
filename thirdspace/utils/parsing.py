"""Pure helpers for turning raw model output into parseable JSON / sentences.
No Bedrock, no project imports — safe to use anywhere."""
import json
import re


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
            # strict=False so a block is accepted even when it contains the
            # literal newlines/tabs Claude often emits inside string values.
            json.loads(block, strict=False)
            return block
        except json.JSONDecodeError:
            continue
    # No candidate parsed cleanly. Return the first (most promising) one;
    # call_llm_json will surface the JSONDecodeError as a ValueError.
    return candidates[0] if candidates else raw.strip()


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

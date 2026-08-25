"""Prompts that adapt ThirdSpace output into a dataset's label vocabulary.

Two prompts live here, and the difference between them is the entire
experiment:

  READOUT   sees only what the pipeline produced (The Occurring + the final
            mediation). It never sees the original story text as written, and
            never sees the gold label or the dataset's explanation field.
  BASELINE  sees the raw record and nothing else. It is the control — the
            number that says whether the pipeline earned its ~60 calls.

Both are told the exact label set, which is discovered from the sampled data
rather than hard-coded, so the same code works whether MDD says
"moral/immoral" or "ethical/unethical/grey area".
"""
from __future__ import annotations

import json

# Label glosses. Keys are canonical buckets; the dataset's own word for each
# bucket is substituted in at render time.
NORMAD_GLOSS = {
    "acceptable": "the behaviour fits the social norms of that cultural context",
    "unacceptable": "the behaviour breaches the social norms of that cultural context",
    "contested": ("the behaviour is not governed by a norm of that context either way — "
                  "the culture takes no position on it"),
}

MDD_GLOSS = {
    "acceptable": "the active agent's action was morally justified in this case",
    "unacceptable": "the active agent's action was morally wrong in this case",
    "contested": ("the case sits in a genuine grey area — the parameters do not "
                  "settle it in either direction"),
}

GLOSSES = {"normad": NORMAD_GLOSS, "mdd": MDD_GLOSS}


def _label_block(vocab, dataset: str) -> str:
    gloss = GLOSSES.get(dataset, {})
    lines = []
    for label in vocab.labels:
        canon = vocab.canon.get(label)
        note = gloss.get(canon or "", "")
        lines.append(f'  "{label}"' + (f" — {note}" if note else ""))
    return "\n".join(lines)


READOUT_SYSTEM = """\
You convert the output of a structured conflict-analysis pipeline into a single
label.

WHAT YOU ARE READING. A scenario was decomposed into five object nodes (The
Occurring), read twice through two different cultural standpoints, and the two
readings were compared cell by cell. You receive the decomposition, the points
the two readings agreed on, the points where they conflicted, and the
mediator's synthesis. You do NOT receive the original text and you do not need
it — decide from the analysis in front of you.

HOW TO DECIDE.
- Agreement between the two readings that the behaviour fits what the context
  expects is evidence FOR acceptability, not merely evidence that the analysis
  was easy.
- A high-severity conflict about VALUE or ASSUMPTION, typed normative or
  identity_related, means the two standpoints disagree about whether the
  behaviour was right. Ask which standpoint the label is defined by — for a
  cultural-norm dataset, the verdict is that of the context the scenario is
  set in, not the outsider's.
- Conflicts that are factual or interpretive, or that are low severity, say
  the analysis found ambiguity — they are not by themselves a verdict against
  the behaviour.
- A conflict named "unresolvable" reports that no remedy exists, not that the
  behaviour was wrong.

Pick exactly one label from the allowed set. Do not invent a label, do not
hedge across two, and do not answer in a synonym. If the analysis genuinely
does not settle the question, pick the allowed label that means "unsettled" if
one exists, and otherwise pick the better-supported of the remaining labels.

Respond ONLY with valid JSON. No preamble, no markdown fences:
{
  "label": "<exactly one of the allowed labels>",
  "confidence": <number between 0 and 1>,
  "rationale": "<at most two sentences>",
  "evidence": ["<conflict id or agreement you relied on>"]
}
"""


def readout_user(record, digest: dict, signals: dict, vocab) -> str:
    """The user turn: analysis artifacts, the question, and the label set."""
    shape = {
        "conflicts": signals.get("n_conflicts"),
        "agreements": signals.get("n_agreements"),
        "mean_severity": round(signals.get("mean_severity", 0.0), 2),
        "dominant_conflict_type": signals.get("dominant_type"),
        "type_fractions": {k: round(v, 2)
                           for k, v in (signals.get("type_fractions") or {}).items()},
        "converged": signals.get("converged"),
    }
    doc = digest.get("final_document")
    doc_block = f"\nTHE ANALYSIS WRITTEN FOR THE PARTIES:\n{doc}\n" if doc else ""

    return f"""\
QUESTION TO ANSWER:
{record.question}

THE STANDPOINTS THE TWO READINGS WERE PRODUCED FROM:
  reading 1 (main): {record.main_frame or "no assigned frame — neutral analyst"}
  reading 2 (challenger): {record.challenger_frame or "no assigned frame — neutral skeptic"}

THE SITUATION, AS DECOMPOSED:
{json.dumps(digest.get("occurring"), ensure_ascii=False)}

WHAT BOTH READINGS AGREED ON:
{json.dumps(digest.get("agreements"), ensure_ascii=False)}

WHERE THEY CONFLICTED (highest severity first):
{json.dumps(digest.get("conflicts"), ensure_ascii=False)}

THE MEDIATOR'S SYNTHESIS:
{json.dumps(digest.get("wholeness"), ensure_ascii=False)}

SHAPE OF THE DISAGREEMENT:
{json.dumps(shape, ensure_ascii=False)}
{doc_block}
ALLOWED LABELS — answer with one of these strings exactly:
{_label_block(vocab, record.dataset)}

Return the JSON object only."""


BASELINE_SYSTEM = """\
You judge a single case and return one label. No analysis, no explanation
beyond one sentence. You are a control condition in an experiment: answer
directly from what you are given.

Respond ONLY with valid JSON. No preamble, no markdown fences:
{"label": "<exactly one of the allowed labels>", "rationale": "<one sentence>"}
"""


def baseline_user(record, vocab) -> str:
    frame_line = (
        f"CULTURAL CONTEXT: {record.main_frame}\n" if record.main_frame else "")
    return f"""\
{frame_line}SITUATION:
{record.scenario}

QUESTION:
{record.question}

ALLOWED LABELS — answer with one of these strings exactly:
{_label_block(vocab, record.dataset)}

Return the JSON object only."""

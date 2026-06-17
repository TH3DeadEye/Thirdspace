import math

from models import ImplyingNodes
from llm import embed_text


def conflict_jaccard(prev: ImplyingNodes, curr: ImplyingNodes) -> float:
    #Jaccard overlap of the (facet, dimension) conflist sets of two cycles
    #first cycle: identify conflict cells (fully stable)
    #disjoints ( if every conflict is new)


    prev_set = {(c.facet, c.dimension) for c in prev.conflicts}
    curr_set = {(c.facet, c.dimension) for c in curr.conflicts}

    if not prev_set and not curr_set:
        return 1.0
    return len(prev_set & curr_set) / len(prev_set | curr_set)


# ---------------------------------------------------------------------------
# Embedding-based convergence — content-aware alternative to Jaccard.
#
# Jaccard only sees WHICH (facet, dimension) cells fired, not WHAT they found.
# The embedding metric compares the semantic substance of two cycles' findings,
# so the same idea migrating to a different cell no longer reads as a brand-new
# conflict (and a totally rewritten conflict on the same cell no longer reads as
# "stable"). Both metrics return [0, 1] with 1.0 = stable, so they're
# interchangeable behind `convergence_similarity`.
# ---------------------------------------------------------------------------

def conflict_signature(m: ImplyingNodes) -> str:
    """One string capturing the substance of a cycle's findings.

    Built from each conflict's `implication` (the "This reveals that..." insight)
    plus the pattern-level `what_conflicts_imply`. Empty parts are skipped so a
    cycle with no conflicts yields "" (which short-circuits the similarity).
    """
    parts = [c.implication for c in m.conflicts] + list(m.what_conflicts_imply)
    return "\n".join(p for p in parts if p and p.strip())


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors. Pure Python, no numpy."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def conflict_embedding_similarity(
    prev: ImplyingNodes,
    curr: ImplyingNodes,
    embed_fn=None,
) -> float:
    """Cosine similarity of the two cycles' embedded conflict signatures.

    `embed_fn` is injectable so tests can stub it (no Bedrock traffic); when not
    passed it resolves the module-level `embed_text` at call time, so
    monkeypatching `metrics.embed_text` also works. When both cycles produced an
    empty signature we return 1.0 to match `conflict_jaccard`'s empty-set
    contract (and stay offline) without embedding anything.
    """
    sig_prev = conflict_signature(prev)
    sig_curr = conflict_signature(curr)
    if not sig_prev and not sig_curr:
        return 1.0
    if not sig_prev or not sig_curr:
        return 0.0
    fn = embed_fn if embed_fn is not None else embed_text
    return _cosine(fn(sig_prev), fn(sig_curr))


def convergence_similarity(
    prev: ImplyingNodes,
    curr: ImplyingNodes,
    metric: str = "embedding",
) -> float:
    """Dispatch to the configured convergence metric. Both return [0, 1]."""
    if metric == "jaccard":
        return conflict_jaccard(prev, curr)
    return conflict_embedding_similarity(prev, curr)
"""Context-engineering utilities: pure data-shaping over cycle history and
grid cells. No Bedrock calls."""


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
            "what_conflicts_imply": older.get("what_conflicts_imply", [])
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

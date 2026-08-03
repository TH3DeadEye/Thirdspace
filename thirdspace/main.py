import hashlib
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from models import SessionMemory, TheOccurring, CultureAxis, CulturalFrame
from graph import build_graph
from budget import set_active_memory, reset_active_memory
from utils.prompts import render_frame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default location for persisted session JSON, one file per scenario.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def persist_session(memory: SessionMemory, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Write the full session state (grids, mediations, judge scores, output)
    to outputs/{scenario_id}.json. Returns the path written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{memory.scenario_id}.json"
    # Atomic write: write a temp file in the same dir, then os.replace (atomic
    # on POSIX). A crash mid-write leaves the old file intact, never a partial.
    tmp = path.with_suffix(".json.tmp")
    # Force UTF-8: the session JSON contains non-ASCII characters (e.g. the "→"
    # in mediation assumption updates, "…" from truncation). Path.write_text
    # defaults to the OS locale codec, which is cp1252 on Windows and raises
    # UnicodeEncodeError on those characters.
    tmp.write_text(memory.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _make_scenario_id(
    raw_scenario: str,
    main_frame: CulturalFrame | None,
    challenger_frame: CulturalFrame | None,
) -> str:
    """Deterministic 12-char id from the inputs, so re-running the same scenario
    writes to the same output file (reproducible; no orphaned partial runs).
    Frames are rendered first so a dimensional frame and its string form produce
    stable, comparable ids. In auto-detect mode (both frames None) the frames
    aren't known yet, so the id hashes the raw scenario alone."""
    content = f"{raw_scenario}|{render_frame(main_frame)}|{render_frame(challenger_frame)}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def run_scenario(
    raw_scenario: str,
    main_cultural_frame: CulturalFrame | None = None,
    challenger_cultural_frame: CulturalFrame | None = None,
    max_cycles: int = 3,
    mediator_frame: str | None = None,
    skip_generation: bool = False,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
) -> SessionMemory:
    """Run one scenario end to end and return the completed SessionMemory.

    Frames are always supplied by the caller — there is no inference step.
    Evaluation runs read them from the dataset (NormAd -> country, MDD ->
    active/passive agent) via prompts.read_cultural_frames. Passing None (or "")
    for a side puts that agent in NEUTRAL mode, where the prompt drops its frame
    block entirely rather than rendering an empty one.

    skip_generation stops the run after the loop. The convergence-based (NormAd)
    and dominant-conflict-type (MDD) classifiers read the mediation output, so an
    evaluation that does not need the document should not pay for it.
    """
    # A lone None coerces to "" — neutral for that side.
    main_cultural_frame = "" if main_cultural_frame is None else main_cultural_frame
    challenger_cultural_frame = (
        "" if challenger_cultural_frame is None else challenger_cultural_frame
    )

    scenario_id = _make_scenario_id(
        raw_scenario, main_cultural_frame, challenger_cultural_frame
    )
    fields = {
        "scenario_id": scenario_id,
        # SessionMemory.derive_identities builds the typed CulturalIdentity from
        # these frames. The challenger's working perspective starts empty
        # (prose=None); the loop refines it without ever touching the identity.
        "main_cultural_frame": main_cultural_frame,
        "challenger_cultural_frame": challenger_cultural_frame,
        "occurring": TheOccurring(
            scenario_id=scenario_id,
            raw_scenario=raw_scenario,
            nodes=[],
        ),
        "max_cycles": max_cycles,
        "skip_generation": skip_generation,
    }
    if mediator_frame is not None:
        fields["mediator_frame"] = mediator_frame
    memory = SessionMemory(**fields)
    graph = build_graph()
    # Register the active memory so call_llm can accumulate budget / latency
    # / failure telemetry and trip BudgetExceeded / CircuitOpen.
    token = set_active_memory(memory)
    try:
        # run_name / metadata / tags surface on the root LangSmith trace so a
        # run is identifiable by scenario without opening it. No-op if tracing
        # is disabled.
        result = graph.invoke(
            {"memory": memory},
            config={
                "run_name": f"thirdspace[{scenario_id}]",
                "metadata": {
                    "scenario_id": scenario_id,
                    "main_cultural_frame": main_cultural_frame,
                    "challenger_cultural_frame": challenger_cultural_frame,
                    "max_cycles": max_cycles,
                },
                "tags": ["thirdspace", "pipeline"],
            },
        )
    finally:
        reset_active_memory(token)
    final = result["memory"]
    if output_dir is not None:
        path = persist_session(final, output_dir)
        logger.info(f"[{final.scenario_id}] Persisted session to {path}")
    return final


if __name__ == "__main__":
    scenario = """
I'm a German PhD student in Toronto. A few weeks ago a colleague in my lab, Karim,
who moved here from Jordan, invited me and three other students to his apartment for
dinner. He cooked for hours — there was far too much food, and he kept refusing every
time I offered to help or to bring something. I had a great evening. The next morning,
to do the right thing, I sent him 35 dollars by e-transfer for my share of the
groceries, plus a little extra as a thank-you, with a friendly message about how good
the meal was. He never acknowledged the transfer. When I saw him in the lab he was
polite but distant, and he has since turned down my last two invitations to grab lunch,
saying he's busy. Another colleague hinted that I "offended" him, which makes no sense
to me. I paid my fair share, I didn't freeload, and I even added extra to thank him.
I genuinely do not understand what I did wrong.
    """

    # Frames are always supplied by the caller. Three forms:
    #
    # 1. Free-text label — what the datasets supply, and what is used below.
    #    NormAd gives a country; MDD gives an agent description.
    #        main_cultural_frame="germany", challenger_cultural_frame="jordan"
    #
    # 2. Dimensional axes (matches the Hofstede vocabulary; less stereotyping):
    #    main_cultural_frame=[
    #        CultureAxis(dimension="power_distance", value="low"),
    #        CultureAxis(dimension="individualism", value="individualist"),
    #    ],
    #
    # 3. Neutral baseline — "" for either side drops that agent's frame block
    #    entirely, so a no-frame run isolates the frame as an experimental
    #    variable instead of leaving dangling references to a frame that was
    #    never supplied:
    #        main_cultural_frame="", challenger_cultural_frame="",
    #
    # Convergence metric is selectable on SessionMemory (convergence_metric=
    # "embedding" default, or "jaccard"); run both to A/B the loop behavior.
    memory = run_scenario(
        raw_scenario=scenario,
        main_cultural_frame="germany",
        challenger_cultural_frame="jordan",
        # One cycle: multiple cycles at ~60 calls each cannot fit the account's
        # Bedrock rate limit in any reasonable wall time. Raise after a quota bump.
        max_cycles=1,
    )

    if memory.final_output:
        print("\n" + "=" * 60)
        print("FINAL OUTPUT")
        print("=" * 60)
        print(memory.final_output)

    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Cycles completed:  {len(memory.mediation_history)}")
    print(f"Converged:         {memory.converged} ({memory.convergence_reason})")
    if memory.mediation_history:
        last = memory.mediation_history[-1]
        summary = last.conflict_type_summary
        print(f"Agreements found:  {len(last.agreements)}")
        print(f"Conflicts found:   {len(last.conflicts)}")
        if summary:
            print(f"Dominant type:     {summary.dominant_type}")
            print(f"Type distribution: "
                  f"{ {t: getattr(summary, t) for t in ('factual', 'procedural', 'interpretive',
                       'power_related', 'identity_related', 'normative')} }")

    print("\n" + "=" * 60)
    print("TOKEN & CALL USAGE")
    print("=" * 60)
    print(f"LLM calls:         {memory.llm_calls_made}")
    print(f"Input tokens:      {memory.total_input_tokens:,}")
    print(f"Output tokens:     {memory.total_output_tokens:,}")
    print(f"Total tokens:      {memory.total_tokens:,}")
    print(f"Total latency:     {memory.total_latency_ms / 1000:.1f}s")

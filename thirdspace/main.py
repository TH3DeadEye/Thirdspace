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
    tmp.write_text(memory.model_dump_json(indent=2))
    os.replace(tmp, path)
    return path


def _make_scenario_id(
    raw_scenario: str,
    main_frame: CulturalFrame,
    challenger_frame: CulturalFrame,
) -> str:
    """Deterministic 12-char id from the inputs, so re-running the same scenario
    writes to the same output file (reproducible; no orphaned partial runs).
    Frames are rendered first so a dimensional frame and its string form produce
    stable, comparable ids."""
    content = f"{raw_scenario}|{render_frame(main_frame)}|{render_frame(challenger_frame)}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def run_scenario(
    raw_scenario: str,
    main_cultural_frame: CulturalFrame,
    challenger_cultural_frame: CulturalFrame,
    max_cycles: int = 3,
    challenger_style: str | None = None,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
) -> SessionMemory:
    scenario_id = _make_scenario_id(
        raw_scenario, main_cultural_frame, challenger_cultural_frame
    )
    fields = {
        "scenario_id": scenario_id,
        "main_cultural_frame": main_cultural_frame,
        "challenger_cultural_frame": challenger_cultural_frame,
        # Seed the working perspective from the cultural frame; the loop refines it.
        "active_challenger_perspective": challenger_cultural_frame,
        "occurring": TheOccurring(
            scenario_id=scenario_id,
            raw_scenario=raw_scenario,
            nodes=[],
        ),
        "max_cycles": max_cycles,
    }
    if challenger_style is not None:
        fields["challenger_style"] = challenger_style
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
    I'm a Canadian working at a tech company. My new colleague is from Iran.
    Last week I gave him feedback on his code in our team meeting — I pointed
    out some inefficiencies directly in front of everyone, the way I always do
    with the team. He went completely silent, didn't respond, and has barely
    spoken to me since. My manager told me he seems upset but won't say why.
    I don't understand what I did wrong — I was just being honest and direct.
    """

    # Frames accept three forms (see models.CulturalFrame):
    #
    # 1. Free-text label (back-compat) — used below.
    #
    # 2. Dimensional axes (matches a Hofstede/Meyer fine-tune; less stereotyping):
    #    main_cultural_frame=[
    #        CultureAxis(dimension="directness_of_negative_feedback", value="high / direct"),
    #        CultureAxis(dimension="context", value="low"),
    #        CultureAxis(dimension="power_distance", value="low"),
    #    ],
    #    challenger_cultural_frame=[
    #        CultureAxis(dimension="directness_of_negative_feedback", value="low / indirect"),
    #        CultureAxis(dimension="context", value="high"),
    #        CultureAxis(dimension="face_honor_sensitivity", value="high"),
    #    ],
    #
    # 3. Neutral baseline (no frame for either agent — drops the frame scaffolding
    #    and the "name your frame" lines, to test the output-verbosity hypothesis):
    #    main_cultural_frame="", challenger_cultural_frame="",
    #
    # Convergence metric is selectable on SessionMemory (convergence_metric=
    # "embedding" default, or "jaccard"); run both to A/B the loop behavior.
    memory = run_scenario(
        raw_scenario=scenario,
        main_cultural_frame="Canadian low-context direct communication culture",
        challenger_cultural_frame="Iranian high-context honor-based communication culture",
        max_cycles=3,
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

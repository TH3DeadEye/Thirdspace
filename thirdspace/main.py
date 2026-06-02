import logging
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from models import SessionMemory, TheOccurring
from graph import build_graph
from budget import set_active_memory, reset_active_memory

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
    path.write_text(memory.model_dump_json(indent=2))
    return path


def run_scenario(
    raw_scenario: str,
    main_cultural_frame: str,
    challenger_cultural_frame: str,
    max_cycles: int = 3,
    challenger_style: str | None = None,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
) -> SessionMemory:
    scenario_id = str(uuid.uuid4())[:8]
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

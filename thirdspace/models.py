from pydantic import BaseModel, computed_field
from typing import Literal
from pydantic import Field
# --- Constants (never use raw integers in logic) ---
NUM_FACETS = 6
NUM_DIMENSIONS = 7
NUM_CELLS = NUM_FACETS * NUM_DIMENSIONS

FACET_LIST = [
    "individuals",
    "interactional_event",
    "process",
    "evidence_and_data",
    "concepts_theories_rules",
    "inferences",
]

DIMENSION_LIST = [
    "objective",
    "query_formulation",
    "perspective",
    "assumptions",
    "evidence",
    "concepts",
    "implications",
]

FACETS = Literal[
    "individuals",
    "interactional_event",
    "process",
    "evidence_and_data",
    "concepts_theories_rules",
    "inferences",
]

CT_DIMENSIONS = Literal[
    "objective",
    "query_formulation",
    "perspective",
    "assumptions",
    "evidence",
    "concepts",
    "implications",
]


# --- Phase 1: The Occurring ---

class OccurringNode(BaseModel):
    node_id: str
    facet: FACETS
    content: str
    assumptions: list[str]
    last_updated_cycle: int = 0


class TheOccurring(BaseModel):
    scenario_id: str
    raw_scenario: str
    nodes: list[OccurringNode]  # always exactly NUM_FACETS nodes

    def get_node(self, facet: str) -> OccurringNode | None:
        return next((n for n in self.nodes if n.facet == facet), None)


# --- Phase 2 & 3: Reasoning Grids ---

class ReflectionFlag(BaseModel):
    measure: str   # clarity | correctness | precision | proportion |
                   # depth | comprehensiveness | logic | fairness
    issue: str


class NodeDimensionCell(BaseModel):
    facet: FACETS
    dimension: CT_DIMENSIONS
    content: str
    reflection_flags: list[ReflectionFlag] = []
    # Challenger-only field. None for main agent cells.
    response_type: Literal["challenge", "reframe", "confirm"] | None = None
    cycle_last_updated: int = 0
    # Telemetry: how many self-refinement retries this cell needed and why.
    # 0/[] means the model's first attempt passed validation.
    refinement_attempts: int = 0
    refinement_reasons: list[str] = []


class NodeReasoningGrid(BaseModel):
    cycle: int
    agent: Literal["main", "challenger"]
    cells: list[NodeDimensionCell]  # exactly NUM_CELLS cells

    def get_cell(self, facet: str, dimension: str) -> NodeDimensionCell | None:
        return next(
            (c for c in self.cells if c.facet == facet and c.dimension == dimension),
            None,
        )

    def get_facet_row(self, facet: str) -> list[NodeDimensionCell]:
        return [c for c in self.cells if c.facet == facet]


# --- Phase 4: Mediation ---

class CellConflict(BaseModel):
    facet: str
    dimension: str
    main_summary: str
    challenger_summary: str
    challenger_response_type: Literal["challenge", "reframe", "confirm"]
    implication: str


class WholenessImplyingNodes(BaseModel):
    """Distilled output that feeds the next cycle and constrains final generation."""
    query_assumptions: list[str]
    reformulated_query: str | None
    new_objective: str | None
    updated_challenger_perspective: str | None
    objective_assumptions: list[str]
    perspective_assumptions: list[str]


class ImplyingNodes(BaseModel):
    cycle: int
    agreements: list[str]
    conflicts: list[CellConflict]
    what_conflicts_imply: list[str]
    updated_assumptions: list[str]
    wholeness: WholenessImplyingNodes
    converged: bool


# --- LLM-as-a-judge scores ---

class JudgeScore(BaseModel):
    cycle: int
    empathy: int = Field(ge=1, le=5)          # 1-5
    consensus: int = Field(ge=1, le=5)       # 1-5
    escalation_risk: int = Field(ge=1, le=5)  # 1-5
    cultural_insight: int = Field(ge=1, le=5) # 1-5
    reasoning: str 


# --- Session Memory ---

class SessionMemory(BaseModel):
    scenario_id: str
    main_cultural_frame: str
    challenger_cultural_frame: str
    # Argumentation style set by the mediator expert. Independent of the
    # cultural identity above (see PDF item 4: the challenger's procedural
    # style is separate from the role of the cultures involved).
    challenger_style: str = (
        "procedural adversarial argumentation: surface load-bearing assumptions, "
        "demand they be justified rather than asserted, and refuse to let any "
        "single cultural norm stand in as a universal fact"
    )
    # Working perspective for the challenger. Seeded from challenger_cultural_frame
    # and updated each cycle from the mediation's implying nodes.
    active_challenger_perspective: str | None = None
    occurring: TheOccurring | None = None
    main_grids: list[NodeReasoningGrid] = []
    challenger_grids: list[NodeReasoningGrid] = []
    mediation_history: list[ImplyingNodes] = []
    # Two separate judge-score lists removes the fragile `judge_scores[-1]`
    # indexing that depended on graph ordering. Cycle judges score in-loop
    # mediations; output judges score the final generated document.
    cycle_judge_scores: list[JudgeScore] = []
    output_judge_scores: list[JudgeScore] = []
    current_cycle: int = 0
    max_cycles: int = 3
    # Generation <-> evaluation regeneration loop controls.
    max_regen: int = 1
    regen_attempts: int = 0
    output_score_threshold: int = 3
    # Consensus plateau as a secondary convergence signal. If the last two
    # cycle judges both score consensus >= threshold and differ by <= 1, the
    # loop terminates even if the mediator did not self-declare converged.
    consensus_plateau_threshold: int = 4
    # Cost / latency guardrails. call_llm increments llm_calls_made and
    # total_latency_ms on every Bedrock call. BudgetExceeded raises when
    # llm_calls_made would exceed llm_call_budget. The circuit breaker counts
    # consecutive_failures; CircuitOpen raises when it exceeds the cap.
    llm_call_budget: int = 1000
    llm_calls_made: int = 0
    total_latency_ms: int = 0
    consecutive_failures: int = 0
    max_consecutive_failures: int = 5
    final_output: str | None = None
    converged: bool = False

    @computed_field
    @property
    def judge_scores(self) -> list[JudgeScore]:
        """Backwards-compatible combined view of all judge scores."""
        return [*self.cycle_judge_scores, *self.output_judge_scores]

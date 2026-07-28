from __future__ import annotations

#importing dependencies
import difflib
import json
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


#creating the Ontology classes (Facets and Dimentions for both main and challenger agents)
class Facet(StrEnum):
    AGENTS = "agents"
    ACTIONS = "actions"
    CONTEXT = "context"
    REPRESENTATIONS = "representations"
    CONSEQUENCES = "consequences"

class Dimension(StrEnum):
    OBSERVATION = "observation"      # what is directly supported?
    INTERPRETATION = "interpretation"  # what meaning is assigned?
    ASSUMPTION = "assumption"        # what is taken for granted?
    VALUE = "value"                  # what is important, right, or legitimate?
    CONSTRAINT = "constraint"        # what rules/resources/power limit action?
    CONSEQUENCE = "consequence"      # what follows if this is accepted?

FACET_LIST: list[Facet] = list(Facet)
DIMENSION_LIST: list[Dimension] = list(Dimension)
NUM_FACET = len(Facet)  
NUM_DIMENSION = len(Dimension)
NUM_CELLS = NUM_FACET * NUM_DIMENSION


"""# The reactive challenger sees only this slice of the main grid (design §2).
CHALLENGER_VISIBLE_DIMENSIONS: list[Dimension] = [
    Dimension.ASSUMPTION, Dimension.INTERPRETATION, Dimension.VALUE,
]"""

# Implementing the  CONFLICT TAXONOMY (Habermas/Gadamer 6-type)

class ConflictType(StrEnum):
    FACTUAL = "factual"                  # 1: dispute over what occurred
    PROCEDURAL = "procedural"            # 2: which rules/process govern
    INTERPRETIVE = "interpretive"        # 3: agree on facts, dispute meaning
    POWER_RELATED = "power_related"      # 4: who has authority/legitimacy
    IDENTITY_RELATED = "identity_related"  # 5: dignity / recognition
    NORMATIVE = "normative"              # 6: incompatible values (DEFAULT)


class Resolvability(StrEnum):
    CONVERSATIONAL = "conversational"
    FACILITATED = "facilitated"
    STRUCTURAL_CHANGE = "structural_change"
    EXTERNAL_STANDARD = "external_standard"
    UNRESOLVABLE = "unresolvable"


ALL_RESOLVABILITIES: frozenset[Resolvability] = frozenset(Resolvability)

@dataclass(frozen=True)
class ConflictTypeDefinition:
    """Single source of truth for one conflict type. The mediation prompt (Omid)
    and the validators both derive from this — the decision order, yes/no test,
    allowed resolvabilities, and intervention prose live in one place."""
    key: ConflictType
    label: str
    order: int                 # cascade position; mediator stops at first Yes
    is_default: bool           # fires when no earlier step fired (normative)
    decision_test: str
    allowed_resolvabilities: frozenset[Resolvability]
    intervention_text: str     # plain prose injected into generation (no rubric labels)

_R = Resolvability  # registry shorthand

_CONFLICT_REGISTRY: list[ConflictTypeDefinition] = [
    ConflictTypeDefinition(
        key=ConflictType.FACTUAL, label="Factual conflict", order=1, is_default=False,
        decision_test="Would a shared recording of events settle it, before any interpretation?",
        allowed_resolvabilities=frozenset({_R.EXTERNAL_STANDARD, _R.UNRESOLVABLE}),
        intervention_text=(
            "The parties disagree about the bare sequence of events. Name exactly which "
            "facts are disputed and what record would settle them. If none is available, "
            "say the dispute cannot be settled on the facts — that is itself the finding."),
    ),
    ConflictTypeDefinition(
        key=ConflictType.PROCEDURAL, label="Procedural conflict", order=2, is_default=False,
        decision_test="Would agreement on which procedure applies settle it, regardless of values or identity?",
        allowed_resolvabilities=frozenset({_R.FACILITATED, _R.EXTERNAL_STANDARD, _R.STRUCTURAL_CHANGE}),
        intervention_text=(
            "The parties agree on what happened but not on which rule or process governs it. "
            "Name the specific procedure in contention and what authority or shared standard "
            "could establish which one applies. More conversation between them will not settle "
            "it — the missing piece is an agreed procedure."),
    ),
    ConflictTypeDefinition(
        key=ConflictType.INTERPRETIVE, label="Interpretive conflict", order=3, is_default=False,
        decision_test="Would a shared explanatory framework settle it, even if values still differ?",
        allowed_resolvabilities=frozenset({_R.CONVERSATIONAL, _R.FACILITATED}),
        intervention_text=(
            "The parties agree on the facts but assign them different meanings. Reconstruct "
            "each account and show what shared framework would let both make sense of the same "
            "events. If they are using one word in two operational senses, name the word and "
            "propose a joint definition."),
    ),
    ConflictTypeDefinition(
        key=ConflictType.POWER_RELATED, label="Power-related conflict", order=4, is_default=False,
        decision_test="Would this reproduce between any two people in these structural positions, regardless of their values or identity?",
        allowed_resolvabilities=frozenset({_R.STRUCTURAL_CHANGE, _R.EXTERNAL_STANDARD, _R.UNRESOLVABLE}),
        intervention_text=(
            "The conflict is driven by who holds the authority or legitimacy to define, decide, "
            "or enforce. Name the structural asymmetry generating it, what would have to change "
            "about the situation, and who has the standing to change it. Better dialogue cannot "
            "repair a distortion in who gets to speak."),
    ),
    ConflictTypeDefinition(
        key=ConflictType.IDENTITY_RELATED, label="Identity-related conflict", order=5, is_default=False,
        decision_test="Is the core injury being misrecognized or excluded, rather than the outcome of the decision?",
        allowed_resolvabilities=frozenset({_R.FACILITATED, _R.STRUCTURAL_CHANGE, _R.UNRESOLVABLE}),
        intervention_text=(
            "The core injury is about dignity, worth, or recognition — resolution would require "
            "one party to accept a diminished account of who they are. Name the misrecognition "
            "and what acknowledgment the injured party needs. Recognition cannot be mandated by "
            "an outside rule, so do not propose one as the remedy."),
    ),
    ConflictTypeDefinition(
        key=ConflictType.NORMATIVE, label="Normative conflict", order=6, is_default=True,
        decision_test="Do both understand fully, agree on facts/rules/meaning, yet hold incompatible views of what ought to happen?",
        allowed_resolvabilities=frozenset({_R.FACILITATED, _R.EXTERNAL_STANDARD, _R.STRUCTURAL_CHANGE, _R.UNRESOLVABLE}),
        intervention_text=(
            "Both parties understand each other and disagree about what ought to happen; neither "
            "will move with more information. Name the incompatible value frameworks and present "
            "realistic options — an external standard neither authored, separating the value "
            "question from the practical decision, or naming the incompatibility as permanent. "
            "Do not recommend better communication; understanding is not their problem."),
    ),
]

CONFLICT_TYPE_DEFS: dict[ConflictType, ConflictTypeDefinition] = {d.key:d for d in _CONFLICT_REGISTRY}

#building safe guards to fail fast if a type is missing from the registry
assert set(CONFLICT_TYPE_DEFS) == set(ConflictType), "registry vs ConflictType drift"
assert sum(d.is_default for d in _CONFLICT_REGISTRY) == 1, "exactly one default type"
for _d in _CONFLICT_REGISTRY:
    assert _d.allowed_resolvabilities <= ALL_RESOLVABILITIES, f"{_d.key}: Bad resolvabilty set"




def get_intervention_text(conflict_type: ConflictType) -> str: 
    d = CONFLICT_TYPE_DEFS.get(conflict_type)
    if d is None:
        raise KeyError(f"Unknown conflict type {conflict_type!r}. Valid: {list(ConflictType)}")

    return d.intervention_text


def get_allowed_resolvabilities(conflict_type: ConflictType) -> frozenset[Resolvability]:
    d = CONFLICT_TYPE_DEFS.get(conflict_type)
    return d.allowed_resolvabilities if d else ALL_RESOLVABILITIES


# Internal rubric vocabulary that must never appear in a GENERATED DOCUMENT.
# Single source of truth for tests/test_leak_phrases.py and the eval output grep.
# (The mediation prompt may legitimately use these words; this governs the doc.)
LEAK_PHRASES: list[str] = [
    "intervention class", "absolute prohibition", "critical prohibition",
    "false resolution trap", "prohibited:", "severity >=",
    "of conflicts are structural", "resolvability", "conflict type distribution",
]


#implementing ConflictType summary

_TYPE_NAMES:list[str] = [t.value for t in ConflictType]

class ConflictTypeSummary(BaseModel):
    factual: int = 0
    procedural: int = 0
    interpretive: int = 0
    power_related: int = 0
    identity_related: int = 0
    normative: int = 0
    total: int = 0
    dominant_type: ConflictType
    power_related_fraction: float          
    conversational_fraction: float
    unresolvable_fraction: float
    power_related_severity_score: float
    scope: Literal["cycle", "cross_cycle"] = "cycle"
    cycles_included: list[int] = []


    @classmethod
    def from_conflicts(cls, conflicts: list["CellConflict"],
                       scope: Literal["cycle", "cross_cycle"] = "cycle",
                       cycles_included: list[int] | None = None) -> "ConflictTypeSummary | None":
        if not conflicts:
            return None
        counts = Counter(c.conflict_type for c in conflicts)
        total = len(conflicts)
        return cls(
            **{name: counts.get(name, 0) for name in _TYPE_NAMES},
            total=total,
            dominant_type=counts.most_common(1)[0][0],
            power_related_fraction=counts.get("power_related", 0) / total,
            conversational_fraction=sum(1 for c in conflicts if c.resolvability == "conversational") / total,
            unresolvable_fraction=sum(1 for c in conflicts if c.resolvability == "unresolvable") / total,
            power_related_severity_score=sum(
                c.severity for c in conflicts if c.conflict_type == "power_related") / (5 * total),
            scope=scope,
            cycles_included=sorted(cycles_included or []),
        )

    @classmethod
    def aggregate(cls, per_cycle: list["ConflictTypeSummary | None"]) -> "ConflictTypeSummary | None":
        summaries = [s for s in per_cycle if s is not None]
        total = sum(s.total for s in summaries)
        if total == 0:
            return None
        counts = {name: sum(getattr(s, name) for s in summaries) for name in _TYPE_NAMES}
        return cls(
            **counts, total=total,
            dominant_type=max(counts, key=counts.get),
            power_related_fraction=counts["power_related"] / total,
            conversational_fraction=sum(s.conversational_fraction * s.total for s in summaries) / total,
            unresolvable_fraction=sum(s.unresolvable_fraction * s.total for s in summaries) / total,
            power_related_severity_score=sum(s.power_related_severity_score * s.total for s in summaries) / total,
            scope="cross_cycle",
            cycles_included=sorted({c for s in summaries for c in s.cycles_included}),
        )
    



#The updated Occuring:

class OccurringNode(BaseModel):
    node_id:str
    facet:Facet
    content:str
    assumptions:list[str]
    last_updated_cycle:int = 0


def _content_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()

def _truncate_content(text: str, limit: int = 380) -> str:
    if len(text) <= limit:
        return text
    cut = text[:limit]
    last_space = cut.rfind(" ")
    if last_space > 100:
        cut = cut[:last_space]
    return cut + "…"

def _slim_budget_per_node(nodes: list["OccurringNode"], focus_facet: Facet,
                          total_context_chars: int = 4000, focus_fraction: float = 0.5) -> dict[Facet, int]:
    focus_budget = int(total_context_chars * focus_fraction)
    n_others = len(nodes) - 1
    other_budget = int((total_context_chars * (1 - focus_fraction)) / max(n_others, 1))
    return {n.facet: focus_budget if n.facet == focus_facet else other_budget for n in nodes}


class TheOccurring(BaseModel):
    scenario_id:str
    raw_scenario:str
    nodes: list[OccurringNode]

    def get_node(self, facet: str) -> OccurringNode | None:
        return next((n for n in self.nodes if n.facet == facet), None)

    def assumptions_by_facet(self) -> dict[str, list[str]]:
        """Occurring Assumptions fed to the challenger in cycle 1 (design §2)."""
        return {str(n.facet): n.assumptions for n in self.nodes}
    
    def to_nodes_json(self) -> str:
        return json.dumps(
            {"scenario_id": self.scenario_id, "nodes": [n.model_dump(mode="json") for n in self.nodes]},
            ensure_ascii=False,
        )

    def slim_context(self, focus_facet: str, total_context_chars: int = 4000) -> str:
        focus = self.get_node(focus_facet)
        budgets = _slim_budget_per_node(self.nodes, focus_facet, total_context_chars)
        context_summary = {
            str(n.facet): _truncate_content(n.content, limit=budgets[n.facet])
            for n in self.nodes if n.facet != focus_facet
        }
        return json.dumps({
            "focus_node": ({"facet": str(focus.facet), "content": focus.content,
                            "assumptions": focus.assumptions} if focus else None),
            "context_summary": context_summary,
            "context_budget": {
                "focus_chars": len(focus.content) if focus else 0,
                "truncated_nodes": [str(n.facet) for n in self.nodes
                                    if n.facet != focus_facet and len(n.content) > budgets[n.facet]],
            },
        }, ensure_ascii=False)


class NodeUpdateDelta(BaseModel):
    facet: Facet
    similarity: float
    model_claimed_changed: bool
    actually_changed: bool
    content_kept: str
    content_proposed: str
    conflict: bool


class TheOccurringUpdateResult(BaseModel):
    occurring: TheOccurring
    deltas: list[NodeUpdateDelta]

    @property
    def any_meaningful_change(self) -> bool:
        return any(d.actually_changed for d in self.deltas)

    @property
    def conflicted_nodes(self) -> list[str]:
        return [str(d.facet) for d in self.deltas if d.conflict]



#reasoning girds
class ReflectionFlag(BaseModel):
    measure: str 
    issue: str


class NodeDimensionCell(BaseModel):
    facet: Facet
    dimension: Dimension
    content: str
    reflection_flags: list[ReflectionFlag] = []
    cycle_last_updated: int = 0
    refinement_attempts: int = 0
    refinement_reasons: list[str] = []
    content_hashes: list[str] = []   # sentence hashes, for cross-cycle dedup


class NodeReasoningGrid(BaseModel):
    cycle: int
    agent: Literal["main", "challenger"]
    cells: list[NodeDimensionCell]   # exactly NUM_CELLS

    def get_cell(self, facet: str, dimension: str) -> NodeDimensionCell | None:
        return next((c for c in self.cells if c.facet == facet and c.dimension == dimension), None)

    def get_facet_row(self, facet: str) -> list[NodeDimensionCell]:
        return [c for c in self.cells if c.facet == facet]


#MEDIATION reasoning 
class CellConflict(BaseModel):
    facet: Facet
    dimension: Dimension
    main_summary: str          # keep SHORT — a reference, not a restatement
    challenger_summary: str
    implication: str
    conflict_type: ConflictType
    conflict_type_rationale: str
    is_semantic_subtype: bool = False   # only meaningful when interpretive
    severity: int = Field(ge=1, le=5)
    resolvability: Resolvability

    @field_validator("implication")
    @classmethod
    def implication_must_reveal(cls, v: str) -> str:
        v = v.strip()
        starters = ("this reveals", "this suggests", "this exposes",
                    "this shows", "this uncovers", "this indicates")
        if not any(v.lower().startswith(s) for s in starters):
            raise ValueError(f"implication must begin with 'This reveals that...' etc. Got: {v[:80]!r}")
        if len(v) < 60:
            raise ValueError(f"implication too short ({len(v)} chars)")
        return v

    @field_validator("conflict_type_rationale")
    @classmethod
    def rationale_substantive(cls, v: str) -> str:
        if len(v.strip()) < 40:
            raise ValueError("conflict_type_rationale must be at least one complete sentence")
        return v.strip()

    @model_validator(mode="after")
    def consistency(self) -> "CellConflict":
        if self.resolvability not in get_allowed_resolvabilities(self.conflict_type):
            raise ValueError(f"{self.resolvability!r} not allowed for {self.conflict_type!r}")
        if self.is_semantic_subtype and self.conflict_type != ConflictType.INTERPRETIVE:
            raise ValueError("is_semantic_subtype is only valid for interpretive conflicts")
        return self


class WholenessImplyingNodes(BaseModel):
    """Mediator's constraining context for the next cycle (design 'Wholeness schema').
    Flat — no main/challenger split (the challenger now sees the whole grid).
    COORDINATE these field names with Omid's mediation prompt so it emits them 1:1."""
    what_conflicts_imply: list[str] = []      # root pattern across conflict cells
    updated_assumptions: list[str] = []       # assumptions to revise in The Occurring
    comparison_assumptions: list[str] = []    # visible only by comparing main vs challenger
    reformulated_queries: list[str] = []      # clearer next-cycle questions
    new_objectives: list[str] = []            # revised reasoning goals
    updated_perspectives: list[str] = []      # revised interpretive lenses


class ImplyingNodes(BaseModel):
    cycle: int
    agreements: list[str]
    conflicts: list[CellConflict]
    wholeness: WholenessImplyingNodes
    converged: bool
    conflict_type_summary: ConflictTypeSummary | None = None

    @model_validator(mode="after")
    def compute_type_summary(self) -> "ImplyingNodes":
        self.conflict_type_summary = ConflictTypeSummary.from_conflicts(
            self.conflicts, scope="cycle", cycles_included=[self.cycle])
        return self

    def to_context(self, cycle: int) -> dict | None:
        """One constraining-context payload fed to BOTH agents in the next cycle
        (design: no main/challenger split; the challenger sees the full grid)."""
        w = self.wholeness
        guidance = {k: v for k, v in {
            "what_conflicts_imply": w.what_conflicts_imply,
            "updated_assumptions": w.updated_assumptions,
            "comparison_assumptions": w.comparison_assumptions,
            "reformulated_queries": w.reformulated_queries,
            "new_objectives": w.new_objectives,
            "updated_perspectives": w.updated_perspectives,
        }.items() if v}
        if not guidance:
            return None
        return {"from_cycle": cycle, "guidance": guidance,
                "_note": "Produced by the mediator last cycle. Refine — do not repeat."}



#cultral and Moral Framing
class CultureAxis(BaseModel):
    dimension: str   #moral foundation (moral)
    value: str


CulturalFrame = str | list[CultureAxis]


class CulturalIdentity(BaseModel):
    axes: list[CultureAxis] = []
    label: str = ""

    @classmethod
    def from_frame(cls, frame: "CulturalFrame") -> "CulturalIdentity":
        if isinstance(frame, list):
            return cls(axes=frame, label="")
        return cls(axes=[], label=frame or "")

    @property
    def is_neutral(self) -> bool:
        return not self.axes and not self.label

    @property
    def is_dimensional(self) -> bool:
        return bool(self.axes)

    def render(self) -> str:
        if self.is_neutral:
            return ""
        if self.is_dimensional:
            return "; ".join(f"{a.dimension}: {a.value}" for a in self.axes)
        return self.label


class WorkingPerspective(BaseModel):
    cycle_last_updated: int = 0
    prose: str | None = None

    def render_for_prompt(self, identity: "CulturalIdentity") -> str:
        identity_str = identity.render()
        if self.prose is None:
            return identity_str
        if not identity_str:
            return self.prose
        return f"{identity_str}\n\nMediating perspective (cycle {self.cycle_last_updated}): {self.prose}"


CULTURE_DIMENSION_MENU: dict[str, tuple[str, str]] = {
    "power_distance": ("high", "low"),
    "individualism": ("individualist", "collectivist"),
    "masculinity": ("masculine", "feminine"),
    "uncertainty_avoidance": ("high", "low"),
    "long_term_orientation": ("long", "short"),
    "indulgence": ("indulgent", "restrained"),
}

# Moral arm: Moral Foundations, same shape
MORAL_VALUE_MENU: dict[str, tuple[str, str]] = {
    "care": ("care", "harm"),
    "fairness": ("fairness", "cheating"),
    "loyalty": ("loyalty", "betrayal"),
    "authority": ("authority", "subversion"),
    "sanctity": ("sanctity", "degradation"),
    "liberty": ("liberty", "oppression"),
}

FRAME_MENUS: dict[str, dict[str, tuple[str, str]]] = {
    "cultural": CULTURE_DIMENSION_MENU,
    "moral": MORAL_VALUE_MENU,
}


class DetectedFrames(BaseModel):
    main: list[CultureAxis]
    challenger: list[CultureAxis]
    rationale: str
    menu: Literal["cultural", "moral"] = "cultural"


#session Memeory
class CompressedHistoryEntry(BaseModel):
    cycle: int
    dominant_conflict_type: str | None
    agreement_count: int
    conflict_count: int
    key_agreements: list[str]
    key_conflicts: list[str]
    what_conflicts_imply: list[str]
    reformulated_query: str | None
    new_objective: str | None
    converged: bool

    @classmethod
    def from_mediation(cls, m: "ImplyingNodes") -> "CompressedHistoryEntry":
        w = m.wholeness
        return cls(
            cycle=m.cycle,
            dominant_conflict_type=(str(m.conflict_type_summary.dominant_type)
                                    if m.conflict_type_summary else None),
            agreement_count=len(m.agreements),
            conflict_count=len(m.conflicts),
            key_agreements=m.agreements[:3],
            key_conflicts=[c.implication for c in m.conflicts if c.severity >= 3][:5],
            what_conflicts_imply=w.what_conflicts_imply,
            reformulated_query=(w.reformulated_queries[0] if w.reformulated_queries else None),
            new_objective=(w.new_objectives[0] if w.new_objectives else None),
            converged=m.converged,
        )


class SessionMemory(BaseModel):
    scenario_id: str
    main_cultural_frame: CulturalFrame
    challenger_cultural_frame: CulturalFrame
    auto_detect_frames: bool = False
    frame_detection: "DetectedFrames | None" = None
    frame_menu: Literal["cultural", "moral"] = "cultural"
    challenger_style: str = (
        "procedural adversarial argumentation: surface load-bearing assumptions, "
        "demand they be justified rather than asserted, and refuse to let any single "
        "cultural norm stand in as a universal fact")
    mediator_frame: str = (
        "procedural rationalist: prioritizes logical consistency, explicit reasoning, "
        "and symmetrical treatment of both parties")
    main_identity: CulturalIdentity = Field(default_factory=CulturalIdentity)
    challenger_identity: CulturalIdentity = Field(default_factory=CulturalIdentity)
    challenger_working_perspective: WorkingPerspective = Field(default_factory=WorkingPerspective)
    occurring: TheOccurring | None = None
    main_grids: list[NodeReasoningGrid] = []
    challenger_grids: list[NodeReasoningGrid] = []
    mediation_history: list[ImplyingNodes] = []
    compressed_history: list[CompressedHistoryEntry] = []
    current_cycle: int = 0
    max_cycles: int = 3 
    convergence_metric: Literal["jaccard", "embedding"] = "embedding"
    jaccard_convergence_threshold: float = 0.7
    embedding_convergence_threshold: float = 0.85
    convergence_reason: Literal["llm_declared", "structural_convergence", "max_cycles"] | None = None
    llm_call_budget: int = 1000
    llm_calls_made: int = 0
    total_latency_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    consecutive_failures: int = 0
    max_consecutive_failures: int = 5
    final_output: str | None = None
    converged: bool = False

    @model_validator(mode="after")
    def derive_identities(self) -> "SessionMemory":
        if self.main_identity.is_neutral:
            self.main_identity = CulturalIdentity.from_frame(self.main_cultural_frame)
        if self.challenger_identity.is_neutral:
            self.challenger_identity = CulturalIdentity.from_frame(self.challenger_cultural_frame)
        return self

"""Run configuration. One dataclass in, one run out — so a run's settings can
be serialized next to its predictions and a result is reproducible."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal

DatasetName = Literal["normad", "mdd"]
ClassifierName = Literal["structural", "readout", "hybrid"]

# NormAd is evaluated in three context conditions in the original paper. The
# frame the pipeline consumes is the country, so `country` is the default;
# the others exist to isolate how much of the accuracy comes from the frame
# rather than from the reasoning.
NormAdContext = Literal["none", "country", "rot", "background"]


@dataclass
class StructuralThresholds:
    """Cut points for the deterministic classifier.

    These are STARTING VALUES, not tuned constants. Run `python -m eval
    calibrate --run-dir <dir>` on a completed run to fit them on cached
    signals (no LLM calls) and pass the result back with --thresholds.
    """
    # Below this conflict density, and above this agreement ratio, the two
    # readings substantially agree -> the action fits the norm.
    low_conflict_density: float = 0.25
    high_agreement_ratio: float = 0.50
    # Above both of these, the disagreement is about what OUGHT to happen and
    # it is load-bearing -> the action violates the norm.
    norm_pressure: float = 0.45
    mean_severity: float = 3.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "StructuralThresholds":
        if not d:
            return cls()
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @classmethod
    def from_file(cls, path: str | Path | None) -> "StructuralThresholds":
        if not path:
            return cls()
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class EvalConfig:
    # --- what to evaluate on -------------------------------------------------
    dataset: DatasetName = "normad"
    path: str | None = None              # local csv/tsv/json/parquet; else HF hub
    split: str = "train"
    n: int = 20                          # how many records to run
    seed: int = 0
    stratify: bool = True                # equal-ish per gold label
    countries: list[str] = field(default_factory=list)   # normad filter
    domains: list[str] = field(default_factory=list)     # mdd filter

    # --- how to run the pipeline --------------------------------------------
    max_cycles: int = 1                  # each cycle is ~60+ Bedrock calls
    convergence_metric: Literal["jaccard", "embedding"] = "embedding"
    skip_generation: bool = True         # the classifiers read mediation, not prose
    mediator_frame: str | None = None
    llm_call_budget: int = 400           # per scenario; a hard stop, not a target
    workers: int = 1                     # scenarios in parallel (Bedrock quota!)

    # --- how to turn output into a label ------------------------------------
    classifier: ClassifierName = "hybrid"
    thresholds: StructuralThresholds = field(default_factory=StructuralThresholds)
    baseline: bool = False               # also predict with a single raw-record call
    readout_model: str | None = None     # defaults to llm.JUDGE_MODEL_ID

    # --- dataset-specific adaptation ----------------------------------------
    normad_context: NormAdContext = "country"
    # NormAd ships one country per row, so the challenger has no dataset frame.
    # "neutral" leaves it unframed (prompts.py drops the frame block entirely);
    # anything else is used verbatim as the challenger's frame.
    normad_challenger: str = "neutral"
    mdd_label_field: str | None = None   # override auto-detection
    mdd_scenario_field: str | None = None

    # --- bookkeeping ---------------------------------------------------------
    # None -> <repo>/outputs/eval, resolved from where the algorithm folder was
    # found rather than from the shell's working directory, so a run started
    # from inside eval/ writes to the same place as one started from the root.
    out_dir: str | None = None
    run_id: str | None = None
    resume: bool = False
    persist_sessions: bool = False       # write the full SessionMemory per record
    src: str | None = None               # path to the thirdspace/ algorithm folder

    def to_dict(self) -> dict:
        d = asdict(self)
        d["thresholds"] = self.thresholds.to_dict()
        d["out_dir"] = str(self.resolved_out_dir())
        return d

    def resolved_out_dir(self) -> Path:
        if self.out_dir:
            return Path(self.out_dir).expanduser()
        from ._bootstrap import default_out_dir
        return default_out_dir(self.src)

    def run_dir(self) -> Path:
        return self.resolved_out_dir() / (self.run_id or "unnamed")

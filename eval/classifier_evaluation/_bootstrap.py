"""Put the ThirdSpace source root on sys.path.

The pipeline uses flat, absolute imports (`from models import ...`,
`from utils import prompts`), so its source directory must be ON sys.path
rather than imported as a package. Rather than hard-code a layout, we look
for the directory that actually contains models.py + graph.py.

Resolution order:
  1. $THIRDSPACE_SRC
  2. an explicit path passed by the caller (--src on the CLI)
  3. common layouts relative to this file: ../src, ../thirdspace, ..,
     ../../src, and so on up three levels.

Every import of pipeline code goes through here, so a repo with a different
layout only needs THIRDSPACE_SRC set once.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# The two modules that identify a ThirdSpace source root.
_MARKERS = ("models.py", "graph.py")

_resolved: Path | None = None


def _looks_like_src(path: Path) -> bool:
    return path.is_dir() and all((path / m).exists() for m in _MARKERS)


def _candidates(explicit: str | Path | None) -> list[Path]:
    here = Path(__file__).resolve().parent          # .../eval
    roots = [here.parent, here.parent.parent, here.parent.parent.parent]
    out: list[Path] = []
    env = os.environ.get("THIRDSPACE_SRC")
    if env:
        out.append(Path(env).expanduser().resolve())
    if explicit:
        out.append(Path(explicit).expanduser().resolve())
    for root in roots:
        out.extend([root / "src", root / "thirdspace", root / "third_space", root])
    return out


def ensure_src_on_path(explicit: str | Path | None = None) -> Path:
    """Idempotently add the ThirdSpace source root to sys.path and return it."""
    global _resolved
    if _resolved is not None and explicit is None:
        return _resolved

    for candidate in _candidates(explicit):
        if _looks_like_src(candidate):
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            _resolved = candidate
            return candidate

    searched = "\n  ".join(str(c) for c in _candidates(explicit))
    raise ImportError(
        "Could not locate the ThirdSpace source root (a directory containing "
        f"{' and '.join(_MARKERS)}).\nSearched:\n  {searched}\n"
        "Set THIRDSPACE_SRC=/path/to/src, or pass --src on the command line."
    )


def import_pipeline(explicit: str | Path | None = None):
    """Import the pipeline modules the harness needs, after fixing sys.path.

    Returns a simple namespace so callers do not each repeat the bootstrap.
    Imported lazily: `python -m eval inspect` should work with no boto3
    credentials and no langgraph installed.
    """
    ensure_src_on_path(explicit)

    import importlib
    from types import SimpleNamespace

    models = importlib.import_module("models")
    graph = importlib.import_module("graph")
    llm = importlib.import_module("llm")
    budget = importlib.import_module("budget")
    prompts = importlib.import_module("utils.prompts")
    return SimpleNamespace(
        models=models, graph=graph, llm=llm, budget=budget, prompts=prompts
    )


def import_frame_reader(explicit: str | Path | None = None):
    """Import ONLY utils.prompts.read_cultural_frames / render_frame.

    Dataset loading needs the project's frame-reading rules but must not drag
    in boto3 or langgraph, so this is separated from import_pipeline.
    """
    ensure_src_on_path(explicit)
    import importlib

    prompts = importlib.import_module("utils.prompts")
    return prompts.read_cultural_frames, prompts.render_frame

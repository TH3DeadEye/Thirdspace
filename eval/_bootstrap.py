"""Path resolution between the two folders: `thirdspace/` (the algorithm) and
`eval/` (this package).

The separation is deliberate — eval imports thirdspace, never the reverse, and
nothing in thirdspace/ knows this package exists. That one-way edge is what
keeps the pipeline runnable on its own (`python thirdspace/main.py`) while the
harness stays a pure consumer of it.

Three jobs, in the order they must happen:

  1. find the source tree   — the directory holding models.py / graph.py, which
                              must go ON sys.path (not be imported as a package)
                              because the pipeline uses flat absolute imports:
                              `from models import ...`, `from utils import prompts`.
  2. load .env              — llm.py reads BEDROCK_MODEL_ID at MODULE IMPORT
                              time. main.py gets away with loading dotenv first;
                              anything else importing llm must do the same or
                              the model ids arrive empty and every call fails
                              with a ValidationException.
  3. import the pipeline    — only after 1 and 2.

`doctor()` checks all of it without running anything, and is wired to
`python -m eval doctor`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Modules that identify a ThirdSpace source root.
_MARKERS = ("models.py", "graph.py", "agents.py")
# The utils subpackage the pipeline imports as `utils.prompts` / `utils.parsing`.
_UTILS_MARKER = Path("utils") / "prompts.py"
# Decorator JSONs prompts.py loads at import time — a missing one is an
# ImportError three frames deep, so it is worth naming here.
_REQUIRED_DECORATORS = ("first-principles", "red-team", "blind-spots")

_src_root: Path | None = None
_env_loaded = False


# ---------------------------------------------------------------------------
# Locating the source tree
# ---------------------------------------------------------------------------

def _looks_like_src(path: Path) -> bool:
    return path.is_dir() and all((path / m).exists() for m in _MARKERS)


def _candidates(explicit: str | Path | None) -> list[Path]:
    """Every place the algorithm folder might live, most specific first."""
    here = Path(__file__).resolve().parent            # .../<repo>/eval
    out: list[Path] = []
    env = os.environ.get("THIRDSPACE_SRC")
    if env:
        out.append(Path(env).expanduser().resolve())
    if explicit:
        out.append(Path(explicit).expanduser().resolve())
    # Walk up from eval/: sibling folder first, then the repo root itself.
    for root in (here.parent, here.parent.parent, here.parent.parent.parent):
        out.extend([
            root / "thirdspace",
            root / "third_space",
            root / "src",
            root / "thirdspace" / "thirdspace",
            root / "src" / "thirdspace",
            root,
        ])
    # Never treat eval/ as the source tree.
    seen: set[Path] = set()
    unique = []
    for c in out:
        if c != here and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def ensure_src_on_path(explicit: str | Path | None = None) -> Path:
    """Add the algorithm folder to sys.path (idempotent) and return it."""
    global _src_root
    if _src_root is not None and explicit is None:
        return _src_root

    for candidate in _candidates(explicit):
        if _looks_like_src(candidate):
            if str(candidate) not in sys.path:
                # Prepend: the pipeline's `metrics.py` must win over any
                # same-named module further down sys.path.
                sys.path.insert(0, str(candidate))
            _src_root = candidate
            return candidate

    searched = "\n  ".join(str(c) for c in _candidates(explicit))
    raise ImportError(
        "Could not find the ThirdSpace algorithm folder (a directory containing "
        f"{', '.join(_MARKERS)}).\nSearched:\n  {searched}\n\n"
        "Expected layout:\n"
        "  <repo>/thirdspace/   models.py graph.py agents.py llm.py budget.py utils/\n"
        "  <repo>/eval/         this package\n\n"
        "Fix with:  export THIRDSPACE_SRC=/path/to/thirdspace   (or pass --src)"
    )


def get_src_root() -> Path | None:
    return _src_root


def find_repo_root(explicit: str | Path | None = None) -> Path:
    """The directory that holds thirdspace/ and eval/.

    Everything the harness writes (outputs/eval/...) and reads by relative path
    (data/MDD_100.csv) is anchored here rather than to the shell's working
    directory, so `python -m eval run ...` behaves the same from anywhere.
    """
    try:
        src = ensure_src_on_path(explicit)
    except ImportError:
        return Path(__file__).resolve().parent.parent
    here = Path(__file__).resolve().parent.parent      # the folder holding eval/
    # Layout A: <repo>/thirdspace + <repo>/eval  -> src.parent is the repo root.
    if src.parent == here or src.parent.exists() and (src.parent / "eval").exists():
        return src.parent
    # Layout B: the source tree IS the repo root (flat).
    return here


def resolve_path(path: str | Path | None, explicit_src: str | Path | None = None) -> Path | None:
    """Resolve a user-supplied path against CWD first, then the repo root.

    `--path data/MDD_100.csv` should work whether it is run from the repo root
    or from inside eval/.
    """
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_absolute() or p.exists():
        return p.resolve()
    candidate = find_repo_root(explicit_src) / p
    return candidate.resolve() if candidate.exists() else p


def default_out_dir(explicit_src: str | Path | None = None) -> Path:
    return find_repo_root(explicit_src) / "outputs" / "eval"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _parse_env_file(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader, so eval works without python-dotenv."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_env(explicit_src: str | Path | None = None) -> Path | None:
    """Load <repo>/.env before any pipeline module is imported.

    main.py does this itself; the harness does not go through main.py, so it
    must. Existing environment variables always win — an exported override on
    the command line should not be clobbered by the file.
    """
    global _env_loaded
    if _env_loaded:
        return None
    _env_loaded = True

    root = find_repo_root(explicit_src)
    for candidate in (root / ".env", Path.cwd() / ".env"):
        if not candidate.exists():
            continue
        try:
            from dotenv import load_dotenv
            load_dotenv(candidate)
        except ImportError:
            for key, value in _parse_env_file(candidate).items():
                os.environ.setdefault(key, value)
        return candidate
    return None


# Environment variables the pipeline needs. BEDROCK_MODEL_ID and
# CHALLENGER_MODEL_ID have EMPTY defaults in llm.py, so they are mandatory.
REQUIRED_ENV = ("BEDROCK_MODEL_ID", "CHALLENGER_MODEL_ID")
OPTIONAL_ENV = ("JUDGE_MODEL_ID", "BEDROCK_EMBED_MODEL_ID", "AWS_REGION",
                "THIRDSPACE_CELL_WORKERS", "LANGSMITH_API_KEY")


# ---------------------------------------------------------------------------
# Importing the pipeline
# ---------------------------------------------------------------------------

def import_pipeline(explicit: str | Path | None = None):
    """sys.path -> .env -> import. Returns a namespace of pipeline modules.

    Lazy on purpose: `eval inspect` and `eval report` must work on a machine
    with no boto3, no langgraph and no AWS credentials.
    """
    ensure_src_on_path(explicit)
    load_env(explicit)

    import importlib
    from types import SimpleNamespace

    missing = [k for k in REQUIRED_ENV if not os.environ.get(k, "").strip()]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} is empty. llm.py declares these with no "
            f"default, so every Bedrock call would fail. Set them in <repo>/.env "
            f"or export them. Run `python -m eval doctor` for the full check."
        )

    return SimpleNamespace(
        models=importlib.import_module("models"),
        graph=importlib.import_module("graph"),
        llm=importlib.import_module("llm"),
        budget=importlib.import_module("budget"),
        prompts=importlib.import_module("utils.prompts"),
        src_root=get_src_root(),
    )


def import_frame_reader(explicit: str | Path | None = None):
    """Import only utils.prompts.read_cultural_frames / render_frame.

    Dataset loading needs the project's frame-reading rules but must not drag
    in boto3 or langgraph, so this is kept separate from import_pipeline.
    Note utils.prompts still imports models and the decorator registry, so a
    broken registry surfaces here.
    """
    ensure_src_on_path(explicit)
    import importlib

    prompts = importlib.import_module("utils.prompts")
    return prompts.read_cultural_frames, prompts.render_frame


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def doctor(explicit: str | Path | None = None, data_path: str | Path | None = None) -> list[dict]:
    """Check every path, import and environment variable a run depends on.

    Returns a list of {name, ok, detail}. Costs nothing and calls nothing.
    """
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    here = Path(__file__).resolve().parent
    add("eval package", True, str(here))

    try:
        src = ensure_src_on_path(explicit)
        add("algorithm folder", True, str(src))
    except ImportError as exc:
        add("algorithm folder", False, str(exc).splitlines()[0])
        return checks

    root = find_repo_root(explicit)
    add("repo root", True, str(root))
    add("one-way import edge", not (src / "eval").exists(),
        "eval/ is NOT inside the algorithm folder"
        if not (src / "eval").exists()
        else f"eval/ found inside {src} — move it up one level")

    for marker in _MARKERS:
        add(f"{marker}", (src / marker).exists(), str(src / marker))
    add("utils/prompts.py", (src / _UTILS_MARKER).exists(), str(src / _UTILS_MARKER))
    add("utils/parsing.py", (src / "utils" / "parsing.py").exists(),
        "imported by llm.py as utils.parsing")
    add("utils/__init__.py", (src / "utils" / "__init__.py").exists(),
        "required for `from utils import prompts` to resolve as a package")

    registry = src / "utils" / "decorator_registry"
    add("utils/decorator_registry/", registry.is_dir(),
        str(registry) + (" (decorators.py resolves it as Path(__file__).parent"
                         " / 'decorator_registry', so it must sit beside"
                         " decorators.py)" if not registry.is_dir() else ""))
    if registry.is_dir():
        for name in _REQUIRED_DECORATORS:
            add(f"decorator {name}.json", (registry / f"{name}.json").exists(),
                "loaded at import time by utils/prompts.py")

    env_file = load_env(explicit)
    exported = all(os.environ.get(k, "").strip() for k in REQUIRED_ENV)
    add(".env", env_file is not None or exported,
        str(env_file) if env_file
        else (f"none at {root / '.env'}, but the variables are exported" if exported
              else f"no .env at {root / '.env'} and nothing exported"))
    for key in REQUIRED_ENV:
        value = os.environ.get(key, "").strip()
        add(f"env {key}", bool(value), value or "EMPTY (no default in llm.py)")
    for key in OPTIONAL_ENV:
        value = os.environ.get(key, "").strip()
        add(f"env {key}", True, value or "(unset — using llm.py default)")

    for module in ("pydantic", "langgraph", "boto3", "langsmith", "json_repair"):
        try:
            __import__(module)
            add(f"import {module}", True, "ok")
        except ImportError as exc:
            add(f"import {module}", False, str(exc))
    try:
        __import__("datasets")
        add("import datasets", True, "NormAd can load from the Hub")
    except ImportError:
        add("import datasets", True, "not installed — NormAd needs --path to a local CSV")

    for module in ("models", "utils.prompts"):
        try:
            __import__(module)
            add(f"pipeline import {module}", True, "ok")
        except Exception as exc:                      # noqa: BLE001
            add(f"pipeline import {module}", False, f"{type(exc).__name__}: {exc}")

    out = default_out_dir(explicit)
    try:
        out.mkdir(parents=True, exist_ok=True)
        add("output dir writable", os.access(out, os.W_OK), str(out))
    except OSError as exc:
        add("output dir writable", False, f"{out}: {exc}")

    if data_path:
        resolved = resolve_path(data_path, explicit)
        add("dataset file", resolved is not None and resolved.exists(), str(resolved))

    return checks

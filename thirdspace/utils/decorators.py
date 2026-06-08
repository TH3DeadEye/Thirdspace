import json
from pathlib import Path

_REGISTRY = Path(__file__).parent / "decorator_registry"

def load_decorator(name:str, **params) -> str:
    """
    name is the filename stem, e.g. "blind-spots".
    Parameters pull extra text from the JSON's parameterMapping:
      - enum params (depth, position) use valueMap[value]
      - scalar/array params (categories, perspectives) use a format string
    """

    spec = json.loads((_REGISTRY / f"{name}.json").read_text())
    tt = spec["transformationTemplate"]
    parts = [tt["instruction"]]

    pmap = tt.get("parameterMapping", {})
    for param, value in params.items():
        cfg = pmap.get(param,{})
        if "valueMap" in cfg:
            extra = cfg["valueMap"].get(str(value))
            if extra:
                parts.append(extra)
        elif "format" in cfg:                       # scalar or array param
            v = ", ".join(value) if isinstance(value, (list, tuple)) else value
            parts.append(cfg["format"].format(value=v))
    return " ".join(parts)


def decorate(system_prompt: str, *blocks: str) -> str:
    """Prepend one or more decorator instruction blocks to a system prompt."""
    if not blocks:
        return system_prompt
    return "\n\n".join([*blocks, system_prompt])

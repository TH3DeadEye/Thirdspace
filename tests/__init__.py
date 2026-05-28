"""
Test package init. Makes `thirdspace/` importable both under
`python -m unittest discover tests` and `pytest tests`.
"""
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "thirdspace"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

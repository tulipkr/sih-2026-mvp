"""
Ensures the project root (ais_attribution/) is importable as `src.xxx`
regardless of the working directory pytest is invoked from -- this is what
lets `from src.search_window import ...` resolve reliably in CI/another
machine, without relying on the caller happening to `cd` into this exact
folder first.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

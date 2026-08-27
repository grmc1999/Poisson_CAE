"""Ensures the repo root is importable when running pytest from tests/.

Adds the repository root to sys.path so that `Utils` and `models` packages
resolve regardless of the invoking directory.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

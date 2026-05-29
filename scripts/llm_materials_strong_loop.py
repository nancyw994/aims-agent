#!/usr/bin/env python
"""Generic entrypoint for LLM-guided materials robustness loops."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from llm_spall_strong_loop import main


if __name__ == "__main__":
    main()

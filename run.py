#!/usr/bin/env python3
"""Entry point: python run.py [options]"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running straight from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vdvr.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

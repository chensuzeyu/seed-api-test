#!/usr/bin/env python3
"""
End-to-end Seedance 2.0 style-transfer smoke pipeline:

1) extract first frame
2) Seedream 5.0 Pro (Lite fallback) style edit — 科技夜晚
3) Seedance 2.0 edit-video mode — fixed 720p / 4s
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_step(script: str) -> None:
    cmd = [sys.executable, str(ROOT / script)]
    print(f"\n######## RUN: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    run_step("extract_first_frame.py")
    run_step("run_seedream_style_first_frame.py")
    run_step("run_seedance2_style_transfer.py")
    print("\n[OK] pipeline finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

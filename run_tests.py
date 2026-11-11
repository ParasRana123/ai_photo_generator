#!/usr/bin/env python3
"""
Master test runner for AI Photo Generator (Phases 1, 2 & 3).
Executes Python pytest suite and Express Node.js API test suite.
"""

import subprocess
import sys
from pathlib import Path
import pytest


def main():
    print("=" * 60)
    print("  RUNNING PYTHON TEST SUITE (PHASES 1, 2 & 3 AI API)")
    print("=" * 60)

    py_exit = pytest.main([
        "tests",
        "-v",
        "--tb=short"
    ])

    print("\n" + "=" * 60)
    print("  RUNNING EXPRESS BACKEND TEST SUITE (PHASE 3)")
    print("=" * 60)

    backend_dir = Path(__file__).parent / "backend"
    node_res = subprocess.run(
        ["node", "--test", "tests/api.test.js"],
        cwd=backend_dir
    )

    print("\n" + "=" * 60)
    if py_exit == 0 and node_res.returncode == 0:
        print("  ALL PYTHON & NODE.JS TESTS PASSED SUCCESSFULLY! [OK]")
        sys.exit(0)
    else:
        print(f"  TEST FAILURES DETECTED (Python: {py_exit}, Node: {node_res.returncode})")
        sys.exit(1)


if __name__ == "__main__":
    main()

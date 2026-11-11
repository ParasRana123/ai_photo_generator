#!/usr/bin/env python3
"""
Test runner script for Phase 1 AI Photo Generator.
Executes pytest across unit and integration tests and prints structured results.
"""

import sys
import pytest


def main():
    print("=" * 60)
    print("  RUNNING AI PHOTO GENERATOR TEST SUITE (PHASES 1 & 2)")
    print("=" * 60)

    exit_code = pytest.main([
        "tests",
        "-v",
        "--tb=short"
    ])

    print("\n" + "=" * 60)
    if exit_code == 0:
        print("  ALL TESTS PASSED SUCCESSFULLY! [OK]")
    else:
        print(f"  TESTS FINISHED WITH EXIT CODE: {exit_code}")
    print("=" * 60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Unified Phase 3 Full-Stack Launcher.
Starts all three services concurrently:
  1. Python FastAPI AI Engine (http://127.0.0.1:8000)
  2. Express.js Backend Gateway (http://127.0.0.1:5000)
  3. React Vite Frontend Studio (http://localhost:5173)
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Start PersonaAI Full-Stack Web Application")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Disable dry-run mock mode on Python AI engine and use real SDXL weights"
    )
    args = parser.parse_args()

    root = Path(__file__).parent.resolve()
    python_exe = sys.executable

    env = os.environ.copy()
    env["AI_MOCK_MODE"] = "false" if args.production else "true"

    print("=" * 65)
    print("  AI PHOTO GENERATOR - PHASE 3 FULL-STACK LAUNCHER")
    print("=" * 65)
    print("  1. Python AI Engine : http://127.0.0.1:8000")
    print("  2. Express Gateway  : http://127.0.0.1:5000")
    print("  3. React Frontend   : http://localhost:5173")
    print(f"  AI Mock Mode        : {env['AI_MOCK_MODE']}")
    print("  Press Ctrl+C to stop all services.")
    print("=" * 65)

    processes = []
    try:
        # 1. Start Python FastAPI server
        py_cmd = [python_exe, str(root / "python_api.py")]
        if args.production:
            py_cmd.append("--production")
        processes.append(subprocess.Popen(py_cmd, cwd=root, env=env))

        # 2. Start Express backend
        express_cmd = ["node", str(root / "backend" / "src" / "server.js")]
        processes.append(subprocess.Popen(express_cmd, cwd=root / "backend", env=env))

        # 3. Start Vite React frontend
        npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
        frontend_cmd = [npm_cmd, "run", "dev"]
        processes.append(subprocess.Popen(frontend_cmd, cwd=root / "frontend", env=env))

        while True:
            for proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(f"A service exited unexpectedly with code {proc.returncode}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down all Phase 3 services...")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        print("[OK] All services stopped.")


if __name__ == "__main__":
    main()


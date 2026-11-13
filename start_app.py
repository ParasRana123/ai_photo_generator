#!/usr/bin/env python3
"""
Unified Phase 3 Full-Stack Launcher.
Starts all three services sequentially with health verification:
  1. Python FastAPI AI Engine (http://127.0.0.1:8000)
  2. Express.js Backend Gateway (http://127.0.0.1:5000)
  3. React Vite Frontend Studio (http://localhost:5173)
"""

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Checks whether a local TCP port is already open/listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_service(url: str, name: str, timeout: int = 25) -> bool:
    """Polls a health endpoint until it returns HTTP 200 or times out."""
    start = time.time()
    print(f"  [WAIT] Initializing {name}...", end="", flush=True)
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    print(f" [OK] Ready ({round(time.time() - start, 1)}s)")
                    return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(0.8)
    print(" [WARNING] Timeout reached")
    return False


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
    print("  AI PHOTO GENERATOR - FULL-STACK ORCHESTRATOR")
    print("=" * 65)

    # Pre-flight check: detect already running instances
    ports_status = {
        8000: ("Python AI Engine", is_port_in_use(8000)),
        5000: ("Express Gateway", is_port_in_use(5000)),
        5173: ("React Frontend", is_port_in_use(5173)),
    }

    in_use = [f"{desc} (port {port})" for port, (desc, used) in ports_status.items() if used]
    if in_use:
        print("[NOTICE] Existing active services detected:")
        for item in in_use:
            print(f"  • {item} is already listening")
        print("  Reusing active services where available.\n")

    processes = []
    try:
        # 1. Start Python FastAPI server if not already running
        if not ports_status[8000][1]:
            py_cmd = [python_exe, str(root / "python_api.py")]
            if args.production:
                py_cmd.append("--production")
            py_proc = subprocess.Popen(py_cmd, cwd=root, env=env)
            processes.append(py_proc)
            wait_for_service("http://127.0.0.1:8000/health", "Python FastAPI AI Engine", timeout=30)
        else:
            print("  [OK] Python AI Engine already running on http://127.0.0.1:8000")

        # 2. Start Express backend if not already running
        if not ports_status[5000][1]:
            express_cmd = ["node", str(root / "backend" / "src" / "server.js")]
            exp_proc = subprocess.Popen(express_cmd, cwd=root / "backend", env=env)
            processes.append(exp_proc)
            wait_for_service("http://127.0.0.1:5000/api/health", "Express API Gateway", timeout=15)
        else:
            print("  [OK] Express Gateway already running on http://127.0.0.1:5000")

        # 3. Start Vite React frontend if not already running
        if not ports_status[5173][1]:
            npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
            frontend_cmd = [npm_cmd, "run", "dev"]
            vite_proc = subprocess.Popen(frontend_cmd, cwd=root / "frontend", env=env)
            processes.append(vite_proc)
            wait_for_service("http://localhost:5173", "React Vite Frontend", timeout=15)
        else:
            print("  [OK] React Frontend already running on http://localhost:5173")

        print("=" * 65)
        print("  ALL SERVICES ARE RUNNING AND FULLY CONNECTED!")
        print("  • Python AI Engine : http://127.0.0.1:8000")
        print("  • Express Gateway  : http://127.0.0.1:5000")
        print("  • Web Application  : http://localhost:5173")
        print("=" * 65)
        print("  Press Ctrl+C in this terminal anytime to stop all services.\n")

        while True:
            for proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError(f"A service exited unexpectedly with code {proc.returncode}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[INFO] Gracefully shutting down all services...")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        print("[OK] All services stopped.")


if __name__ == "__main__":
    main()



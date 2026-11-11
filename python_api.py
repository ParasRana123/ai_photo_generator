#!/usr/bin/env python3
"""
Entry point to launch the Python FastAPI AI microservice.
Runs Uvicorn serving SDXL LoRA training and inference endpoints.
"""

import argparse
import os
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start Python FastAPI AI Service")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--production", action="store_true", help="Disable mock mode and use real SDXL weights")
    args = parser.parse_args()

    if args.production:
        os.environ["AI_MOCK_MODE"] = "false"
    else:
        os.environ.setdefault("AI_MOCK_MODE", "true")

    print("=" * 60)
    print("  AI PHOTO GENERATOR - PYTHON FASTAPI AI ENGINE")
    print("=" * 60)
    print(f" Listening on: http://{args.host}:{args.port}")
    print(f" Mock Mode:    {os.environ['AI_MOCK_MODE']}")
    print("=" * 60)

    uvicorn.run("src.api.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()


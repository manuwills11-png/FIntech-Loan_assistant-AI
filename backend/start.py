"""
FinEdge – Development server startup script.

Usage:
    python start.py

This script:
  1. Trains the ML model if not already present.
  2. Starts the FastAPI server via Uvicorn.
"""

import os
import subprocess
import sys
from pathlib import Path

# Ensure UTF-8 encoding for stdout/stderr on Windows (avoids UnicodeEncodeError
# when logging translated text containing non-Latin characters like Hindi).
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

MODEL_PATH = Path("./ml/loan_risk_model.pkl")


def train_model_if_needed():
    if not MODEL_PATH.exists():
        print("[FinEdge] ML model not found. Training now...")
        result = subprocess.run([sys.executable, "ml/train_model.py"], check=False)
        if result.returncode != 0:
            print("[FinEdge] Model training failed. Exiting.")
            sys.exit(1)
        print("[FinEdge] Model trained successfully.")
    else:
        print(f"[FinEdge] ML model found at {MODEL_PATH}.")


if __name__ == "__main__":
    train_model_if_needed()

    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    print(f"[FinEdge] Starting server at http://{host}:{port}")
    print(f"[FinEdge] Swagger docs: http://localhost:{port}/docs")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )

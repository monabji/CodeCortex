"""Launch one local Phase 5 API worker with the frozen Phase 4 model."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts/phase4")
    args = parser.parse_args()
    if min(args.batch_size, args.cpu_threads) < 1 or not 1 <= args.port <= 65535:
        parser.error("Use positive batch/thread counts and a valid port")
    os.environ.update(MUTANTSCOPE_DEVICE=args.device, MUTANTSCOPE_BATCH_SIZE=str(args.batch_size),
                      MUTANTSCOPE_CPU_THREADS=str(args.cpu_threads), MUTANTSCOPE_ARTIFACTS=str(args.artifacts_dir.resolve()))
    import uvicorn
    from mutantscope.api import app
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()

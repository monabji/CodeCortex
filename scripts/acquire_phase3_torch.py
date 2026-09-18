"""Resume the pinned CUDA wheel using verified HTTP byte ranges.

Useful when a large uninterrupted pip transfer stalls. The final publisher
SHA-256 must match before this helper promotes the wheel for installation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://download.pytorch.org/whl/cu130/torch-2.12.1%2Bcu130-cp314-cp314-win_amd64.whl"
SIZE = 1933532482
SHA256 = "d1cd8a4fd0556b2604db5447d9298323b8fba1ba67501fb3d8b22485c764a3a6"
CHUNK = 8 * 1024 * 1024


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-partial", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be from 1 to 8")
    directory = ROOT / "data/cache/phase3/runtime"
    directory.mkdir(parents=True, exist_ok=True)
    wheel = directory / "torch-2.12.1+cu130-cp314-cp314-win_amd64.whl"
    if wheel.exists():
        if digest(wheel) != SHA256:
            raise ValueError("Existing wheel has a wrong checksum")
        print(f"Verified wheel: {wheel}", flush=True)
        return
    partial, progress = wheel.with_suffix(".whl.part"), wheel.with_suffix(".progress.json")
    done = set()
    if progress.exists():
        state = json.loads(progress.read_text())
        if state["sha256"] != SHA256 or state["size"] != SIZE or not partial.exists():
            raise ValueError("Incompatible wheel download progress")
        done = set(state["chunks"])
    elif args.seed_partial:
        if not 0 < args.seed_partial.stat().st_size < SIZE:
            raise ValueError("Seed must be an incomplete prefix of the selected wheel")
        shutil.copyfile(args.seed_partial, partial)
        done = set(range(partial.stat().st_size // CHUNK))
    else:
        partial.touch(exist_ok=False)
    with partial.open("r+b") as stream:
        stream.truncate(SIZE)
    lock = threading.Lock()
    started = time.monotonic()
    initial = len(done)
    total = (SIZE + CHUNK - 1) // CHUNK

    def save_progress():
        temporary = progress.with_suffix(".json.part")
        temporary.write_text(json.dumps({"sha256": SHA256, "size": SIZE, "chunks": sorted(done)}))
        temporary.replace(progress)

    save_progress()

    def fetch(index):
        start, end = index * CHUNK, min((index + 1) * CHUNK, SIZE) - 1
        for attempt in range(6):
            try:
                request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                        raise ValueError("Server did not return the requested wheel byte range")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise ValueError("Incomplete wheel range")
                with partial.open("r+b") as stream:
                    stream.seek(start)
                    stream.write(data)
                with lock:
                    done.add(index)
                    save_progress()
                    elapsed = time.monotonic() - started
                    rate = (len(done) - initial) * CHUNK / elapsed / 1024 / 1024
                    print(json.dumps({"chunks": len(done), "total_chunks": total,
                                      "MB_per_second": round(rate, 2)}), flush=True)
                return
            except Exception as error:
                if attempt == 5:
                    raise
                print(json.dumps({"retry_chunk": index, "attempt": attempt + 1,
                                  "reason": str(error)}), flush=True)
                time.sleep(min(5 * (attempt + 1), 25))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch, [i for i in range(total) if i not in done]))
    if digest(partial) != SHA256:
        raise ValueError("Downloaded wheel failed publisher SHA-256; do not install it")
    partial.replace(wheel)
    print(f"Verified wheel: {wheel}", flush=True)


if __name__ == "__main__":
    main()

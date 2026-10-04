"""Download pinned HF snapshots without loading Torch. Run before the lab.

Uses only Python's standard library; reruns reuse verified complete files.
"""

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / ".models"
MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-m3",
    "BAAI/bge-reranker-v2-m3",
]


def download_weights(url, destination):
    """Download independently verified HTTP ranges and assemble atomically."""
    request = urllib.request.Request(url + "?range=0-0", headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or "/" not in content_range:
            raise OSError("Model host does not support byte ranges")
        size = int(content_range.split("/")[-1])
    step = 16 * 1024 * 1024
    parts = destination.parent / (destination.name + ".parts")
    parts.mkdir(exist_ok=True)

    def fetch(start):
        end = min(size - 1, start + step - 1)
        target = parts / str(start)
        if target.exists() and target.stat().st_size == end - start + 1:
            return target
        for attempt in range(3):
            try:
                request = urllib.request.Request(
                    url + f"?range={start}-{end}",
                    headers={"Range": f"bytes={start}-{end}"},
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    if (
                        response.headers.get("Content-Range")
                        != f"bytes {start}-{end}/{size}"
                    ):
                        raise OSError("Model host returned an unexpected byte range")
                    data = response.read()
                if len(data) != end - start + 1:
                    raise OSError("Incomplete model range")
                target.write_bytes(data)
                return target
            except Exception:
                if attempt == 2:
                    raise

    ranges = list(range(0, size, step))
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(fetch, start) for start in ranges]
        for i, future in enumerate(as_completed(futures)):
            future.result()
            if (i + 1) % 8 == 0:
                print(
                    f"{destination.parent.name}: {min((i + 1) * step, size) / 1024**2:.0f}/{size / 1024**2:.0f} MiB",
                    flush=True,
                )
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as output:
        for start in ranges:
            with (parts / str(start)).open("rb") as source:
                while block := source.read(4 * 1024 * 1024):
                    output.write(block)
    if temporary.stat().st_size != size:
        raise OSError("Incomplete assembled model")
    temporary.replace(destination)
    # Delete only the known numbered part files inside this snapshot directory.
    for start in ranges:
        (parts / str(start)).unlink()
    parts.rmdir()


def download_model(name):
    with urllib.request.urlopen(
        f"https://huggingface.co/api/models/{name}", timeout=60
    ) as response:
        info = json.load(response)
    revision = info["sha"]
    directory = ROOT / name.replace("/", "--")
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "download_manifest.json").exists():
        print(f"Already complete: {name}", flush=True)
        return
    files = [
        item["rfilename"]
        for item in info["siblings"]
        if (
            "/" not in item["rfilename"]
            and item["rfilename"].endswith((".json", ".model", ".safetensors", ".bin"))
        )
        or item["rfilename"] == "1_Pooling/config.json"
    ]
    # Avoid fetching both equivalent weight formats.
    if "model.safetensors" in files:
        files = [f for f in files if f != "pytorch_model.bin"]
    for filename in files:
        destination = directory / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            continue
        if filename.endswith((".safetensors", ".bin")):
            download_weights(
                f"https://huggingface.co/{name}/resolve/{revision}/{filename}",
                destination,
            )
            print(f"Downloaded {name}/{filename}", flush=True)
            continue
        temporary = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(3):
            try:
                url = f"https://huggingface.co/{name}/resolve/{revision}/{filename}"
                with (
                    urllib.request.urlopen(url, timeout=120) as response,
                    temporary.open("wb") as output,
                ):
                    size = int(response.headers.get("Content-Length", 0))
                    written, previous = 0, time.monotonic()
                    while block := response.read(4 * 1024 * 1024):
                        output.write(block)
                        written += len(block)
                        if time.monotonic() - previous > 30:
                            print(
                                f"{name}: {filename} {written / 1024**2:.0f} MiB",
                                flush=True,
                            )
                            previous = time.monotonic()
                    if size and written != size:
                        raise OSError("Incomplete model download")
                temporary.replace(destination)
                print(f"Downloaded {name}/{filename}", flush=True)
                break
            except Exception:
                if attempt == 2:
                    raise
    (directory / "download_manifest.json").write_text(
        json.dumps({"model": name, "revision": revision, "files": files}, indent=2),
        encoding="utf-8",
    )
    print(f"Complete: {name}", flush=True)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download_model, MODELS))

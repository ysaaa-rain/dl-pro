"""Resumable, Windows-compatible recovery of the two pinned WT GEO matrices."""
import gzip
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import requests
import pandas as pd

CHUNK = 4 * 1024 * 1024
WORKERS = 8
ROOT = Path("data/raw")
ROOT.mkdir(parents=True, exist_ok=True)
samples = pd.read_csv("data/annotations/samples.csv")
wt = samples[samples.condition == "37C"].sort_values("replicate")
lengths = {1: 470399779, 2: 456862495}


def download_chunk(url, target, index, length):
    start = index * CHUNK
    end = min(start + CHUNK, length) - 1
    for attempt in range(5):
        try:
            response = requests.get(
                url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 90)
            )
            response.raise_for_status()
            assert response.status_code == 206
            assert response.headers.get("Content-Range") == f"bytes {start}-{end}/{length}"
            assert len(response.content) == end - start + 1
            with target.open("r+b") as out:
                out.seek(start)
                out.write(response.content)
                out.flush()
            return index
        except Exception:
            if attempt == 4:
                raise


def fetch_one(row, compressed_length=None):
    repeat = int(row.replicate)
    length = lengths[repeat] if compressed_length is None else int(compressed_length)
    label = f"{row.condition} repeat {repeat}"
    compressed = ROOT / (Path(row.path).name + ".gz")
    partial = ROOT / (compressed.name + ".ranges.part")
    journal = ROOT / (compressed.name + ".ranges.json")
    if not compressed.exists() or compressed.stat().st_size != length:
        count = math.ceil(length / CHUNK)
        fresh_partial = not partial.exists()
        if fresh_partial:
            with partial.open("wb") as out:
                out.truncate(length)
        assert partial.stat().st_size == length
        completed = (set(json.loads(journal.read_text())["completed"])
                     if journal.exists() and not fresh_partial else set())
        pending = [i for i in range(count) if i not in completed]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(download_chunk, row.source_url, partial, i, length) for i in pending]
            for future in as_completed(futures):
                index = future.result()
                completed.add(index)
                journal.write_text(json.dumps({"length": length, "completed": sorted(completed)}))
                if len(completed) % 8 == 0 or len(completed) == count:
                    print(f"{label}: {len(completed)}/{count} chunks", flush=True)
        assert len(completed) == count
        partial.replace(compressed)
    assert compressed.stat().st_size == length
    compressed_hash = hashlib.sha256()
    with compressed.open("rb") as source:
        while data := source.read(8 * 1024 * 1024):
            compressed_hash.update(data)
    if isinstance(row.gzip_sha256, str) and len(row.gzip_sha256) == 64:
        assert compressed_hash.hexdigest() == row.gzip_sha256
    print(f"{label}: compressed complete", flush=True)
    output = Path(row.path)
    if not output.exists() or output.stat().st_size != int(row.bytes):
        temporary = output.with_suffix(output.suffix + ".decompress.part")
        digest = hashlib.sha256()
        with gzip.open(compressed, "rb") as source, temporary.open("wb") as sink:
            while data := source.read(8 * 1024 * 1024):
                digest.update(data)
                sink.write(data)
        assert temporary.stat().st_size == int(row.bytes)
        assert digest.hexdigest() == row.sha256
        temporary.replace(output)
    digest = hashlib.sha256()
    with output.open("rb") as source:
        while data := source.read(8 * 1024 * 1024):
            digest.update(data)
    assert digest.hexdigest() == row.sha256
    print(f"{label}: Cooler SHA256 verified", flush=True)
    return dict(sample_id=row.sample_id, source_url=row.source_url,
                gzip_bytes=length, gzip_sha256=compressed_hash.hexdigest(),
                gzip_hash_previously_registered=isinstance(row.gzip_sha256, str),
                cooler_sha256=digest.hexdigest(), cooler_bytes=int(row.bytes),
                registered_cooler_sha256=row.sha256)


if __name__ == "__main__":
    verification = [fetch_one(row) for row in wt.itertuples()]
    output = Path("results/wt_source_restore/verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")

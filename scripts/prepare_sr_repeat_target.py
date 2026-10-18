"""Extract WT repeat-2 native targets at the frozen repeat-1 train/val loci.

The original split and source repeat-1 40 bp observations are unchanged.
Only train/validation repeat-2 reads are new; test repeat-2 reads are already
in the repository cache. The uncompressed target array is local-only.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from dlpro.experiment_core import ContactReader, save_json


OUT = Path("data/processed/sr_repeat_target")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv("data/processed/sr_expanded/windows.csv")
    original = np.load("data/processed/sr_expanded/raw_counts.npy", mmap_mode="r")
    chosen = frame[(frame.replicate == 1) & frame.split.isin(["train", "val"])]
    assert len(chosen) == 76 and sum(chosen.split == "train") == 64
    sample = pd.read_csv("data/annotations/samples.csv").query(
        'condition == "37C" and replicate == 2'
    ).iloc[0]
    source = Path(sample.path)
    assert source.stat().st_size == int(sample.bytes)
    assert sha256(source) == sample.sha256
    reader = ContactReader(source)
    target = np.lib.format.open_memmap(
        OUT / "raw_counts.npy", mode="w+", dtype="uint32", shape=(len(chosen), 1024, 1024)
    )
    index = []
    for out_index, (source_index, row) in enumerate(chosen.iterrows()):
        counts = reader.window(int(row.start) // 10, 1024, 10, False)
        if not np.array_equal(counts, np.rint(counts)):
            raise ValueError("Noninteger native repeat-2 counts")
        target[out_index] = counts.astype("uint32")
        index.append(dict(target_index=out_index, original_index=int(source_index),
                          start=int(row.start), group=int(row.group), split=row.split))
        if out_index % 8 == 0:
            print("repeat target", out_index + 1, "/", len(chosen), flush=True)
    target.flush()
    pd.DataFrame(index).to_csv(OUT / "windows.csv", index=False)
    test = frame[frame.split == "test"]
    assert len(test) == 32 and set(test.replicate) == {1, 2}
    assert original.shape == (len(frame), 1024, 1024)
    save_json(OUT / "preparation.json", dict(
        repeat2_source_sha256=sample.sha256, n_train=64, n_val=12,
        split_source="Original sr_expanded/windows.csv, unchanged",
        target="Raw WT repeat-2 10 bp symmetric integer contact matrices",
        existing_test="Original 16 paired test loci in sr_expanded/raw_counts.npy",
    ))


if __name__ == "__main__":
    main()

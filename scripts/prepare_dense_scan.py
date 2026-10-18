"""Extract 3.2 kb WT scan lattice with grouped sparse Cooler reads.

The even-indexed centers reproduce the frozen 6.4 kb cache. Grouping nearby
windows into a larger sparse read reduces repeated Cooler I/O without
interpolating any contacts.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from dlpro.experiment_core import ContactReader, LENGTH, centered_indices, save_json


OUT = Path("data/processed")
STRIDE_BP = 3200
GROUP_SIZE = 8


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while data := source.read(8 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def main():
    centers = np.arange(0, LENGTH, STRIDE_BP, dtype=int)
    assert len(centers) == 1451
    scan = pd.DataFrame({"center": centers, "background_block": centers // 100000})
    scan.to_csv(OUT / "scan_dense_windows.csv", index=False)
    samples = pd.read_csv("data/annotations/samples.csv")
    wt = samples[samples.condition == "37C"].sort_values("replicate")
    sources = {}
    checks = {}
    for row in wt.itertuples():
        repeat = int(row.replicate)
        path = Path(row.path)
        assert path.stat().st_size == int(row.bytes)
        assert sha256(path) == row.sha256
        sources[row.sample_id] = row.sha256
        reader = ContactReader(path)
        output = np.empty((len(scan), 2, 128, 128), dtype="float32")
        for first in range(0, len(scan), GROUP_SIZE):
            chunk = centers[first : first + GROUP_SIZE]
            starts = np.asarray([centered_indices(int(c)) for c in chunk])
            for channel, bin_bp in enumerate((100, 200)):
                offsets = starts[:, channel] - starts[0, channel]
                assert (offsets >= 0).all()
                image = reader.window(
                    int(starts[0, channel]),
                    n=int(offsets.max()) + 128,
                    bin_bp=bin_bp,
                )
                for j, offset in enumerate(offsets):
                    output[first + j, channel] = image[offset : offset + 128, offset : offset + 128]
            if first % 160 == 0:
                print("dense scan", repeat, first, "/", len(scan), flush=True)
        old = np.load(OUT / f"scan_rep{repeat}.npz")["raw"]
        assert len(output[::2]) == len(old)
        max_error = float(np.max(abs(output[::2] - old)))
        assert np.allclose(output[::2], old, rtol=2e-5, atol=2e-7), max_error
        checks[str(repeat)] = max_error
        np.savez_compressed(OUT / f"scan_dense_rep{repeat}.npz", raw=output)
        print("saved dense repeat", repeat, "old-grid max error", max_error, flush=True)
    save_json(
        "results/dense_scan/preparation.json",
        dict(
            centers=len(scan), stride_bp=STRIDE_BP, group_size=GROUP_SIZE,
            scales_bp=[100, 200], widths_bp=[12800, 25600],
            source_sha256=sources, old_grid_max_abs_error=checks,
            method="Grouped sparse Cooler reads, exact crop; no interpolation",
        ),
    )


if __name__ == "__main__":
    main()

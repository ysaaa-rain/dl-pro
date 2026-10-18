"""Measure how much 10 bp detail survives 40 bp pooling across WT repeats.

This is a descriptive diagnostic on previously observed spatial test blocks,
not a theoretical reconstruction ceiling or an independent biological study.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path("data/processed/sr_expanded")
OUT = Path("results/sr_information_limit")


def pooled(a, factor):
    n = a.shape[-1] // factor
    return a[: n * factor, : n * factor].reshape(n, factor, n, factor).mean((1, 3))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "windows.csv")
    raw = np.load(ROOT / "raw_counts.npy", mmap_mode="r")
    assert raw.shape == (len(frame), 1024, 1024)
    pairs = []
    for (start, end), group in frame[frame.split == "test"].groupby(["start", "end"]):
        assert set(group.replicate) == {1, 2} and len(group) == 2
        pairs.append(
            (int(group.index[group.replicate == 1][0]),
             int(group.index[group.replicate == 2][0]),
             int(start), int(end))
        )
    assert len(pairs) == 16

    train_hist = np.zeros(11, dtype=np.int64)
    offdiag40 = np.triu(np.ones((256, 256), dtype=bool), 5)
    zero10 = []
    for index in frame.index[frame.split == "train"]:
        high = np.asarray(raw[index])
        zero10.append(float((high == 0).mean()))
        block = high.reshape(256, 4, 256, 4).sum((1, 3))
        train_hist += np.bincount(
            np.minimum(block[offdiag40], 10).astype(int), minlength=11
        )

    pooled_values = {bp: [[], []] for bp in (10, 40, 100, 200)}
    detail = {minimum: np.zeros(4, dtype=float) for minimum in (1, 5, 10, 20)}
    per_location = []
    for i, j, start, end in pairs:
        a = np.asarray(raw[i], dtype="float32")
        b = np.asarray(raw[j], dtype="float32") * float(frame.loc[j, "depth_multiplier"])
        record = {"start": start, "end": end, "group": int(frame.loc[i, "group"])}
        for bp in pooled_values:
            factor = bp // 10
            x, y = pooled(a, factor), pooled(b, factor)
            n = len(x)
            mask = np.triu(np.ones((n, n), dtype=bool), max(1, 200 // bp))
            pooled_values[bp][0].append(np.log1p(x[mask]))
            pooled_values[bp][1].append(np.log1p(y[mask]))
            record[f"pooled_{bp}bp_pearson"] = float(
                np.corrcoef(np.log1p(x[mask]), np.log1p(y[mask]))[0, 1]
            )
        aa = a.reshape(256, 4, 256, 4).transpose(0, 2, 1, 3).reshape(256, 256, 16)
        bb = b.reshape(256, 4, 256, 4).transpose(0, 2, 1, 3).reshape(256, 256, 16)
        sa, sb = aa.sum(-1), bb.sum(-1)
        for minimum in detail:
            valid = offdiag40 & (sa >= minimum) & (sb >= minimum)
            if not valid.any():
                continue
            da = aa[valid] / sa[valid, None] - 1 / 16
            db = bb[valid] / sb[valid, None] - 1 / 16
            dot = float(np.sum(da * db))
            norm_a = float(np.sum(da * da))
            norm_b = float(np.sum(db * db))
            detail[minimum] += [int(valid.sum()), dot, norm_a, norm_b]
            record[f"detail_min{minimum}_blocks"] = int(valid.sum())
            record[f"detail_min{minimum}_correlation"] = dot / np.sqrt(norm_a * norm_b)
        per_location.append(record)

    pooled_corr = {}
    for bp, (left, right) in pooled_values.items():
        pooled_corr[bp] = float(np.corrcoef(np.concatenate(left), np.concatenate(right))[0, 1])
    detail_corr = {
        minimum: {
            "n_blocks": int(values[0]),
            "pattern_correlation": float(values[1] / np.sqrt(values[2] * values[3])),
        }
        for minimum, values in detail.items()
    }
    pd.DataFrame(per_location).to_csv(OUT / "per_location.csv", index=False)
    result = {
        "n_train_windows": len(zero10),
        "n_paired_test_locations": len(pairs),
        "n_test_spatial_groups": len(set(row["group"] for row in per_location)),
        "train_10bp_zero_fraction": float(np.mean(zero10)),
        "train_offdiagonal_40bp_block_count_histogram_0_to_9_and_10plus": train_hist.tolist(),
        "train_offdiagonal_40bp_zero_fraction": float(train_hist[0] / train_hist.sum()),
        "train_offdiagonal_nonzero_blocks_with_at_most_one_count": float(
            train_hist[1] / train_hist[1:].sum()
        ),
        "pooled_log_count_cross_repeat_pearson": pooled_corr,
        "within_40bp_block_10bp_pattern_cross_repeat": detail_corr,
        "scope": "Same-study retrospective diagnostic; only two biological repeats",
        "interpretation_limit": "Pattern correlation removes each 40 bp block total and excludes <200 bp; it is not a mathematical upper bound on all possible estimators",
    }
    (OUT / "diagnostic.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

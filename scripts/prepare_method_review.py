"""Freeze spatial groups before extracting paired native count windows."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dlpro.experiment_core import ContactReader, LENGTH, centered_indices
from dlpro.method_review import DEFAULT_CONFIG, interior_mask, load_config, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out, cache = Path(cfg["output"]) / "data", Path(cfg["cache"])
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    signature = sha256(args.config, True)
    completion = out / "preparation.json"
    old = __import__("json").loads(completion.read_text(encoding="utf-8")) if completion.exists() else None
    def preserve_or_write(frame, path, ignore=()):
        if old is not None:
            existing = pd.read_csv(path)
            columns = [key for key in existing.columns if key not in ignore]
            pd.testing.assert_frame_equal(existing[columns], frame[columns], check_dtype=False)
        else:
            frame.to_csv(path, index=False)
    block = cfg["block_bp"]
    groups = np.arange(LENGTH // block)
    perm = np.random.default_rng(cfg["seed"]).permutation(groups)
    fold_map = {int(g): i % cfg["discovery"]["folds"] for i, g in enumerate(perm)}
    preserve_or_write(pd.DataFrame(dict(group=groups, fold=[fold_map[int(g)] for g in groups])), out / "discovery_blocks.csv")
    known = pd.read_csv("data/processed/known_windows.csv")
    known["eligible"] = interior_mask(known.window_start, known.window_end, block, LENGTH)
    known["reason"] = np.where(known.eligible, "full_context_in_block", "input_crosses_fixed_block_boundary")
    known["spatial_block"] = known.center // block
    known["new_fold"] = known.spatial_block.map(fold_map).fillna(-1).astype(int)
    known["evaluation_eligible"] = known.eligible & known.new_fold.ge(0)
    known.loc[known.eligible & ~known.evaluation_eligible, "reason"] = "incomplete_terminal_block_has_no_CV_fold"
    preserve_or_write(known, out / "known_eligibility.csv", ignore=("reason",))
    scan = pd.read_csv("data/processed/scan_windows.csv")
    starts = np.array([centered_indices(int(c))[1] * 200 for c in scan.center])
    scan["eligible"] = interior_mask(starts, starts + cfg["discovery"]["max_context_bp"], block, LENGTH)
    scan["spatial_block"] = scan.center // block
    scan["new_fold"] = scan.spatial_block.map(fold_map).fillna(-1).astype(int)
    scan["evaluation_eligible"] = scan.eligible & scan.new_fold.ge(0)
    preserve_or_write(scan, out / "scan_eligibility.csv")
    old_plan = pd.read_csv("data/processed/sr_expanded/windows.csv")
    historical = old_plan[old_plan.replicate == 1]
    train_groups = np.sort(historical.loc[historical.split == "train", "group"].unique())
    order = np.random.default_rng(cfg["seed"]).permutation(train_groups)
    sr = cfg["sr"]
    nt, nv = sr["new_test_groups"], sr["new_val_groups"]
    test_groups, val_groups = set(order[:nt]), set(order[nt:nt + nv])
    patch_bp = sr["patch_bins"] * sr["native_bp"]
    rows = []
    for group in train_groups:
        split = "new_test" if group in test_groups else "val" if group in val_groups else "train"
        sparse = [int((block * fraction) // sr["native_bp"] * sr["native_bp"])
                  for fraction in sr["sample_fractions"]]
        offsets = (sorted(set(range(0, block - patch_bp + 1, sr["stride_bp"])) | set(sparse))
                   if split == "train" else sparse)
        for offset in offsets:
            start = int(group * block + offset)
            rows.append(dict(group=int(group), start=start, end=start + patch_bp, split=split,
                             sparse_training=split == "train" and offset in sparse))
    for row in historical[historical.split == "test"].itertuples():
        rows.append(dict(group=int(row.group), start=int(row.start), end=int(row.end),
                         split="historical_test", sparse_training=False))
    plan = pd.DataFrame(rows).sort_values(["group", "start"]).reset_index(drop=True)
    plan["sample_index"] = np.arange(len(plan))
    assert interior_mask(plan.start, plan.end, block, LENGTH).all()
    assert plan.groupby("group").split.nunique().eq(1).all()
    assert set(plan.loc[plan.split == "train", "group"]).isdisjoint(test_groups | val_groups)
    assert not set(plan.loc[plan.split == "train", "group"]) & set(historical.loc[historical.split != "train", "group"])
    preserve_or_write(plan, out / "sr_windows.csv")
    if old is not None and all((cache / name).exists() for name in old["cache_sha256"]):
        for name, expected in old["cache_sha256"].items():
            assert sha256(cache / name) == expected
        print("Verified geometry-compatible frozen plans and complete caches; no artifacts overwritten", flush=True)
        return
    samples = pd.read_csv("data/annotations/samples.csv").query('condition == "37C"').sort_values("replicate")
    sources, libraries, cache_hashes = {}, {}, {}
    for source in samples.itertuples():
        repeat = int(source.replicate)
        actual = sha256(source.path)
        assert actual == source.sha256
        sources[str(repeat)] = actual
        reader = ContactReader(source.path)
        libraries[str(repeat)] = reader.library_size
        path = cache / f"raw_rep{repeat}.npy"
        if path.exists():
            if old is None:
                raise FileExistsError(f"Incomplete extraction retained, use a new cache directory: {path}")
            actual = sha256(path)
            assert actual == old["cache_sha256"][path.name]
            cache_hashes[path.name] = actual
            continue
        array = np.lib.format.open_memmap(path, mode="w+", dtype="uint32",
                                          shape=(len(plan), sr["patch_bins"], sr["patch_bins"]))
        for group, part in plan.groupby("group"):
            start = int(group * block // sr["native_bp"])
            end = int((group + 1) * block // sr["native_bp"])
            native = reader.matrix[start:end, start:end].tocsr()
            for index, row in part.iterrows():
                offset = int((row.start - group * block) // sr["native_bp"])
                image = native[offset:offset + sr["patch_bins"], offset:offset + sr["patch_bins"]].toarray()
                assert image.shape == (sr["patch_bins"], sr["patch_bins"])
                assert np.array_equal(image, image.T) and (image >= 0).all()
                assert np.array_equal(image, np.rint(image))
                array[index] = image
            print("extracted", repeat, "group", int(group), "windows", len(part), flush=True)
        array.flush()
        del array
        cache_hashes[path.name] = sha256(path)
        if old is not None:
            assert cache_hashes[path.name] == old["cache_sha256"][path.name]
    if old is not None:
        assert cache_hashes == old["cache_sha256"]
        print("Missing ignored caches restored byte-for-byte; frozen manifests preserved", flush=True)
        return
    write_json(completion, dict(config_sha256=signature, source_sha256=sources, library_sizes=libraries,
                               cache_sha256=cache_hashes, samples=len(plan),
                               counts_by_split=plan.split.value_counts().to_dict(),
                               groups_by_split=plan.groupby("split").group.nunique().to_dict(),
                               sparse_training_samples=int(plan.sparse_training.sum()),
                               known_eligibility=known.groupby(["type", "eligible"]).size().to_dict().__str__(),
                               test_status="New reservation within historically exposed genome; no external validation"))


if __name__ == "__main__":
    main()

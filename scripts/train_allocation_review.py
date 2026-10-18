"""Fixed sample-size and conditional-objective contrasts; no test access."""

import argparse
import copy
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import torch

from dlpro.experiment_core import configure
from dlpro.method_review import DEFAULT_CONFIG, load_config, sha256, write_json
from dlpro.sr_review import exact_count_log_quantile, load_data, make_batch, make_model, objective
from dlpro.structure_losses import fit_expectations


def validation(model, arm, raw, depth, ids, scale, expectations, sr):
    values = []
    model.eval()
    with torch.no_grad():
        for r in (0, 1):
            for index in ids:
                high, other = make_batch(raw, depth, [index], [r])
                values.append(float(objective(model, arm, high, other, scale, expectations, sr)))
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    cfg = load_config(args.config)
    sr = cfg["sr"]
    out = Path(cfg["output"]) / sr.get("result_subdir", "sr")
    out.mkdir(parents=True, exist_ok=True)
    plan, raw, depth = load_data(cfg)
    train = np.flatnonzero(plan.split == "train")
    val = np.flatnonzero(plan.split == "val")
    sparse = np.flatnonzero(plan.sparse_training)
    torch.set_num_threads(cfg["threads"])
    assert sr["factor"] == 4, "Legacy model protocol uses native10bp to40bp"
    if sr.get("normalization_method") != "exact_count_histogram":
        raise ValueError("Biased grid normalization was retired; use the corrected overlay config")
    scale = exact_count_log_quantile(raw, train, depth, sr["normalization_quantile"])
    print("exact training pixel normalization", scale, flush=True)
    assert scale > 0
    expectations_all = [fit_expectations(a, train) for a in raw]
    expectations = {bp: sum(e[bp] * m for e, m in zip(expectations_all, depth)) / len(raw)
                    for bp in sr["morphology_scales_bp"]}
    profile = np.zeros(sr["patch_bins"])
    for a, multiplier in zip(raw, depth):
        for index in train:
            image = a[index].astype("float64") * multiplier
            profile += np.array([np.diagonal(image, offset=d).mean() for d in range(len(profile))])
    profile /= len(train) * len(raw)
    profile = np.exp(gaussian_filter1d(np.log(np.maximum(profile, 1e-8)), sr["prior_smoothing_sigma"]))
    np.savez_compressed(out / "normalization.npz", scale=scale, profile=profile, training_indices=train,
                        **{f"expected_{bp}": value for bp, value in expectations.items()})
    probe = []
    for arm in sr["arms"]:
        configure(sr["train_seeds"][0])
        torch.set_num_threads(cfg["threads"])
        model = make_model(arm, scale, profile, sr)
        optimizer = torch.optim.Adam(model.parameters(), lr=sr["learning_rate"])
        high, other = make_batch(raw, depth, train[:sr["batch_size"]], np.zeros(sr["batch_size"], int))
        begin = time.monotonic()
        for _ in range(sr["resource_probe_steps"]):
            optimizer.zero_grad()
            loss = objective(model, arm, high, other, scale, expectations, sr)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        seconds = (time.monotonic() - begin) / sr["resource_probe_steps"]
        probe.append(dict(arm=arm, seconds_per_step=seconds,
                          estimated_training_seconds=seconds * sr["steps"] * len(sr["train_seeds"]),
                          parameters=sum(p.numel() for p in model.parameters())))
        print("resource probe", probe[-1], flush=True)
    write_json(out / "resource_probe.json", probe)
    config_hash = sha256(args.config, True)
    all_history = []
    for seed in sr["train_seeds"]:
        for arm in sr["arms"]:
            selected_path = out / f"{arm}_seed{seed}.pt"
            if selected_path.exists():
                raise FileExistsError(f"Do not overwrite completed model: {selected_path}")
            configure(seed)
            torch.set_num_threads(cfg["threads"])
            rng = np.random.default_rng(seed)
            model = make_model(arm, scale, profile, sr)
            optimizer = torch.optim.Adam(model.parameters(), lr=sr["learning_rate"])
            pool = sparse if "sparse" in arm else train
            best, best_state, history = float("inf"), None, []
            started = time.monotonic()
            losses = []
            for step in range(1, sr["steps"] + 1):
                model.train()
                ids = rng.choice(pool, sr["batch_size"], replace=False)
                repeats = rng.integers(0, 2, len(ids))
                high, other = make_batch(raw, depth, ids, repeats, rng.random(len(ids)) < .5)
                lr = sr["min_learning_rate"] + (sr["learning_rate"] - sr["min_learning_rate"]) * .5 * (1 + math.cos(math.pi * (step - 1) / sr["steps"]))
                optimizer.param_groups[0]["lr"] = lr
                optimizer.zero_grad()
                loss = objective(model, arm, high, other, scale, expectations, sr)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite objective {arm} seed{seed} step{step}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), sr["gradient_clip"])
                optimizer.step()
                losses.append(float(loss.detach()))
                if step % sr["validate_every"] == 0:
                    value = validation(model, arm, raw, depth, val, scale, expectations, sr)
                    row = dict(arm=arm, seed=seed, step=step, train_loss=float(np.mean(losses)),
                               validation_objective=value, elapsed_seconds=time.monotonic() - started)
                    history.append(row)
                    print("validation", row, flush=True)
                    if value < best:
                        best, selected_step, best_state = value, step, copy.deepcopy(model.state_dict())
                    pd.DataFrame(history).to_csv(out / f"{arm}_seed{seed}_history.csv", index=False)
                    losses = []
            torch.save(dict(state_dict=best_state, scale=scale, profile=torch.from_numpy(profile.astype("float32")),
                            selected_step=selected_step, validation_objective=best, seed=seed, arm=arm,
                            config_sha256=config_hash, training_indices=pool.tolist()), selected_path)
            all_history.extend(history)
            print("completed", arm, seed, "selected", selected_step, flush=True)
    pd.DataFrame(all_history).to_csv(out / "training_history.csv", index=False)
    write_json(out / "training_status.json", dict(completed_models=len(sr["arms"]) * len(sr["train_seeds"]),
                                                  scale=scale, config_sha256=config_hash,
                                                  train_windows=len(train), sparse_windows=len(sparse),
                                                  validation_windows=len(val), test_read_during_training=False))


if __name__ == "__main__":
    main()

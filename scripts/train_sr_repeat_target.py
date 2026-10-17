"""Fixed one-seed cross-repeat SR training on the original 64 train loci.

Repeat 1 supplies 40 bp totals. A block-normalized repeat 2 image supplies
the 10 bp allocation target. Validation uses the original 12 held-out loci.
Every 80-step state is resumable; no test data enters training or selection.
"""

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from dlpro.experiment_core import configure, save_json
from dlpro.refinement import ContextCountSR, project_counts
from dlpro.structure_losses import structure_terms


OUT = Path("results/sr_repeat_target")
STEPS = 960


def make_batch(source, repeat2, indices, source_rows, scale, device):
    high1 = torch.from_numpy(np.asarray(source[indices], dtype="float32").copy())[:, None].to(device)
    multiplier = torch.as_tensor(source_rows.depth_multiplier.to_numpy()[indices],
                                  dtype=torch.float32, device=device)[:, None, None, None]
    high2 = torch.from_numpy(np.asarray(repeat2[indices], dtype="float32").copy())[:, None].to(device)
    low = F.avg_pool2d(high1, 4)
    target = project_counts(high2 * multiplier, low)
    return torch.log1p(low) / scale, target, low


def objective(model, low, target, expectations, scale):
    prediction_log = model(low)
    prediction = torch.expm1(prediction_log * scale)
    native = F.mse_loss(prediction_log.clamp(0, 1),
                        (torch.log1p(target) / scale).clamp(0, 1))
    terms = structure_terms(prediction, target, expectations)
    mse = torch.stack([entry[0] for entry in terms.values()]).mean()
    ssim = torch.stack([entry[1] for entry in terms.values()]).mean()
    return native + 0.1 * mse + 0.05 * ssim, (native, mse, ssim)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configure(42)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = pd.read_csv("data/processed/sr_repeat_target/windows.csv")
    source_plan = pd.read_csv("data/processed/sr_expanded/windows.csv")
    assert len(rows) == 76 and rows.split.value_counts().to_dict() == {"train": 64, "val": 12}
    source_rows = source_plan.iloc[rows.original_index.to_numpy()].reset_index(drop=True)
    assert np.array_equal(source_rows.start, rows.start)
    source_raw = np.load("data/processed/sr_expanded/raw_counts.npy", mmap_mode="r")
    source = np.asarray(source_raw[rows.original_index.to_numpy()], dtype="float32")
    repeat2 = np.load("data/processed/sr_repeat_target/raw_counts.npy", mmap_mode="r")
    assert source.shape == repeat2.shape == (76, 1024, 1024)
    with np.load("results/sr_data_controls/expectations.npz") as saved:
        expectations = {int(key): saved[key] for key in saved.files}
    assert set(expectations) == {50, 100, 160}
    baseline = torch.load("results/sr_data_controls/dense_multiscale_seed42.pt", weights_only=True)
    scale = float(baseline["scale"])
    model = ContextCountSR(scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    rng = np.random.default_rng(42)
    training = np.flatnonzero(rows.split == "train")
    validation = np.flatnonzero(rows.split == "val")
    history = []
    best = float("inf")
    selected = None
    start_step = 1
    latest = OUT / "resume.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.bit_generator.state = state["rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        history = state["history"]
        best = state["best"]
        selected = state["selected"]
        start_step = state["step"] + 1
        print("resume", start_step, flush=True)
    for step in range(start_step, STEPS + 1):
        model.train()
        ids = rng.choice(training, 4, replace=False)
        low, target, measured = make_batch(source, repeat2, ids, source_rows, scale, device)
        reverse = rng.random(4) < 0.5
        low[reverse] = low[reverse].flip((-2, -1))
        target[reverse] = target[reverse].flip((-2, -1))
        optimizer.param_groups[0]["lr"] = 0.0001 + 0.0009 * 0.5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
        optimizer.zero_grad()
        loss, _ = objective(model, low, target, expectations, scale)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % 80 == 0:
            model.eval()
            scores = []
            with torch.no_grad():
                for ids_val in np.array_split(validation, 3):
                    x, y, _ = make_batch(source, repeat2, ids_val, source_rows, scale, device)
                    total, parts = objective(model, x, y, expectations, scale)
                    scores.append([float(total), *[float(x) for x in parts]])
            mean = np.mean(scores, axis=0)
            history.append(dict(step=step, val_objective=float(mean[0]),
                                val_native_mse=float(mean[1]),
                                val_oe_mse=float(mean[2]), val_ssim_loss=float(mean[3])))
            if mean[0] < best:
                best = float(mean[0])
                selected = step
                torch.save(dict(state_dict={key: value.detach().cpu().clone()
                                            for key, value in model.state_dict().items()},
                                scale=scale, seed=42, selected_step=step,
                                validation_objective=best, recipe="cross_repeat_target"),
                           OUT / "model_seed42.pt")
            pd.DataFrame(history).to_csv(OUT / "validation_history.csv", index=False)
            torch.save(dict(step=step, model=copy.deepcopy(model.state_dict()),
                            optimizer=optimizer.state_dict(), rng=rng.bit_generator.state,
                            torch_rng=torch.get_rng_state(), history=history,
                            best=best, selected=selected), latest)
            print("repeat-target", step, "val", float(mean[0]),
                  "best", best, flush=True)
    save_json(OUT / "training_protocol.json", dict(
        seed=42, steps=STEPS, batch_size=4, optimizer="Adam, cosine 0.001 to 0.0001",
        training_loci=64, validation_loci=12,
        source="WT repeat1 observed 40 bp means, original split",
        target="WT repeat2 native 10 bp allocation normalized to repeat1 measured 40 bp block totals",
        target_zero_fallback="Uniform within a 40 bp block when repeat2 has zero counts",
        architecture="Unchanged ContextCountSR",
        scale=scale, expectations="Frozen original64 repeat1 training windows",
        objective="Clipped native log MSE + .1 mean 50/100/160bp offdiagonal O/E MSE + .05 mean masked O/E SSIM loss",
        checkpoint_selection="Lowest cross-repeat target validation objective every80 steps",
        selected_step=selected, validation_objective=best,
        test_used_for_selection=False, retrospective=True,
        device=str(device),
    ))


if __name__ == "__main__":
    main()

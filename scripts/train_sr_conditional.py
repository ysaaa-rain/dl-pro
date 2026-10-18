"""Two fixed matched SR arms using observable target-count allocation NLL."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from dlpro.count_allocation import conditional_allocation_nll
from dlpro.experiment_core import configure, save_json
from dlpro.refinement import ContextCountSR


OUT = Path("results/sr_conditional")
STEPS = 960


def batch(source, repeat2, ids, target_repeat, scale, device):
    high1 = torch.from_numpy(np.asarray(source[ids], dtype="float32").copy())[:, None].to(device)
    target = high1 if target_repeat == 1 else torch.from_numpy(
        np.asarray(repeat2[ids], dtype="float32").copy()
    )[:, None].to(device)
    low = F.avg_pool2d(high1, 4)
    return torch.log1p(low) / scale, target, low


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = pd.read_csv("data/processed/sr_repeat_target/windows.csv")
    plan = pd.read_csv("data/processed/sr_expanded/windows.csv")
    raw = np.load("data/processed/sr_expanded/raw_counts.npy", mmap_mode="r")
    source = np.asarray(raw[rows.original_index.to_numpy()], dtype="float32")
    repeat2 = np.load("data/processed/sr_repeat_target/raw_counts.npy", mmap_mode="r")
    assert source.shape == repeat2.shape == (76, 1024, 1024)
    assert np.array_equal(plan.iloc[rows.original_index.to_numpy()].start, rows.start)
    training = np.flatnonzero(rows.split == "train")
    validation = np.flatnonzero(rows.split == "val")
    assert len(training) == 64 and len(validation) == 12
    scale = float(torch.load("results/sr_data_controls/dense_multiscale_seed42.pt", weights_only=True)["scale"])
    summaries = []
    for target_repeat in (1, 2):
        recipe = f"conditional_rep{target_repeat}"
        history_path = OUT / f"{recipe}_history.csv"
        selected_path = OUT / f"{recipe}_seed42.pt"
        latest = OUT / f"{recipe}_resume.pt"
        configure(42)
        torch.set_num_threads(8)
        rng = np.random.default_rng(42)
        model = ContextCountSR(scale).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        history, best, selected, start = [], float("inf"), None, 1
        if latest.exists():
            saved = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            rng.bit_generator.state = saved["rng"]
            torch.set_rng_state(saved["torch_rng"].cpu())
            history, best, selected = saved["history"], saved["best"], saved["selected"]
            start = saved["step"] + 1
        for step in range(start, STEPS + 1):
            model.train()
            ids = rng.choice(training, 4, replace=False)
            x, target, low = batch(source, repeat2, ids, target_repeat, scale, device)
            reverse = rng.random(4) < .5
            x[reverse] = x[reverse].flip((-2, -1))
            target[reverse] = target[reverse].flip((-2, -1))
            low[reverse] = low[reverse].flip((-2, -1))
            optimizer.param_groups[0]["lr"] = .0001 + .0009 * .5 * (1 + math.cos(math.pi * (step - 1) / STEPS))
            optimizer.zero_grad()
            counts = torch.expm1(model(x) * scale)
            loss = conditional_allocation_nll(counts, target, low)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite {recipe} training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if step % 80 == 0:
                model.eval()
                values = []
                with torch.no_grad():
                    for ids_val in np.array_split(validation, 3):
                        vx, vy, vl = batch(source, repeat2, ids_val, target_repeat, scale, device)
                        prediction = torch.expm1(model(vx) * scale)
                        values.append(float(conditional_allocation_nll(prediction, vy, vl)))
                score = float(np.mean(values))
                history.append(dict(step=step, validation_nll=score))
                if score < best:
                    best, selected = score, step
                    torch.save(dict(state_dict={key: value.detach().cpu().clone()
                                                for key, value in model.state_dict().items()},
                                    scale=scale, seed=42, recipe=recipe, selected_step=step,
                                    validation_objective=score), selected_path)
                pd.DataFrame(history).to_csv(history_path, index=False)
                torch.save(dict(step=step, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                                history=history, best=best, selected=selected), latest)
                print(recipe, step, "val NLL", score, "best", best, flush=True)
        summaries.append(dict(recipe=recipe, target_repeat=target_repeat, selected_step=selected,
                              validation_nll=best))
    pd.DataFrame(summaries).to_csv(OUT / "training_summary.csv", index=False)
    save_json(OUT / "training_protocol.json", dict(
        steps=STEPS, seed=42, batch_size=4, train_loci=64, val_loci=12,
        recipes=["conditional_rep1", "conditional_rep2"], source_repeat=1,
        architecture="Unchanged ContextCountSR", source_bp=40, target_bp=10,
        objective="Equal near<200bp/far count-normalized conditional composite NLL; source-zero or target-zero blocks excluded",
        optimizer="Adam cosine0.001 to0.0001; genome reversal; clip gradient norm1",
        selection="Each arm lowest matching-target validation NLL at fixed80-step intervals",
        test_used_for_selection=False, retrospective=True, device=str(device),
        limitations=["Symmetric pixels are not independent counts; composite likelihood only",
                     "Only two biological repeats; previously exposed spatial test positions",
                     "Both loss and target differ from prior multiscale-log-MSE recipe"],
    ))


if __name__ == "__main__":
    main()

"""Apply a frozen review branch to an actual 40bp block-mean count image.

Outputs are expected10bp image counts, not measured integer contacts. The input
must use the training library's count units for neural branches. The normalized
distance-prior allocation is invariant to a global input depth multiplier.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dlpro.method_review import load_config, sha256
from dlpro.sr_review import infer_model, make_model
from evaluate_allocation_review import simple_predictions, COUNT_RELATIVE_TOLERANCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/method_review_v1_corrected.json")
    parser.add_argument("--input-low", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="validation_selected")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    sr = cfg["sr"]
    root = Path(cfg["output"]) / sr.get("result_subdir", "sr")
    if args.output.exists():
        raise FileExistsError(f"Preserve existing prediction: {args.output}")
    method = args.method
    if method == "validation_selected":
        method = json.loads((root / "evaluation/validation_selection.json").read_text(encoding="utf-8"))["selected_method"]
    elif method == "primary":
        method = sr["primary_arm"]
    torch.set_num_threads(cfg["threads"])
    array = np.load(args.input_low, allow_pickle=False)
    if array.ndim != 2 or array.shape[0] != array.shape[1] or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Expected a finite nonnegative square40bp-mean image")
    if not np.allclose(array, array.T, rtol=COUNT_RELATIVE_TOLERANCE, atol=0):
        raise ValueError("Input contact means must be symmetric")
    parameters = np.load(root / "normalization.npz")
    profile, scale = parameters["profile"], float(parameters["scale"])
    if len(array) * sr["factor"] > len(profile):
        raise ValueError("Input exceeds the learned distance-profile extent; use the frozen patch geometry")
    lock = json.loads((root / "evaluation/test_lock.json").read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else sr["train_seeds"][0]
    reference_arm = method if method in sr["arms"] else sr["arms"][0]
    checkpoint_path = root / f"{reference_arm}_seed{seed}.pt"
    actual_checkpoint_hash = sha256(checkpoint_path)
    assert actual_checkpoint_hash == lock["checkpoint_sha256"][checkpoint_path.stem]
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert np.array_equal(checkpoint["profile"].numpy(), profile.astype("float32"))
    assert checkpoint["scale"] == scale
    low = torch.from_numpy(array.astype("float32"))[None, None]
    with torch.no_grad():
        if method in sr["arms"]:
            model = make_model(method, scale, profile, sr).eval()
            model.load_state_dict(checkpoint["state_dict"])
            counts = infer_model(model, method, low, scale, sr)[0]
        else:
            predictions = simple_predictions(low, profile, sr["factor"])
            if method not in predictions or method.endswith("unprojected"):
                raise ValueError(f"Unknown or unconstrained reconstruction branch: {method}")
            counts = predictions[method]
    error = float(((F.avg_pool2d(counts, sr["factor"])-low).abs()/low.clamp_min(1e-8)).max())
    symmetry = float((counts-counts.transpose(-1,-2)).abs().max()/counts.max().clamp_min(1e-8))
    assert torch.isfinite(counts).all() and counts.min() >= 0
    assert error < COUNT_RELATIVE_TOLERANCE and symmetry < COUNT_RELATIVE_TOLERANCE
    metadata = dict(method=method, seed=seed if method in sr["arms"] else None,
                    input_sha256=sha256(args.input_low), frozen_reference_checkpoint_sha256=actual_checkpoint_hash,
                    normalization_sha256=sha256(root / "normalization.npz"),
                    block_mean_max_relative_error=error, symmetry_max_relative_error=symmetry,
                    input_unit="40bp block mean", output_unit="10bp expected image counts in supplied input mass units",
                    scope="Experimental branch; existing operational choices unchanged")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, counts=counts.numpy()[0, 0], low_mean=array,
                        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

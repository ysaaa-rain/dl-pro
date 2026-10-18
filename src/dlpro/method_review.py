"""Shared configuration, provenance and spatially grouped inference utilities."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd


DEFAULT_CONFIG = Path("configs/method_review_v1.json")


def load_config(path=DEFAULT_CONFIG):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    parent = value.pop("extends", None)
    if parent is None:
        return value
    def merge(left, right):
        for key, item in right.items():
            if isinstance(item, dict) and isinstance(left.get(key), dict):
                merge(left[key], item)
            else:
                left[key] = item
    base = {}
    for item in (parent if isinstance(parent, list) else [parent]):
        merge(base, load_config(path.parent / item))
    merge(base, value)
    return base


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8", newline="\n")


def sha256(path, canonical=False):
    if canonical:
        return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(config):
    paths = [*Path("src/dlpro").glob("*.py"), *Path("scripts").glob("*.py"),
             *Path("tests").glob("*.py"), *Path("configs").glob("*.json"), Path("docs/第五轮根本瓶颈与固定协议.md")]
    return dict(config=config, git_revision=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
        code_sha256={p.as_posix(): sha256(p, True) for p in paths if p.exists()},
        data_manifest_sha256={p.as_posix(): sha256(p) for p in
                              (Path(config["output"]) / "data").glob("*") if p.is_file()},
        device="cpu", python=sys.version, platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in
                  ("numpy", "pandas", "torch", "scipy", "scikit-learn", "scikit-image", "cooler", "pytest")})


def logged_run(config, stage, command):
    directory = Path(config["output"]) / "runs"
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / f"{stage}.json"
    if record.exists():
        raise FileExistsError(f"Preserve existing run record: {record}")
    started = datetime.now(timezone.utc).isoformat()
    before = time.monotonic()
    payload = dict(provenance(config), command=command, started_utc=started,
                   status="running", exit_code=None)
    archive_path = directory / f"{stage}_source.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, expected in payload["code_sha256"].items():
            source = Path(name).read_bytes().replace(b"\r\n", b"\n")
            assert hashlib.sha256(source).hexdigest() == expected
            archive.writestr(name, source)
    payload["source_archive"] = archive_path.as_posix()
    payload["source_archive_sha256"] = sha256(archive_path)
    write_json(record, payload)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    environment["OMP_NUM_THREADS"] = str(config["threads"])
    exit_code = -1
    try:
        with (directory / f"{stage}.log").open("w", encoding="utf-8") as output:
            result = subprocess.run([sys.executable, *command], env=environment,
                                    stdout=output, stderr=subprocess.STDOUT)
        exit_code = result.returncode
    finally:
        payload.update(ended_utc=datetime.now(timezone.utc).isoformat(),
                       elapsed_seconds=time.monotonic() - before,
                       exit_code=exit_code, status="completed" if exit_code == 0 else "failed")
        write_json(record, payload)
    if exit_code:
        raise RuntimeError(f"Stage {stage} failed with exit code {exit_code}; see its log")


def nms(score, centers, budget, spacing, length):
    keep = []
    for index in np.argsort(-np.asarray(score), kind="stable"):
        distance = np.abs(centers[keep] - centers[index])
        if np.all(np.minimum(distance, length - distance) >= spacing):
            keep.append(int(index))
        if len(keep) >= budget:
            break
    return np.asarray(keep, dtype=int)


def grouped_interval(values, groups, seed, draws):
    values, groups = np.asarray(values, float), np.asarray(groups)
    unique = np.unique(groups)
    if not len(values):
        return dict(mean=None, low=None, high=None, groups=0, n=0)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        draw = rng.choice(unique, len(unique), replace=True)
        samples.append(np.concatenate([values[groups == g] for g in draw]).mean())
    lo, hi = np.quantile(samples, [.025, .975])
    return dict(mean=float(values.mean()), low=float(lo), high=float(hi),
                groups=len(unique), n=len(values))


def interior_mask(start, end, block_bp, length):
    start, end = np.asarray(start), np.asarray(end)
    return ((start >= 0) & (end <= length)
            & (start // block_bp == (end - 1) // block_bp))


def spatial_cv_masks(eligible, folds, test_fold, n_folds):
    folds = np.asarray(folds)
    assigned = np.asarray(eligible, bool) & (folds >= 0) & (folds < n_folds)
    test = assigned & (folds == test_fold)
    validation = assigned & (folds == (test_fold + 1) % n_folds)
    train = assigned & ~(test | validation)
    return train, validation, test


def restore_git_asset(path, revision):
    """Materialize an existing tracked artifact only after its Git blob identity matches."""
    import requests
    path = Path(path)
    oid = subprocess.check_output(["git", "rev-parse", f"{revision}:{path.as_posix()}"], text=True).strip()
    if path.exists():
        data = path.read_bytes()
    else:
        response = requests.get(f"https://raw.githubusercontent.com/ysaaa-rain/dl-pro/{revision}/{path.as_posix()}", timeout=90)
        response.raise_for_status()
        data = response.content
    actual = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if actual != oid:
        raise ValueError(f"Git blob mismatch: {path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return dict(path=path.as_posix(), git_blob=oid, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

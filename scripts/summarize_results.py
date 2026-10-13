"""Refresh report tables from saved first-round measurements; never train models."""
import json
from pathlib import Path
import pandas as pd
root=Path(__file__).resolve().parents[1]
def read(path):return json.loads((root/path).read_text())
def rows(path):return pd.read_csv(root/path).to_dict("records")
s=read("results/experiment/summary.json")
s.update(classification_cv=read("results/classification_cv/summary.json"),per_class=read("results/classification_cv/classification_report.json"),classification_baselines=rows("results/classification/baselines.csv"),classification_seeds=rows("results/classification/metrics.csv"),discovery=read("results/discovery/protocol.json"),known_recall=rows("results/discovery/known_recall.csv"),representation=read("results/discovery/representation_check.json"),multitrack=read("results/multitrack/protocol.json"),sr_downstream=rows("results/sr_expanded/downstream_summary.csv"),sr_protocol=read("results/sr_expanded/protocol.json"),occlusion=rows("results/classification/occlusion.csv"),coverage_qc=read("results/coverage_qc/summary.json"),validation=read("results/experiment/validation.json"))
s["sr_native"]=pd.read_csv(root/"results/sr_expanded/metrics_summary.csv").groupby(["replicate","method"],as_index=False)[["psnr","ssim","shape_ssim_160bp"]].mean().to_dict("records")
s["unannotated_reproducible"]=len(pd.read_csv(root/"results/discovery/unannotated_reproducible.csv"))
(root/"results/experiment/summary.json").write_text(json.dumps(s,ensure_ascii=False,indent=2)+"\n")
print("Updated first-round measured tables; narrative remains a reviewed first-round snapshot.")

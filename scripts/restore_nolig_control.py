"""Restore the single registered NoLig sample for descriptive candidate QC."""
import json
from pathlib import Path
import pandas as pd
import requests
from restore_wt_sources import fetch_one


if __name__ == "__main__":
    frame = pd.read_csv("data/annotations/samples.csv")
    selected = frame[frame.condition == "NoLig"]
    assert len(selected) == 1
    row = next(selected.itertuples())
    compressed = Path(row.gzip_path)
    if compressed.exists():
        length = compressed.stat().st_size
    else:
        for attempt in range(5):
            try:
                response = requests.get(row.source_url, headers={"Range": "bytes=0-0"}, timeout=(20, 90))
                response.raise_for_status()
                assert response.status_code == 206
                length = int(response.headers["Content-Range"].split("/")[-1])
                break
            except Exception:
                if attempt == 4:
                    raise
    verification = fetch_one(row, length)
    out = Path("results/nolig_source_restore/verification.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")

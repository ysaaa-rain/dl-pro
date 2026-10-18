"""Restore missing registered perturbation matrices; never accept partial inputs."""
import argparse
from pathlib import Path
import pandas as pd
import requests
from dlpro.method_review import load_config,sha256,write_json
from restore_wt_sources import fetch_one

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config)
    samples=pd.read_csv('data/annotations/samples.csv');rows=[]
    for row in samples[samples.condition.isin(cfg['diagnostic']['condition_names'])].itertuples():
        try:
            if Path(row.path).exists():
                digest=sha256(row.path);assert digest==row.sha256
                rows.append(dict(sample_id=row.sample_id,status='verified_existing',cooler_sha256=digest));continue
            compressed=Path(row.gzip_path)
            if compressed.exists():length=compressed.stat().st_size
            else:
                response=requests.get(row.source_url,headers={'Range':'bytes=0-0'},timeout=(20,45));response.raise_for_status()
                assert response.status_code==206;length=int(response.headers['Content-Range'].split('/')[-1])
            record=fetch_one(row,length);rows.append(dict(status='restored_verified',**record))
        except Exception as error:
            rows.append(dict(sample_id=row.sample_id,status='unavailable',error=repr(error)))
            print('Unavailable complete matrix',row.sample_id,repr(error),flush=True)
        write_json(Path(cfg['output'])/'condition_source_restore.json',rows)
    write_json(Path(cfg['output'])/'condition_source_restore.json',rows)

if __name__=='__main__':main()

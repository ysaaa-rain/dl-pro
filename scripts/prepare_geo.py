"""Preserve downloaded gzip originals, unpack and audit all Cooler matrices."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import cooler


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def audit(path, compressed=None, url=None):
    c=cooler.Cooler(str(path))
    if c.binsize!=10 or c.chromsizes.to_dict()!={'NC_000913.3':4641652}:
        raise ValueError(f'Unexpected reference or resolution: {path}')
    bins=c.bins()[:16]
    if not (bins.start.to_numpy()==np.arange(16)*10).all():
        raise ValueError(f'Invalid bin geometry: {path}')
    matrix=c.matrix(balance=False).fetch(('NC_000913.3',50000,50160))
    if not np.isfinite(matrix).all() or (matrix<0).any() or not np.array_equal(matrix,matrix.T):
        raise ValueError(f'Invalid count window: {path}')
    stem=path.name.split('.',1)[0]
    accession,sample=stem.split('_',1)
    if '_rep' in sample:
        condition,repeat=sample.rsplit('_rep',1)
    else:
        condition,repeat=sample,None
    return dict(sample_id=stem, accession=accession, condition=condition, replicate=repeat,
        path=str(path), bytes=path.stat().st_size, sha256=sha256(path),
        source_url=url, gzip_path=str(compressed) if compressed else None,
        gzip_sha256=sha256(compressed) if compressed else None,
        binsize=int(c.binsize),chrom='NC_000913.3',chromosome_length=4641652,
        nbins=int(c.info['nbins']),nnz=int(c.info['nnz']),
        library_pairs=int(c.info['sum']),has_weights='weight' in c.bins().columns,storage_mode=c.storage_mode,
        window_read_validated=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--available',action='store_true',help='Prepare available files while downloads continue')
    args=p.parse_args()
    manifest=json.loads(Path('results/geo_download_manifest.json').read_text())
    records=[];missing=[]
    for item in manifest['files']:
        source=Path('data/raw')/item['name']
        if not source.exists():
            missing.append(item['name']);continue
        if source.stat().st_size!=item['content_length']: raise ValueError(f'Incomplete source {source}')
        dest=source.with_suffix('')
        if not dest.exists():
            tmp=Path(str(dest)+'.unpacking')
            with gzip.open(source,'rb') as src,tmp.open('wb') as dst:
                for chunk in iter(lambda:src.read(8*1024**2),b''): dst.write(chunk)
            # Validate HDF5/Cooler before publishing the extraction.
            cooler.Cooler(str(tmp))
            os.replace(tmp,dest)
        record=audit(dest,source,item['url']);records.append(record)
        print(f"Audited {dest.name} ({record['bytes']} bytes)",flush=True)
    for source in sorted(Path('data/raw').glob('GSE272159_37C_rep*.cool.gz')):
        dest=source.with_suffix('')
        if not dest.exists():
            tmp=Path(str(dest)+'.unpacking')
            with gzip.open(source,'rb') as src,tmp.open('wb') as dst:
                for chunk in iter(lambda:src.read(8*1024**2),b''):dst.write(chunk)
            cooler.Cooler(str(tmp));os.replace(tmp,dest)
    for path in sorted(Path('data/raw').glob('GSE272159_37C_rep*.cool')):
        url='https://ftp.ncbi.nlm.nih.gov/geo/series/GSE272nnn/GSE272159/suppl/'+path.name+'.gz'
        records.append(audit(path,url=url))
    complete=not missing and len(records)==29
    Path('results/geo_matrix_audit.json').write_text(json.dumps(dict(complete=complete,
        n_matrices=len(records),expected_matrices=29,missing=missing,matrices=records),indent=2)+'\n')
    pd.DataFrame(records).to_csv('data/annotations/samples.csv',index=False)
    if not complete and not args.available: raise SystemExit(f'Incomplete collection: {missing}')
    print(json.dumps(dict(complete=complete,n_matrices=len(records),missing=missing)),flush=True)

if __name__=='__main__':main()

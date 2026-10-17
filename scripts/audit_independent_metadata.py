"""Small official metadata and annotation checks; never download contact matrices."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import cooler
from dlpro.method_review import load_config,sha256,write_json

def limited_get(url,maximum,timeout):
    response=requests.get(url,stream=True,timeout=(10,timeout));response.raise_for_status();payload=bytearray()
    for block in response.iter_content(65536):
        payload.extend(block)
        if len(payload)>maximum:raise ValueError('Metadata download exceeds authorized small-file cap')
    return bytes(payload)

def parse_soft(text,series):
    samples=[];current=None
    for line in text.splitlines():
        if line.startswith('^SAMPLE = '):
            if current is not None:samples.append(current)
            current=dict(series=series,accession=line.split(' = ',1)[1])
        elif current is not None and line.startswith('!Sample_') and ' = ' in line:
            key,value=line[1:].split(' = ',1);current.setdefault(key,[]).append(value)
    if current is not None:samples.append(current)
    return samples

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);ev=cfg['evidence'];root=Path(cfg['output']);out=root/'independent_data';out.mkdir(exist_ok=True);assert not (out/'status.json').exists()
    assert (root/'course_audit/status.json').exists()
    def fetch(series):
        prefix=series[:-3]+'nnn';url=f'https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{series}/soft/{series}_family.soft.gz'
        try:
            payload=limited_get(url,ev['metadata_max_bytes'],ev['network_timeout_seconds']);text=gzip.decompress(payload).decode('utf-8')
            (out/f'{series}.soft.gz').write_bytes(payload)
            return dict(series=series,url=url,status='fetched',sha256=hashlib.sha256(payload).hexdigest(),compressed_bytes=len(payload)),parse_soft(text,series)
        except Exception as error:return dict(series=series,url=url,status='unavailable',error=repr(error)),[]
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(fetch,ev['geo_series']))
    write_json(out/'metadata_sources.json',[x[0] for x in results]);samples=[v for _,rows in results for v in rows];write_json(out/'geo_samples.json',samples)
    local=pd.read_csv('data/annotations/samples.csv');usage=pd.read_csv('data/annotations/sample_usage.csv');sixth=json.loads((Path(cfg['sixth_output'])/'conditions/protocol.json').read_text(encoding='utf-8'));verified={r['sample_id']:r['source_sha256'] for r in sixth['sources']}
    inventory=[]
    for row in local.itertuples():
        used=usage[usage.sample_id==row.sample_id].iloc[0];exists=Path(row.path).exists();is_wt=row.condition=='37C';known_used=row.sample_id in verified
        item=dict(sample_id=row.sample_id,condition=row.condition,replicate=row.replicate,path=row.path,source_url=row.source_url,registered_sha256=row.sha256,registered_library_pairs=row.library_pairs,registered_native_bp=row.bin_bp if hasattr(row,'bin_bp') else 10,local_complete=exists,source='GSE272159',previous_usage=used.status,sixth_candidate_review_used=known_used,independent_wt_validation=False,role='Historical WT fitting/evaluation' if is_wt else ('Previously inspected descriptive perturbation/control' if known_used else 'Uninspected perturbation, not matched WT replicate'),compatibility='Same registered genome; treatment/protocol differs from WT' if not is_wt else 'Matched WT but already used')
        if exists:
            c=cooler.Cooler(row.path);item.update(actual_binsize=c.binsize,actual_chromsizes=str(c.chromsizes.to_dict()),actual_library_pairs=int(c.info['sum']),prior_verified_sha256=verified.get(row.sample_id,''),native_reference_compatible=c.binsize==10 and c.chromsizes.to_dict()=={'NC_000913.3':4641652})
        inventory.append(item)
    pd.DataFrame(inventory).to_csv(out/'local_data_inventory.csv',index=False)
    remote_rows=[]
    local_gsms={str(s).split('_')[0] for s in local.sample_id if str(s).startswith('GSM')}
    for sample in samples:
        title=' | '.join(sample.get('Sample_title',[]));chars=' | '.join(sample.get('Sample_characteristics_ch1',[]));files=sample.get('Sample_supplementary_file',[])
        remote_rows.append(dict(series=sample['series'],accession=sample['accession'],title=title,organism=' | '.join(sample.get('Sample_organism_ch1',[])),characteristics=chars,protocol=' | '.join(sample.get('Sample_extract_protocol_ch1',[])),processing=' | '.join(sample.get('Sample_data_processing',[])),supplementary_files=' | '.join(files),accession_in_local_gsm_ids=sample['accession'] in local_gsms,modality_or_reference='Requires explicit protocol/reference verification; no independent Micro-C compatibility inferred from organism alone'))
    pd.DataFrame(remote_rows).to_csv(out/'remote_sample_inventory.csv',index=False)
    # Check author annotation changes without executing or importing remote code.
    author=ev['author_repository'];author_record={};label_rows=[]
    try:
        payload=limited_get(f'https://api.github.com/repos/{author}/commits/main',ev['metadata_max_bytes'],ev['network_timeout_seconds']);commit=json.loads(payload)['sha'];(out/'author_latest_commit.json').write_bytes(payload)
        old_record_path=Path('results/experiment/reference_resolution.json')
        if old_record_path.exists():old=json.loads(old_record_path.read_text(encoding='utf-8'))
        else:
            path=root/'course_audit/committed_evidence/results/experiment/reference_resolution.json';old=json.loads(path.read_text(encoding='utf-8'))
        for source in old['provenance']:
            if source['kind']=='genes':continue
            url=f"https://raw.githubusercontent.com/{author}/{commit}/data/{source['name']}";data=limited_get(url,ev['metadata_max_bytes'],ev['network_timeout_seconds']);(out/source['name']).write_bytes(data)
            rows=[line for line in data.decode('utf-8').splitlines() if line.strip() and not line.startswith('#')]
            label_rows.append(dict(type=source['kind'],latest_commit=commit,old_commit=source['commit'],url=url,old_sha256=source['sha256'],latest_sha256=hashlib.sha256(data).hexdigest(),content_unchanged=hashlib.sha256(data).hexdigest()==source['sha256'],latest_rows=len(rows),source_independence='Same author/study manual annotations, not an independent validation label set'))
        author_record=dict(status='verified_small_labels',latest_commit=commit)
    except Exception as error:author_record=dict(status='unavailable',error=repr(error))
    pd.DataFrame(label_rows).to_csv(out/'author_label_compatibility.csv',index=False)
    write_json(out/'author_lookup.json',author_record)
    # Save the candidate external modalities explicitly; actual sample metadata above
    # determines whether any previously unknown replicates exist.
    write_json(out/'status.json',dict(local_samples=len(local),local_complete_samples=sum(x['local_complete'] for x in inventory),historical_wt_repeats=int(local.condition.eq('37C').sum()),already_reviewed_conditions=sixth['unique_selected_centers'],remote_metadata_samples=len(samples),metadata_only=True,large_data_downloaded=False,independent_validation_performed=False,external_compatibility_not_assumed=True,reviewer_available=False,author_label_lookup=author_record))
    print('Independent-evidence inventory and small source/label checks completed',flush=True)

if __name__=='__main__':main()

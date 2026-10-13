"""Resolve rounded course annotations against the paper's versioned BED files."""
import hashlib,json,urllib.request
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

root=Path('data/annotations/source_reference')
record=Path('results/experiment/reference_resolution.json')
commit=json.loads(record.read_text())['provenance'][0]['commit'] if record.exists() else json.load(urllib.request.urlopen('https://api.github.com/repos/irzhegalova/ecoli_microc/commits/main',timeout=30))['sha']
files={'OPCID':'TADS_25.bedpe','CHIN':'hairpins_25.bed','CHID':'hTADS_25.bedpe','genes':'NC_000913.gff'}
provenance=[]
for kind,name in files.items():
    url=f'https://raw.githubusercontent.com/irzhegalova/ecoli_microc/{commit}/data/{name}'
    cached=root/name
    payload=cached.read_bytes() if cached.exists() else urllib.request.urlopen(url,timeout=30).read()
    if record.exists():
        old=next(p for p in json.loads(record.read_text())['provenance'] if p['name']==name)
        if hashlib.sha256(payload).hexdigest()!=old['sha256']:raise ValueError('Pinned reference checksum differs')
    (root/name).write_bytes(payload)
    provenance.append(dict(kind=kind,name=name,url=url,commit=commit,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
source=pd.read_csv('data/annotations/structures_source.csv');frames=[]
for label in ['CHID','CHIN','OPCID']:
    f=source[source.type==label].copy().reset_index(drop=True)
    bed=pd.read_csv(root/files[label],sep='\t',header=None)
    if label=='CHIN': start=bed[1].to_numpy();end=bed[2].to_numpy()
    else: start=bed[1].to_numpy();end=bed[5].to_numpy()
    if len(f)!=len(start): raise ValueError(f'Source count mismatch: {label}')
    cost=abs(f.start.to_numpy()[:,None]-start[None,:])+abs(f.end.to_numpy()[:,None]-end[None,:])
    i,j=linear_sum_assignment(cost)
    if (cost[i,j]>100).any(): raise ValueError('Cannot reliably match course table to paper BED')
    f['course_start']=f.start;f['course_end']=f.end
    f.loc[i,'start']=start[j];f.loc[i,'end']=end[j]
    f['reference_bed']=files[label];f['reference_row']=j+1
    f['coordinate_status']='paper_BED_zero_based_half_open'
    f['center']=(f.start+f.end)//2;f['length_bp']=f.end-f.start
    f['coordinate_change_bp']=abs(f.start-f.course_start)+abs(f.end-f.course_end)
    frames.append(f)
resolved=pd.concat(frames,ignore_index=True)
resolved.to_csv('data/annotations/structures.csv',index=False)
genes=[]
for line in (root/files['genes']).read_text().splitlines():
    if not line or line.startswith('#'):continue
    row=line.split('\t')
    if row[2]!='gene':continue
    if row[0]!='NC_000913.3': raise ValueError('Reference differs from Cooler')
    attrs=dict(x.split('=',1) for x in row[8].split(';') if '=' in x)
    genes.append(dict(chrom=row[0],start=int(row[3])-1,end=int(row[4]),strand=row[6],name=attrs.get('Name',attrs['ID']),locus_tag=attrs.get('locus_tag','')))
pd.DataFrame(genes).to_csv('data/annotations/genes.csv',index=False)
Path('results/experiment/reference_resolution.json').write_text(json.dumps(dict(provenance=provenance,structures=len(resolved),genes=len(genes),max_total_coordinate_change_bp=int(resolved.coordinate_change_bp.max()),coordinate_convention='BED: zero based half open; GFF: one based inclusive converted start-1',source_preserved='structures_source.csv and original workbook retained'),indent=2)+'\n')
print('resolved',len(resolved),'structures;',len(genes),'genes; maximum boundary difference sum',resolved.coordinate_change_bp.max(),flush=True)

"""Class/fold failure diagnosis and geometric context audit, no scoring changes."""
import argparse
import gzip
import json
from pathlib import Path
import requests
from urllib.parse import quote
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from dlpro.coverage_support import coarse_support
from dlpro.method_review import load_config,write_json,sha256
from dlpro.novelty import grouped_auc

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output']);out=root/'novelty_diagnosis';out.mkdir(exist_ok=True);assert not (out/'status.json').exists()
    source=Path(cfg['sixth_output'])/'background_ae';frame=pd.read_csv(source/'point_scores.csv');fits=json.loads((source/'fits.json').read_text(encoding='utf-8'));rows=[]
    for (hidden,fold,repeat,method),part in frame.groupby(['hidden_type','fold','repeat','method']):
        for negative in ['allowed_known','background']:
            use=part[(part.actual_type==hidden)|((part.actual_type=='background') if negative=='background' else ((part.actual_type!=hidden)&(part.actual_type!='background')))]
            y=use.actual_type.eq(hidden).to_numpy();result=grouped_auc(y,use.score.to_numpy(),use.group.to_numpy(),cfg['seed'],cfg['bootstrap_draws'])
            fit=next(f for f in fits if f['hidden_type']==hidden and f['fold']==fold)
            rows.append(dict(hidden_type=hidden,fold=fold,repeat=repeat,method=method,negative=negative,n_hidden=int(y.sum()),n_negative=int((~y).sum()),background_training=len(fit['background_training_indices']),known_references=len(fit['reference_indices']),validation_references=len(fit['calibration_reference_indices']),fusion_enabled=fit['joint_allowed'],status='valid' if y.any() and (~y).any() else 'unavailable_single_class',**result))
    pd.DataFrame(rows).to_csv(out/'per_fold_novelty.csv',index=False)
    known=pd.read_csv('data/processed/known_windows.csv');scan=pd.read_csv('data/processed/scan_windows.csv');centers=np.r_[known.center,scan.center];n=len(known)
    ids=list(known.structure_id)+[f'scan_{i}' for i in range(len(scan))];nuis=[]
    for r in (1,2):
        raw=np.concatenate([np.load(f'data/processed/known_rep{r}.npz')['raw'],np.load(f'data/processed/scan_rep{r}.npz')['raw']])[:,0]
        support=coarse_support(np.load(f'data/processed/coverage_rep{r}.npz')['coverage'],centers)[:,0]>=cfg['discovery']['support_fraction'];triangle=np.triu(np.ones(raw.shape[-2:],bool),2)
        for i,image in enumerate(raw):
            valid=triangle&support[i,:,None]&support[i,None,:];v=image[valid]
            nuis.append(dict(item_id=ids[i],repeat=r,log_density=float(np.log(max(v.mean() if len(v) else 0,1e-12))),zero_fraction=float((v==0).mean()) if len(v) else 1.,support_fraction=float(support[i].mean()),valid_unique_pixels=int(valid.sum())))
    nuisance=pd.DataFrame(nuis);joined=frame.merge(nuisance,on=['item_id','repeat'],validate='many_to_one');joined.to_csv(out/'all_scores_with_nuisance.csv',index=False)
    correlations=[]
    for key,part in joined.groupby(['hidden_type','fold','repeat','method']):
        for actual,sub in part.groupby('actual_type'):
            for field in ['log_density','zero_fraction','support_fraction','valid_unique_pixels']:
                value=float(spearmanr(sub.score,sub[field]).statistic) if len(sub)>2 and sub[field].nunique()>1 and sub.score.nunique()>1 else None
                correlations.append(dict(zip(['hidden_type','fold','repeat','method'],key),actual_type=actual,n=len(sub),nuisance=field,spearman=value))
    pd.DataFrame(correlations).to_csv(out/'within_class_nuisance_correlations.csv',index=False)
    joined.groupby(['hidden_type','fold','repeat','actual_type'])[['log_density','zero_fraction','support_fraction','valid_unique_pixels']].agg(['mean','median','min','max','count']).to_csv(out/'class_nuisance_distributions.csv')
    reconstruction=pd.read_csv(source/'reconstruction_per_window.csv');reconstruction.groupby(['hidden_type','fold','repeat','actual_type'])[['raw_mse','calibrated_residual','residual_rank']].agg(['mean','median','std','count']).to_csv(out/'class_reconstruction_distributions.csv')
    overlaps=[]
    for i,left in enumerate(known.itertuples()):
        for j,right in enumerate(known.itertuples()):
            if j<=i:continue
            overlap=max(0,min(left.end,right.end)-max(left.start,right.start))
            if overlap:overlaps.append(dict(left_id=left.structure_id,left_type=left.type,right_id=right.structure_id,right_type=right.type,overlap_bp=overlap,left_contains_right=left.start<=right.start and left.end>=right.end,right_contains_left=right.start<=left.start and right.end>=left.end))
    pairs=pd.DataFrame(overlaps);pairs.to_csv(out/'annotation_overlap_pairs.csv',index=False);pairs.groupby(['left_type','right_type']).agg(overlapping_pairs=('overlap_bp','size'),left_contains_right=('left_contains_right','sum'),right_contains_left=('right_contains_left','sum')).to_csv(out/'annotation_overlap_summary.csv')
    taxonomy=pd.read_csv(root/'blind_review/private/context_taxonomy.csv');taxonomy.groupby('context_relation').size().rename('windows').to_csv(out/'candidate_context_taxonomy.csv')
    metadata=root/'independent_data';remote=json.loads((metadata/'geo_samples.json').read_text(encoding='utf-8'));unique={x['accession']:x for x in remote};wt=[];series_files=[]
    for x in unique.values():
        title=' '.join(x.get('Sample_title',[]))
        if 'WT_MicroC_t' in title:wt.append(dict(accession=x['accession'],title=title,characteristics=x.get('Sample_characteristics_ch1',[]),processing=x.get('Sample_data_processing',[]),files=[v for k,values in x.items() if k.startswith('Sample_supplementary_file') for v in values]))
    for series in cfg['evidence']['geo_series']:
        path=metadata/f'{series}.soft.gz'
        if path.exists():
            for line in gzip.decompress(path.read_bytes()).decode('utf-8').splitlines():
                if line.startswith('!Series_supplementary_file = '):series_files.append(dict(series=series,url=line.split(' = ',1)[1]))
    write_json(metadata/'compatibility_summary.json',dict(metadata_rows=len(remote),unique_geo_accessions=len(unique),super_series_duplicates_not_independent=True,wt_library_records=wt,series_supplementary_files=series_files,conclusion='No unused independent matched WT contact matrix established. t1/t2 library titles alone do not prove independence from the two already-used series-level rep1/rep2 matrices; per-library processed WT files are NONE in the fetched metadata. Raw-read provenance/merging would need explicit verification before reuse.',HiC='GSE272158 is same-study Hi-C modality; not a deeper Micro-C native10bp target',RNA_RedC='Gene-expression/RNA-DNA assays are potential functional context, not native DNA-contact ground truth',author_labels='Latest author BED content is identical to the currently pinned344 labels; not independent labels',large_files_downloaded=False))
    # Author-provided condition-specific hairpin tables are small contextual labels,
    # never merged into the frozen WT training set.
    tree=json.loads((metadata/'author_tree.json').read_text(encoding='utf-8')) if (metadata/'author_tree.json').exists() else None
    supplemental=[]
    if tree:
        commit=json.loads((metadata/'author_lookup.json').read_text(encoding='utf-8'))['latest_commit']
        for item in tree.get('tree',[]):
            if item['path'].startswith('data/') and item['path'].endswith('.xlsx') and item.get('size',0)<cfg['evidence']['metadata_max_bytes'] and ('Hp' in item['path']):
                url=f"https://raw.githubusercontent.com/{cfg['evidence']['author_repository']}/{commit}/"+quote(item['path'])
                response=requests.get(url,timeout=cfg['evidence']['network_timeout_seconds']);response.raise_for_status();path=metadata/Path(item['path']).name;path.write_bytes(response.content)
                try:
                    book=pd.read_excel(path,sheet_name=None);sheets={name:dict(rows=len(f),columns=list(map(str,f.columns))) for name,f in book.items()}
                except Exception as error:sheets=dict(unreadable=repr(error))
                supplemental.append(dict(url=url,sha256=sha256(path),sheets=sheets,role='Same-author, condition-specific context only; coordinate and WT-condition compatibility not assumed; not used for scoring or new-class confirmation'))
    write_json(metadata/'supplemental_label_audit.json',supplemental)
    write_json(out/'status.json',dict(all_folds_retained=True,per_fold_rows=len(rows),annotation_overlap_pairs=len(pairs),new_scoring_model_trained=False,old_formal_rules_unchanged=True,geometry_categories_are_not_biological_labels=True,independent_blind_review_completed=False,most_defensible_novelty='Reference-distance signal limited to one held class; AE residual has no all-class robust improvement'))
    print('Class/fold, nuisance, nesting and independent-source limitations audited',flush=True)

if __name__=='__main__':main()

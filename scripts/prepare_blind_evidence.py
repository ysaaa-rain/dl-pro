"""Regenerate frozen candidates and export all anonymous cases, never human verdicts."""
import argparse
import csv
import json
from pathlib import Path
import secrets
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from dlpro.coverage_support import coarse_support
from dlpro.experiment_core import LENGTH,centered_indices
from dlpro.method_review import load_config,nms,sha256,write_json
from dlpro.evidence_review import anonymous_id,context_relation

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output']);out=root/'blind_review';private=out/'private';public=out/'public'
    private.mkdir(parents=True,exist_ok=True);public.mkdir(exist_ok=True);assert not (public/'packet_manifest.json').exists()
    dc,ec=cfg['discovery'],cfg['evidence'];sixth=Path(cfg['sixth_output']);directory=sixth/'background_ae'
    known=pd.read_csv('data/processed/known_windows.csv');scan=pd.read_csv(sixth/'data/scan_eligibility.csv');nk=len(known);old=pd.read_csv(directory/'selected_scan_windows.csv');fits=json.loads((directory/'fits.json').read_text(encoding='utf-8'))
    regenerated=[]
    for fit in fits:
        hidden,fold=fit['hidden_type'],fit['fold'];eligible=np.flatnonzero(scan.eligible&scan.new_fold.eq(fold));scores=np.load(directory/f'{hidden}_fold{fold}/all_scores.npz')
        for r in (1,2):
            for method in sorted(old.method.unique()):
                score=scores[f'{method}_rep{r}'][nk+eligible]
                for budget in dc['global_budgets']:
                    count=int(np.ceil(budget*len(eligible)/len(scan)));selected=eligible[nms(score,scan.center.to_numpy()[eligible],count,dc['nms_bp'],LENGTH)]
                    for rank,i in enumerate(selected,1):regenerated.append(dict(hidden_type=hidden,fold=fold,repeat=r,method=method,global_equivalent_budget=budget,rank=rank,scan_index=int(i),center=int(scan.center.iloc[i]),score=float(scores[f'{method}_rep{r}'][nk+i])))
    frame=pd.DataFrame(regenerated);keys=['hidden_type','fold','repeat','method','global_equivalent_budget','rank']
    pair=old.merge(frame,on=keys,validate='one_to_one',suffixes=('_old','_new'))
    assert len(pair)==len(old)==len(frame) and pair.center_old.eq(pair.center_new).all() and pair.scan_index_old.eq(pair.scan_index_new).all()
    assert np.allclose(pair.score_old,pair.score_new,rtol=0,atol=1e-12)
    frame.to_csv(private/'regenerated_ranked_candidates.csv',index=False)
    secret_path=private/'identity_key.txt'
    if not secret_path.exists():secret_path.write_text(secrets.token_hex(32),encoding='ascii')
    secret=secret_path.read_text(encoding='ascii');unique=frame.sort_values(['hidden_type','fold','center']).drop_duplicates('center')
    unique['case_id']=[anonymous_id(c,secret) for c in unique.center];assert unique.case_id.nunique()==len(unique)
    unique.to_csv(private/'coordinate_mapping.csv',index=False)
    raw={r:np.load(f'data/processed/scan_rep{r}.npz')['raw'] for r in (1,2)};support={r:coarse_support(np.load(f'data/processed/coverage_rep{r}.npz')['coverage'],scan.center.to_numpy())[:,0]>=dc['support_fraction'] for r in (1,2)}
    index_by_center={int(c):i for i,c in enumerate(scan.center)};n=ec['review_image_bins'];distance=abs(np.arange(n)[:,None]-np.arange(n)[None]);triangle=np.triu(np.ones((n,n),bool),2)
    fit_lookup={(f['hidden_type'],f['fold']):f for f in fits};cards=[];geometry=[];(public/'cases').mkdir(exist_ok=True)
    for row in unique.sort_values('case_id').itertuples():
        i=index_by_center[int(row.center)];fit=fit_lookup[(row.hidden_type,row.fold)];train=np.asarray(fit['background_training_indices'],int)-nk
        # Control is selected by nuisance similarity from this case's allowed training background.
        def nuis(ids):return np.stack([np.log(raw[1][ids,0].mean((1,2)).clip(1e-12)),(raw[1][ids,0]==0).mean((1,2)),support[1][ids].mean(1)],axis=1)
        bg=nuis(train);center=bg.mean(0);spread=bg.std(0).clip(1e-8);control=int(train[cdist((nuis([i])-center)/spread,(bg-center)/spread).argmin()])
        expected=np.load(directory/f'{row.hidden_type}_fold{row.fold}/transform.npz')['expected'][0,distance]
        arrays=[];masks=[];fractions=[]
        for index in (i,control):
            pair_arrays=[];pair_masks=[]
            for r in (1,2):
                keep=triangle&support[r][index,:,None]&support[r][index,None,:];oe=raw[r][index,0]/expected;amplitude=max(float(oe[keep].mean()) if keep.any() else 0,1e-12)
                pair_arrays.append(np.log1p(oe/amplitude));pair_masks.append(support[r][index,:,None]&support[r][index,None,:])
            arrays.append(pair_arrays);masks.append(pair_masks);fractions.append(float((pair_masks[0]&pair_masks[1]&triangle).sum()/triangle.sum()))
        start=centered_indices(int(row.center))[1]*dc['input_scales_bp'][-1];end=start+ec['review_context_bp']
        hit=known[(known.start<end)&(known.end>start)]
        annotations=[dict(type=x.type,relative_start_bp=max(-ec['review_context_bp']//2,int(x.start-row.center)),relative_end_bp=min(ec['review_context_bp']//2,int(x.end-row.center)),clipped=bool(x.start<start or x.end>end)) for x in hit.itertuples()]
        category=context_relation(int(row.center),start,end,known.start,known.end,ec['nearby_context_buffer_bp'])
        geometry.append(dict(case_id=row.case_id,center=int(row.center),context_relation=category,strict_original_unannotated=not bool(hit.shape[0]),existing_formal_context_exclusion_unchanged=True))
        npz=public/'cases'/f'{row.case_id}.npz';png=public/'cases'/f'{row.case_id}.png'
        np.savez_compressed(npz,candidate_coarse_signal_per_million=np.stack([raw[r][i,0] for r in (1,2)]),background_coarse_signal_per_million=np.stack([raw[r][control,0] for r in (1,2)]),candidate_log_shape=np.asarray(arrays[0]),background_log_shape=np.asarray(arrays[1]),valid_pixel_masks=np.asarray(masks),relative_axis_bp=np.arange(n)*ec['review_image_bp']-ec['review_window_bp']//2)
        fig,axes=plt.subplots(2,3,figsize=(9,6),constrained_layout=True)
        for j,role in enumerate(['Candidate','Background control']):
            for r in range(2):
                im=axes[j,r].imshow(np.where(masks[j][r],arrays[j][r],np.nan),origin='lower',cmap='magma',vmin=0,vmax=ec['review_shape_vmax'],extent=[-6400,6400,-6400,6400]);axes[j,r].set_title(f'{role} WT{r+1}');axes[j,r].set_xlabel('Relative bp')
            axes[j,2].imshow(np.asarray(masks[j]).all(0),origin='lower',cmap='gray',vmin=0,vmax=1);axes[j,2].set_title(f'Common valid pixels: {fractions[j]:.1%}')
        fig.suptitle(row.case_id);fig.colorbar(im,ax=axes[:,:2],shrink=.75,label='Amplitude-normalized log(1+O/E)');fig.savefig(png,dpi=105);plt.close(fig)
        cards.append(dict(case_id=row.case_id,image=f'cases/{row.case_id}.png',data=f'cases/{row.case_id}.npz',candidate_common_fraction=fractions[0],control_common_fraction=fractions[1],known_context=annotations,image_sha256=sha256(png),data_sha256=sha256(npz)))
        if len(cards)%40==0:print('Anonymous cases',len(cards),'/',len(unique),flush=True)
    pd.DataFrame(geometry).to_csv(private/'context_taxonomy.csv',index=False)
    write_json(public/'packet_manifest.json',dict(cases=cards,review_categories=ec['review_categories'],status='Awaiting human review; automatic quality checks only',mask_definition='Native80percent support; common valid upper triangle excluding first two diagonals',data_definition='100bp coarse mean signal per million, not independent native reads',review_instruction='Inspect every case; known context is metadata, not a model prediction. Record uncertainty and disagreements; no new biological class is established by this review alone.'))
    template=[dict(case_id=c['case_id'],reviewer='',assessment='',evidence='',replicate_visible='',boundary_distinct='',independence_declaration='') for c in cards]
    with (public/'review_template.csv').open('w',encoding='utf-8',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(template[0]));w.writeheader();w.writerows(template)
    # Offline, self-contained index (images/data remain local relative files).
    html='''<!doctype html><meta charset="utf-8"><title>匿名候选复核</title><style>body{font-family:system-ui;max-width:1100px;margin:auto;padding:24px}img{max-width:100%}textarea{width:100%;height:80px}button,select,input{font-size:16px;margin:8px}pre{white-space:pre-wrap}</style><h1>匿名候选复核</h1><p>全部案例按匿名ID展示。请独立查看两个重复、有效像素、背景对照和注释上下文。当前未完成任何人工复核。</p><label>复核人<input id="reviewer"></label><select id="case"></select><img id="image"><p><a id="data">下载实际矩阵</a></p><pre id="context"></pre><select id="assessment"></select><textarea id="notes" placeholder="记录依据和不确定性"></textarea><button id="save">保存本例到本机浏览器</button><button id="export">导出复核JSON</button><p>导出后由项目人员验证并导入；不要发送私有坐标映射、算法评分或排序给复核者。此网页不会自动认定新结构。</p><script>const packet=PACKET;const c=document.getElementById('case'),a=document.getElementById('assessment');packet.cases.forEach(x=>c.add(new Option(x.case_id,x.case_id)));packet.review_categories.forEach(x=>a.add(new Option(x,x)));function show(){const x=packet.cases.find(x=>x.case_id===c.value);image.src=x.image;data.href=x.data;context.textContent=JSON.stringify({known_context:x.known_context,candidate_common_fraction:x.candidate_common_fraction},null,2);notes.value='';}c.onchange=show;save.onclick=()=>{if(!reviewer.value.trim()||!notes.value.trim())return alert('请填写复核人和依据');const r=JSON.parse(localStorage.getItem('blindReviews')||'{}');r[c.value]={case_id:c.value,reviewer:reviewer.value,assessment:a.value,evidence:notes.value,client_time:new Date().toISOString()};localStorage.setItem('blindReviews',JSON.stringify(r));};document.getElementById('export').onclick=()=>{const b=new Blob([JSON.stringify(Object.values(JSON.parse(localStorage.getItem('blindReviews')||'{}')),null,2)],{type:'application/json'});const x=document.createElement('a');x.href=URL.createObjectURL(b);x.download='blind_reviews.json';x.click();};show();</script>'''
    (public/'index.html').write_text(html.replace('PACKET',json.dumps(dict(cases=cards,review_categories=ec['review_categories']),ensure_ascii=False)),encoding='utf-8')
    files=sorted(p for p in public.rglob('*') if p.is_file());packet=out/'blind_review_packet.zip'
    with zipfile.ZipFile(packet,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,path.relative_to(public).as_posix())
    forbidden={'score','rank','method','algorithm','center','prediction','absolute_start','absolute_end'}
    def audit_keys(value):
        if isinstance(value,dict):
            assert not (set(value)&forbidden)
            for x in value.values():audit_keys(x)
        elif isinstance(value,list):
            for x in value:audit_keys(x)
    audit_keys(cards)
    write_json(out/'automatic_quality_audit.json',dict(regenerated_records=len(frame),all_frozen_candidates_identical=True,unique_cases=len(cards),anonymous_ids_unique=True,public_has_no_scoring_or_coordinate_keys=True,zip_has_no_private_members=all(not n.startswith('private/') for n in zipfile.ZipFile(packet).namelist()),packet_sha256=sha256(packet),all_cases_pending=True,independent_human_validation_completed=False))
    write_json(out/'selection_freeze.json',dict(source_selected_sha256=sha256(directory/'selected_scan_windows.csv'),source_fits_sha256=sha256(directory/'fits.json'),source_clustering_sha256=sha256(directory/'cluster_quality.csv'),original_thresholds_unchanged=True,new_structure_class_confirmed=False))
    print('Complete anonymous packet generated; human review remains pending',flush=True)

if __name__=='__main__':main()

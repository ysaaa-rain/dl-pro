"""First-stage audit: protect existing work, rederive every historical classification error."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import requests
import numpy as np
import pandas as pd
import fitz
from sklearn.metrics import roc_auc_score,average_precision_score
from dlpro.method_review import load_config,write_json,sha256
from dlpro.experiment_core import LABELS
from dlpro.classification_review import load_inputs,metrics,check_spatial_partition

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config);root=Path(cfg['output']);out=root/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'status.json').exists()
    state={name:subprocess.check_output(cmd,text=True,encoding='utf-8').strip() for name,cmd in dict(head=['git','rev-parse','HEAD'],branch=['git','branch','--show-current'],status=['git','-c','core.quotepath=false','status','--short','--branch'],staged=['git','diff','--cached','--name-only'],stashes=['git','stash','list'],remote=['git','remote','-v']).items()};assert state['head']==cfg['source_revision'];write_json(out/'initial_git_state.json',state)
    previous=Path(cfg['previous_output']);inventory=pd.read_csv(previous/'verification/change_manifest.csv');protected=json.loads((previous/'protected_existing.json').read_text(encoding='utf-8'))
    for row in inventory.itertuples():assert sha256(row.path)==row.sha256;protected[row.path]=row.sha256
    for path in [previous/'verification/change_manifest.csv',previous/'verification/change_manifest.sha256']:protected[path.as_posix()]=sha256(path)
    for directory,pattern in [('src','*.py'),('scripts','*.py'),('configs','*.json'),('tests','*.py'),('data/processed','*.npz'),('data/processed','*.csv')]:
        for path in Path(directory).rglob(pattern):
            if path.name in ['classification_review.py','classification_review_v1.json','run_classification_review.py','audit_classification_review.py']:continue
            protected[path.as_posix()]=sha256(path)
    for path in Path('results/classification_fine').glob('*'):
        if path.is_file():protected[path.as_posix()]=sha256(path)
    write_json(root/'protected_existing.json',protected)
    doc=fitz.open(cfg['teacher_pdf']);write_json(out/'teacher_requirements.json',dict(path=cfg['teacher_pdf'],sha256=sha256(cfg['teacher_pdf']),pages=[dict(page=i+1,text=doc[i].get_text()) for i in [7,8,9,10,11,12,13]]))
    tree=subprocess.check_output(['git','ls-tree','-r','--name-only','HEAD'],text=True,encoding='utf-8').splitlines();restored=[]
    for name in tree:
        if not name.startswith('results/classification') or Path(name).name not in ['out_of_fold_predictions.csv','protocol.json','summary.csv']:continue
        path=Path(name)
        if not path.exists():
            response=requests.get(f"https://raw.githubusercontent.com/ysaaa-rain/dl-pro/{cfg['source_revision']}/{name}",timeout=cfg['timeout_seconds']);response.raise_for_status();payload=response.content;assert len(payload)<=cfg['maximum_small_asset_bytes'];path=out/'historical_assets'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        payload=path.read_bytes();oid=subprocess.check_output(['git','rev-parse',f'HEAD:{name}'],text=True).strip();assert hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()==oid
        restored.append(dict(original_path=name,local_path=path.as_posix(),sha256=sha256(path),git_blob=oid))
    write_json(out/'historical_assets.json',restored)
    frame,folds,raw,support=load_inputs();n=len(frame);k=len(LABELS);y=frame.type.map(dict(zip(LABELS,range(k)))).to_numpy();assert frame.assign(fold=folds).groupby('group').fold.nunique().max()==1;check_spatial_partition(frame.window_start.to_numpy(),frame.window_end.to_numpy(),folds)
    all_predictions=[]
    for item in restored:
        if not item['original_path'].endswith('out_of_fold_predictions.csv'):continue
        f=pd.read_csv(item['local_path']);f=f.rename(columns={'type':'true_type','prediction':'predicted_type'})
        if 'recipe' not in f:f['recipe']='default'
        folder=Path(item['original_path']).parent.name
        for recipe,part in f.groupby('recipe'):
            part=part.set_index('structure_id').loc[frame.structure_id].reset_index();assert len(part)==n and np.array_equal(part.true_type,frame.type)
            part['method']=folder+'/'+recipe;all_predictions.append(part)
    old=pd.concat(all_predictions,ignore_index=True);old.to_csv(out/'all_historical_predictions.csv',index=False);summary=[];class_rows=[];group_rows=[];fold_rows=[]
    for name,part in old.groupby('method'):
        pred=part.predicted_type.map(dict(zip(LABELS,range(k)))).to_numpy();m=metrics(y,pred,k);summary.append(dict(method=name,n_loci=len(part),accuracy=float(m['accuracy']),macro_f1=float(m['macro_f1'])))
        for c,label in enumerate(LABELS):class_rows.append(dict(method=name,type=label,precision=float(m['precision'][c]),recall=float(m['recall'][c]),f1=float(m['f1'][c]),support=int(m['support'][c])))
        for field,target in [('group',group_rows),('fold',fold_rows)]:
            for group,sub in part.groupby(field):
                a=sub.true_type.map(dict(zip(LABELS,range(k)))).to_numpy();b=sub.predicted_type.map(dict(zip(LABELS,range(k)))).to_numpy();sm=metrics(a,b,k);target.append(dict(method=name,**{field:int(group)},n=len(sub),accuracy=float(sm['accuracy']),macro_f1=float(sm['macro_f1'])))
    pd.DataFrame(summary).to_csv(out/'historical_overall.csv',index=False);pd.DataFrame(class_rows).to_csv(out/'historical_per_class.csv',index=False);pd.DataFrame(group_rows).to_csv(out/'historical_per_group.csv',index=False);pd.DataFrame(fold_rows).to_csv(out/'historical_per_fold.csv',index=False)
    baseline=old[old.method=='classification_fine/fine_consistent_ensemble'].copy();baseline=baseline.set_index('structure_id').loc[frame.structure_id].reset_index();probs=baseline[[f'p_{label}' for label in LABELS]].to_numpy();pred=probs.argmax(1);confusion=pd.crosstab(pd.Categorical(frame.type,categories=LABELS),pd.Categorical(np.asarray(LABELS)[pred],categories=LABELS),dropna=False);confusion.to_csv(out/'baseline_confusion.csv')
    baseline['correct']=pred==y;baseline['confidence']=probs.max(1);baseline['true_probability']=probs[np.arange(n),y];baseline['true_probability_margin']=probs[np.arange(n),y]-np.max(np.where(np.eye(k,dtype=bool)[y],-np.inf,probs),axis=1);baseline['true_rank']=1+(probs>probs[np.arange(n),y,None]).sum(1)
    distance=abs(np.arange(128)[:,None]-np.arange(128)[None,:]);upper=np.triu(np.ones((128,128),bool),2);outer=np.ones_like(upper);outer[32:96,32:96]=False;quality=[]
    for i,row in frame.iterrows():
        q=dict(structure_id=row.structure_id,group=int(row.group),fold=int(folds[i]),type=row.type)
        for channel,bp in enumerate(cfg['classifier']['physical_bins_bp']):
            values=[]
            for repeat in range(2):
                a=raw[i+repeat*n,channel];s=support[i+repeat*n,channel]>=cfg['classifier']['support_fraction'];valid=upper&s[:,None]&s[None,:];v=a[valid];bg=a[valid&outer]
                q.update({f'{bp}bp_rep{repeat+1}_support_fraction':float(valid.sum()/upper.sum()),f'{bp}bp_rep{repeat+1}_mean':float(v.mean()) if len(v) else None,f'{bp}bp_rep{repeat+1}_zero_fraction':float((v==0).mean()) if len(v) else None,f'{bp}bp_rep{repeat+1}_background_mean':float(bg.mean()) if len(bg) else None});values.append(np.log1p(a[upper]))
            q[f'{bp}bp_repeat_log_signal_pearson']=float(np.corrcoef(*values)[0,1]) if min(v.std() for v in values)>0 else None
        quality.append(q)
    quality=pd.DataFrame(quality);joined=baseline.merge(quality,on=['structure_id','group','fold'],validate='one_to_one');joined.to_csv(out/'all_baseline_cases_with_quality.csv',index=False);joined[~joined.correct].to_csv(out/'all_errors_with_quality.csv',index=False)
    joined.groupby(['true_type','predicted_type']).agg(cases=('structure_id','size'),confidence_mean=('confidence','mean'),true_probability_mean=('true_probability','mean')).to_csv(out/'confidence_error_directions.csv')
    fields=[c for c in quality if c not in ['structure_id','group','fold','type']];joined.groupby(['true_type','correct'])[fields].agg(['median','mean','count']).to_csv(out/'quality_by_type_and_error.csv')
    ranking=[dict(type=label,n=int((y==c).sum()),one_vs_rest_auc=float(roc_auc_score(y==c,probs[:,c])),average_precision=float(average_precision_score(y==c,probs[:,c]))) for c,label in enumerate(LABELS)];pd.DataFrame(ranking).to_csv(out/'baseline_ranking.csv',index=False)
    # Exact conflicts are diagnostic only: no label edits, row deletions, or sample-ID features.
    signatures={};collisions=[]
    for i in range(n):
        digest=hashlib.sha256(np.stack([raw[i],raw[i+n]]).tobytes()).hexdigest()
        for j in signatures.get(digest,[]):
            if y[i]!=y[j]:collisions.append(dict(first=frame.structure_id.iloc[j],second=frame.structure_id.iloc[i],first_type=frame.type.iloc[j],second_type=frame.type.iloc[i],same_group=bool(frame.group.iloc[i]==frame.group.iloc[j]),identical_all_three_scales_both_repeats=True))
        signatures.setdefault(digest,[]).append(i)
    pd.DataFrame(collisions,columns=['first','second','first_type','second_type','same_group','identical_all_three_scales_both_repeats']).to_csv(out/'exact_input_conflicts.csv',index=False)
    overlap=[]
    for i,left in frame.iterrows():
        for j,right in frame.iloc[i+1:].iterrows():
            size=max(0,min(left.end,right.end)-max(left.start,right.start))
            if size and left.type!=right.type:overlap.append(dict(first=left.structure_id,second=right.structure_id,first_type=left.type,second_type=right.type,overlap_bp=int(size),same_group=bool(left.group==right.group),first_error=bool(not joined.correct.iloc[i]),second_error=bool(not joined.correct.iloc[j])))
    pd.DataFrame(overlap).to_csv(out/'cross_type_annotation_overlaps.csv',index=False)
    sample_counts=pd.crosstab(folds,frame.type);sample_counts.to_csv(out/'fold_class_counts.csv')
    work=[dict(priority=1,item='分类错误与历史方案审计',current='已实际复核全部折外行及合法质量特征',acceptance='逐类、逐组、逐折、置信度、FP/FN和不利方案均保留'),dict(priority=2,item='至多两种定向分类方法',current='需审计后冻结方案；不重复旧网格',acceptance='空间隔离、训练内拟合、外层同例比较与配对不确定性'),dict(priority=3,item='结构特异性遮挡',current='旧中心/角落不是标注结构区域',acceptance='标注区间代理、等面积等距离支持背景、多填充值与失败例'),dict(priority=4,item='真实独立候选复核',current='第七轮包保留，实际人员待提供',acceptance='人员独立性与真实记录；无记录仍0/243'),dict(priority=5,item='任务三定向核验',current='连续索引和首中末6张检查复用',acceptance='按条件、结构、背景和边界选图，记录实际检查范围'),dict(priority=6,item='任务五冻结',current='强基线和停止结论复用',acceptance='正式选择不变；没有新实现缺陷不新增训练')];pd.DataFrame(work).to_csv(out/'worklist.csv',index=False)
    write_json(out/'status.json',dict(original_teacher_read=True,seventh_manifest_verified=len(inventory),protected_files=len(protected),n_loci=n,n_groups=int(frame.group.nunique()),baseline_metrics={key:float(value) for key,value in metrics(y,pred,k).items() if np.ndim(value)==0},historical_methods=len(summary),actual_primary_checkpoint_files=len(list(Path(cfg['classifier']['source']).glob('fine_consistent_seed*_fold*.pt'))),actual_primary_transform_files=len(list(Path(cfg['classifier']['source']).glob('transform_fold*.npz'))),not_fifteen_weight_hyperparameter_recipes=True,exact_cross_label_input_pairs=len(collisions),outer_context_overlap_pass=True,independent_final_test=False,new_experiments_started=False))
    print('All historical errors, source identities, label conflicts and spatial groups audited',flush=True)

if __name__=='__main__':main()

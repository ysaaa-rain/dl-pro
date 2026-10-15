"""Local annotation assistant with frozen-fold inference and append-only review storage."""
import csv
import hashlib
import io
import json
import sqlite3
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dlpro.classification_models import FineScaleClassifier
from dlpro.experiment_core import LENGTH,LABELS,ContactReader,ShapeTransform,centered_indices

class AnnotationEngine:
    def __init__(self,root):
        self.root=Path(root);torch.set_num_threads(4)
        self.known=pd.read_csv(self.root/'data/processed/known_windows.csv')
        table=pd.read_csv(self.root/'results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id')
        self.folds=table.loc[self.known.structure_id,'fold'].to_numpy();self.models={};self.readers=None
        manifest=json.loads((self.root/'data/processed/fine_windows_sources.json').read_text());self.sources=manifest['sources']
        if manifest['known_plan_sha256']!=hashlib.sha256((self.root/'data/processed/known_windows.csv').read_bytes()).hexdigest():raise ValueError('Known cache coordinate signature mismatch')
        files=sorted((self.root/'results/classification_fine').glob('fine_consistent_seed*_fold*.pt'))+sorted((self.root/'results/classification_fine').glob('transform_fold*.npz'))
        if len(files)!=20:raise ValueError('Expected15frozen models and5transforms')
        self.artifacts={str(p.relative_to(self.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        self.signature=hashlib.sha256(json.dumps(self.artifacts,sort_keys=True).encode()).hexdigest()
    def raw(self,center):
        # Exact saved coordinates reproduce published OOF inputs without requiring large raw downloads.
        ids=np.flatnonzero(self.known.center.to_numpy()==center)
        if len(ids):
            i=int(ids[0]);return np.stack([np.concatenate([np.load(self.root/f'data/processed/fine_rep{r}.npz')['raw'][i],np.load(self.root/f'data/processed/known_rep{r}.npz')['raw'][i]],axis=0) for r in [1,2]]),'known_cache'
        scan=pd.read_csv(self.root/'data/processed/scan_windows.csv');ids=np.flatnonzero(scan.center.to_numpy()==center)
        if len(ids):
            manifest=json.loads((self.root/'data/processed/fine_scan_sources.json').read_text())
            if manifest['sources']!=self.sources or manifest['plan_sha256']!=hashlib.sha256((self.root/'data/processed/scan_windows.csv').read_bytes()).hexdigest():raise ValueError('Scan cache source signature mismatch')
            i=int(ids[0]);return np.stack([np.concatenate([np.load(self.root/f'data/processed/fine_scan_rep{r}.npz')['raw'][i],np.load(self.root/f'data/processed/scan_rep{r}.npz')['raw'][i]],axis=0) for r in [1,2]]),'scan_cache'
        if self.readers is None:
            samples=pd.read_csv(self.root/'data/annotations/samples.csv');samples=samples[samples.condition=='37C'].sort_values('replicate');
            for sample in samples.itertuples():
                if self.sources.get(sample.sample_id)!=sample.sha256:raise ValueError('WT source manifest mismatch')
                if hashlib.sha256((self.root/sample.path).read_bytes()).hexdigest()!=sample.sha256:raise ValueError('WT raw file hash mismatch')
            self.readers=[ContactReader(self.root/r.path) for r in samples.itertuples()]
        starts=[int(np.floor((center+25)/50))-64,*centered_indices(center)]
        return np.stack([np.stack([r.window(s,128,bp) for s,bp in zip(starts,[50,100,200])]) for r in self.readers]),'raw_cool'
    def load(self,fold):
        if fold not in self.models:
            models=[]
            for seed in [42,43,44]:
                m=FineScaleClassifier();m.load_state_dict(torch.load(self.root/f'results/classification_fine/fine_consistent_seed{seed}_fold{fold}.pt',weights_only=True)['state_dict']);m.eval();models.append(m)
            self.models[fold]=(models,ShapeTransform.load(self.root/f'results/classification_fine/transform_fold{fold}.npz'))
        return self.models[fold]
    def predict(self,center):
        if isinstance(center,bool) or not isinstance(center,int) or not 0<=center<LENGTH:raise ValueError(f'坐标必须是0到{LENGTH-1}之间的整数（0-based）')
        # Match actual rounded200bp extraction extents, including circular seams.
        start=centered_indices(center)[1]*200;end=start+25600;overlap=np.zeros(len(self.known),bool)
        for shift in [-LENGTH,0,LENGTH]:overlap|=(self.known.window_start.to_numpy()<end+shift)&(self.known.window_end.to_numpy()>start+shift)
        excluded=np.unique(self.folds[overlap]);fold=int(excluded[0]) if len(excluded)==1 else int(np.argmin([sum(overlap&(self.folds!=k)) for k in range(5)]))
        evaluation_safe=bool(np.all(self.folds[overlap]==fold));raw,source=self.raw(center);models,tf=self.load(fold);x=torch.from_numpy(tf.transform(raw)).requires_grad_();logits=torch.stack([m(x) for m in models]);prob=logits.softmax(-1).mean((0,1));target=int(prob.argmax());score=logits[:,:,target].mean();gradient=torch.autograd.grad(score,x)[0].detach().abs().numpy();probability=prob.detach().numpy();view=tf.transform(raw)
        starts=[int(np.floor((center+25)/50))-64,*centered_indices(center)];intervals=[dict(bin_bp=bp,start_bp=int((s%((LENGTH+bp-1)//bp))*bp),end_bp=int(min(LENGTH,((s+127)%((LENGTH+bp-1)//bp)+1)*bp)),wraps=bool(s%((LENGTH+bp-1)//bp)+128>(LENGTH+bp-1)//bp)) for s,bp in zip(starts,[50,100,200])]
        return dict(center=center,view_intervals=intervals,model_artifacts_sha256=self.artifacts,coordinate_system='0-based center, circular genome',fold=fold,predicted_type=LABELS[target],probabilities=dict(zip(LABELS,map(float,probability))),repeat_probabilities=logits.softmax(-1).mean(0).detach().tolist(),input_source=source,model_sha256=self.signature,source_sha256=self.sources,evaluation_safe=evaluation_safe,overlapping_annotations=self.known.loc[overlap,['structure_id','type']].to_dict('records'),warning='仅供辅助标注：概率未校准；梯度为敏感性，非因果证据。'+(' 当前窗口未进入所用模型的训练/验证。' if evaluation_safe else ' 当前上下文涉及多个折，含训练/验证重叠；只可探索，不可作为折外评价。'),views=view.tolist(),saliency=(gradient/max(float(gradient.max()),1e-12)).tolist(),saliency_target=LABELS[target],saliency_definition='Three-seed mean predicted-class logit gradient, two repeats, joint relative magnitude')

class ReviewStore:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:db.execute('CREATE TABLE IF NOT EXISTS reviews(id INTEGER PRIMARY KEY, created TEXT NOT NULL, center INTEGER NOT NULL, decision TEXT NOT NULL, note TEXT NOT NULL, prediction_json TEXT NOT NULL)')
    def connect(self):return sqlite3.connect(self.path)
    def save(self,prediction,decision,note):
        if decision not in [*LABELS,'不确定','其他']:raise ValueError('无效人工类别')
        if not isinstance(note,str) or len(note)>2000:raise ValueError('备注最长2000字符')
        snapshot={k:v for k,v in prediction.items() if k not in ['views','saliency']}
        with self.connect() as db:
            cur=db.execute('INSERT INTO reviews(created,center,decision,note,prediction_json) VALUES(?,?,?,?,?)',(datetime.now(timezone.utc).isoformat(),prediction['center'],decision,note,json.dumps(snapshot,ensure_ascii=False)));return cur.lastrowid
    def records(self):
        with self.connect() as db:return [dict(id=r[0],created=r[1],center=r[2],decision=r[3],note=r[4],prediction=json.loads(r[5])) for r in db.execute('SELECT * FROM reviews ORDER BY id DESC')]
    def export_csv(self):
        f=io.StringIO();writer=csv.writer(f);writer.writerow(['id','created','center_0based','human_label','note','model_label','model_sha256','fold','evaluation_safe'])
        for r in self.records():
            # Prevent spreadsheet formulas when opening human-authored text in Excel.
            note=r['note'];note="'"+note if note.lstrip().startswith(('=','+','-','@')) else note;p=r['prediction'];writer.writerow([r['id'],r['created'],r['center'],r['decision'],note,p['predicted_type'],p['model_sha256'],p['fold'],p['evaluation_safe']])
        return '\ufeff'+f.getvalue()

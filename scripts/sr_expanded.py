"""Native 10bp SR with larger patches, replicate holdout and full known contexts."""
import copy,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from skimage.metrics import structural_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dlpro.superresolution import block_plan,mean_pool,scores
from dlpro.experiment_core import ContactReader,Classifier,ShapeTransform,LABELS,centered_indices,save_json,configure,SubpixelSR

out=Path('results/sr_expanded');out.mkdir(parents=True,exist_ok=True)
cache=Path('data/processed/sr_expanded');cache.mkdir(exist_ok=True)
samples=pd.read_csv('data/annotations/samples.csv');wt=samples[samples.condition=='37C'].sort_values('replicate')
readers={int(s.replicate):ContactReader(s.path) for s in wt.itertuples()};lib_ref=readers[1].library_size
plan=block_plan(4641652,seed=42,patch_bins=1024);records=[]
for rep in [1,2]:
    for row in plan.itertuples():
        if rep==2 and row.split!='test':continue
        records.append(dict(group=row.group,start=row.start,end=row.end,split=row.split,replicate=rep,depth_multiplier=lib_ref/readers[rep].library_size))
frame=pd.DataFrame(records);frame.to_csv(cache/'windows.csv',index=False)
rawpath=cache/'raw_counts.npy'
if not rawpath.exists():
    raw=np.lib.format.open_memmap(rawpath,mode='w+',dtype='uint32',shape=(len(frame),1024,1024))
    for i,row in frame.iterrows():
        counts=readers[int(row.replicate)].window(int(row.start)//10,n=1024,bin_bp=10,depth_normalize=False)
        if not np.array_equal(counts,np.rint(counts)):raise ValueError('Unexpected noninteger native counts')
        raw[i]=counts.astype('uint32')
        if i%20==0:print(f'SR native extraction {i+1}/{len(frame)}',flush=True)
    raw.flush();del raw
raw=np.load(rawpath,mmap_mode='r')
if raw.shape!=(len(frame),1024,1024):raise ValueError('Unexpected cached geometry')
# Small compressed shards permit reproducing training without the full genome files.
for start in range(0,len(frame),12):
    dest=cache/f'counts_{start:03d}.npz'
    if not dest.exists():np.savez_compressed(dest,raw=raw[start:start+12],start=start)
train=frame.split.eq('train').to_numpy();val=frame.split.eq('val').to_numpy();test=frame.split.eq('test').to_numpy()
hist=np.zeros(1,dtype='int64')
for idx in np.flatnonzero(train):
    h=np.bincount(raw[idx].ravel())
    if len(h)>len(hist):hist=np.pad(hist,(0,len(h)-len(hist)))
    hist[:len(h)]+=h
rank=(hist.sum()-1)*.995;cumulative=np.cumsum(hist)
a=np.searchsorted(cumulative,np.floor(rank)+1);b=np.searchsorted(cumulative,np.ceil(rank)+1);fraction=rank-np.floor(rank)
scale=float((1-fraction)*np.log1p(a)+fraction*np.log1p(b))
if scale<=0:raise ValueError('Degenerate intensity scale')


class Patches(Dataset):
    def __init__(self,indices):self.ids=np.asarray(indices)
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        idx=self.ids[i];counts=raw[idx].astype('float32')*float(frame.iloc[idx].depth_multiplier)
        low=np.clip(np.log1p(mean_pool(counts))/scale,0,1).astype('float32')
        high=np.clip(np.log1p(counts)/scale,0,1).astype('float32')
        return torch.from_numpy(low[None]),torch.from_numpy(high[None])




def quality(high,pred):
    q=scores(high,pred)
    q['shape_ssim_160bp']=float(structural_similarity(mean_pool(high,16),mean_pool(np.clip(pred,0,1),16),data_range=1))
    return q


def infer(model,low):
    with torch.no_grad():return model(torch.from_numpy(low[None,None]).to(device)).clamp(0,1).cpu().numpy()[0,0]


# Record hardware explicitly; determinism on MPS is platform dependent.
device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');configure(42)
probe=SubpixelSR().to(device);probe_in=torch.ones(1,1,16,16,device=device);probe_out=probe(probe_in);probe_out.mean().backward()
if not torch.isfinite(probe_out).all() or not all(torch.isfinite(p.grad).all() for p in probe.parameters()):raise ValueError('Device numerical probe failed')
print('SR device',device,'samples',len(frame),'scale',scale,flush=True)
if '--evaluate-only' in sys.argv:
    test_ids=np.flatnonzero(test)
    summary=pd.read_csv(out/'metrics_summary.csv')
else:
    rows=[];baseline_predictions={};test_ids=np.flatnonzero(test)
    for idx in test_ids:
        low,high=Patches([idx])[0];target=high.numpy()[0]
        for mode in ['bilinear','bicubic']:
            pred=F.interpolate(low[None],scale_factor=4,mode=mode,align_corners=False).clamp(0,1).numpy()[0,0]
            baseline_predictions[(idx,mode)]=pred
            rows.append(dict(seed=0,method=mode,sample_index=int(idx),replicate=int(frame.iloc[idx].replicate),**quality(target,pred)))
    for seed in [42,43,44]:
        configure(seed);model=SubpixelSR().to(device);optim=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');state=None;history=[]
        generator=torch.Generator().manual_seed(seed)
        train_loader=DataLoader(Patches(np.flatnonzero(train)),batch_size=4,shuffle=True,generator=generator,num_workers=0)
        val_loader=DataLoader(Patches(np.flatnonzero(val)),batch_size=4,shuffle=False,num_workers=0)
        for epoch in range(15):
            model.train();losses=[]
            for low,high in train_loader:
                low=low.to(device);high=high.to(device);optim.zero_grad();prediction=model(low)
                # Regional shape consistency is evaluated separately as well.
                loss=F.mse_loss(prediction,high)+.2*F.mse_loss(F.avg_pool2d(prediction,16),F.avg_pool2d(high,16))
                loss.backward();optim.step();losses.append(float(loss.detach().cpu()))
            model.eval();numerator=0;denominator=0
            with torch.no_grad():
                for low,high in val_loader:
                    pred=model(low.to(device)).clamp(0,1);loss=F.mse_loss(pred,high.to(device));numerator+=float(loss.cpu())*len(low);denominator+=len(low)
            vl=numerator/denominator;history.append(dict(epoch=epoch+1,train_objective=float(np.mean(losses)),val_mse=vl))
            if vl<best:best=vl;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};selected=epoch+1
            if epoch%3==0:print(f'SR seed {seed} epoch {epoch+1} val MSE {vl:.6f}',flush=True)
        model.load_state_dict(state);model.eval();torch.save(dict(state_dict=state,scale=scale,selected_epoch=selected,seed=seed,architecture='SubpixelSR'),out/f'model_seed{seed}.pt')
        pd.DataFrame(history).to_csv(out/f'history_seed{seed}.csv',index=False)
        if seed==42:reconstruction=np.lib.format.open_memmap(out/'reconstruction_seed42.npy',mode='w+',dtype='float32',shape=(len(test_ids),1024,1024))
        for local,idx in enumerate(test_ids):
            low,high=Patches([idx])[0];pred=infer(model,low.numpy()[0]);rows.append(dict(seed=seed,method='subpixel_cnn',sample_index=int(idx),replicate=int(frame.iloc[idx].replicate),**quality(high.numpy()[0],pred)))
            if seed==42:reconstruction[local]=pred
        if seed==42:reconstruction.flush();del reconstruction
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics_per_window.csv',index=False)
    summary=metrics.groupby(['seed','replicate','method'])[['psnr','ssim','shape_ssim_160bp','off_diagonal_mse']].mean();summary.to_csv(out/'metrics_summary.csv')
# Full 25.6 kb known contexts are never used for fitting SR or selecting checkpoints.
structures=pd.read_csv('data/processed/known_windows.csv');fold_table=pd.read_csv('results/classification_cv/out_of_fold_predictions.csv').set_index('structure_id')
blocks=plan.drop_duplicates('group').set_index('group').split
eligible=structures.center.floordiv(100000).map(blocks).eq('test')&structures.window_start.floordiv(100000).eq(structures.window_end.sub(1).floordiv(100000))
known_test=structures[eligible].copy();known_test.to_csv(out/'known_test_windows.csv',index=False)
model=SubpixelSR().to(device);model.load_state_dict(torch.load(out/'model_seed42.pt',weights_only=True)['state_dict']);model.eval()
classifiers={};transforms={}
for fold in range(5):
    classifier=Classifier()
    if fold==0:path=Path('results/classification/model_seed42.pt');tf=Path('results/classification/transform.npz')
    else:path=Path(f'results/classification_cv/model_fold{fold}.pt');tf=Path(f'results/classification_cv/transform_fold{fold}.npz')
    classifier.load_state_dict(torch.load(path,weights_only=True)['state_dict']);classifier.eval();classifiers[fold]=classifier;transforms[fold]=ShapeTransform.load(tf)
recovery=[];classifier_rows=[];pictured=set()
for row in known_test.itertuples():
    fold=int(fold_table.loc[row.structure_id].fold);start=int(row.window_start);end=int(row.window_end)
    paired_probs={method:[] for method in ['original','clipped_target','bilinear','bicubic','subpixel_cnn']}
    for repeat in [1,2]:
        counts=readers[repeat].window(start//10,n=2560,bin_bp=10,depth_normalize=False)*(lib_ref/readers[repeat].library_size)
        target=np.clip(np.log1p(counts)/scale,0,1).astype('float32');low=np.clip(np.log1p(mean_pool(counts))/scale,0,1).astype('float32')
        matrices={'original':np.log1p(counts)/scale,'clipped_target':target,
            'bilinear':F.interpolate(torch.from_numpy(low[None,None]),scale_factor=4,mode='bilinear',align_corners=False).clamp(0,1).numpy()[0,0],
            'bicubic':F.interpolate(torch.from_numpy(low[None,None]),scale_factor=4,mode='bicubic',align_corners=False).clamp(0,1).numpy()[0,0],
            'subpixel_cnn':infer(model,low)}
        small=(centered_indices(int(row.center))[0]*100-start)//10
        shape_matrices={}
        for method,matrix in matrices.items():
            decoded=counts if method=='original' else np.expm1(matrix*scale)
            features=np.stack([mean_pool(decoded[small:small+1280,small:small+1280],10),mean_pool(decoded,20)]).astype('float32')*1e6/lib_ref
            transformed=transforms[fold].transform(features[None])
            with torch.no_grad():prob=F.softmax(classifiers[fold](torch.from_numpy(transformed)),1).numpy()[0]
            paired_probs[method].append(prob)
            shape_matrices[method]=transformed[0,1].copy()
            if method=='original':oe_reference=transformed[0,1].copy()
            if method!='original':
                aa=max(0,(int(row.start)-start)//10);bb=min(2560,(int(row.end)-start+9)//10)
                q=scores(target[aa:bb,aa:bb],matrix[aa:bb,aa:bb])
                q['context_log_intensity_ssim_200bp']=float(structural_similarity(mean_pool(target,20),mean_pool(matrix,20),data_range=1))
                q['context_oe_ssim_original']=float(structural_similarity(oe_reference,transformed[0,1],data_range=1))
                triangle=np.triu_indices(128,k=2);q['context_oe_correlation_original']=float(np.corrcoef(oe_reference[triangle],transformed[0,1][triangle])[0,1])
                recovery.append(dict(structure_id=row.structure_id,type=row.type,replicate=repeat,method=method,full_structure_contained=True,**q))
        if repeat==1 and row.type not in pictured:
            pictured.add(row.type);fig,axes=plt.subplots(1,5,figsize=(15,3.4),constrained_layout=True)
            for ax,(method,matrix) in zip(axes,shape_matrices.items()):
                im=ax.imshow(matrix,cmap='magma',vmin=0,vmax=1,origin='lower');aa=(row.start-start)/200;bb=(row.end-start)/200
                ax.add_patch(Rectangle((aa,aa),bb-aa,bb-aa,fill=False,edgecolor='cyan',lw=1));ax.set_title(method,fontsize=8);ax.set_xlabel('200 bp bin')
            axes[0].set_ylabel('200 bp bin');fig.suptitle(f'{row.structure_id}; full 25.6 kb context',fontsize=10);fig.colorbar(im,ax=axes,label='OOF training-scaled log(1+O/E)',shrink=.8)
            fig.savefig(out/f'known_{row.type}.png',dpi=130);plt.close(fig)
    for method,prob in paired_probs.items():
        pp=np.mean(prob,axis=0);classifier_rows.append(dict(structure_id=row.structure_id,true_type=row.type,fold=fold,method=method,predicted_type=LABELS[pp.argmax()]))
    print('SR known recovery',row.structure_id,flush=True)
pd.DataFrame(recovery).to_csv(out/'known_recovery.csv',index=False);classification=pd.DataFrame(classifier_rows);classification.to_csv(out/'downstream_predictions.csv',index=False)
baseline=classification[classification.method=='original'].set_index('structure_id').predicted_type
from sklearn.metrics import f1_score
rows=[]
for method,part in classification.groupby('method'):
    agreement=(part.predicted_type.to_numpy()==baseline.loc[part.structure_id].to_numpy()).mean()
    rows.append(dict(method=method,n_loci=len(part),accuracy=float(part.true_type.eq(part.predicted_type).mean()),macro_f1=float(f1_score(part.true_type,part.predicted_type,labels=LABELS,average='macro',zero_division=0)),agreement_with_original_oof=float(agreement)))
pd.DataFrame(rows).to_csv(out/'downstream_summary.csv',index=False)
save_json(out/'protocol.json',dict(seed_of_spatial_split=42,training_seeds=[42,43,44],device=str(device),epochs=15,train_samples=int(train.sum()),val_samples=int(val.sum()),test_samples=int(test.sum()),native_target_bp=10,low_input_bp=40,training_window_bp=10240,known_context_bp=25600,known_test_loci=len(known_test),known_test_classes=known_test.type.value_counts().to_dict(),
    fit_only_repeat1=True,replicate2_only_evaluated=True,depth_normalization='Both repeats scaled to WT rep1 library pair count before log transform',normalization_scale=scale,normalization_fit='Exact 99.5% training-only log-count quantile from integer histogram',checkpoint_selection='Validation MSE; representative seed42 chosen in advance',loss='Native MSE + 0.2 MSE of 160bp pooled intensity',
    full_known_contexts_in_test_blocks=True,downstream_classifier='Per-locus out-of-fold classifier, no test-locus class labels in its fitting',
    reconstruction_file='reconstruction_seed42.npy',compressed_source_shards='data/processed/sr_expanded/counts_*.npz',
    limitations=['MPS bitwise determinism not guaranteed across machines','targets clip top 0.5% training intensity; clipping reference separately evaluated downstream','same-source spatial holdout and one withheld repeat, not external-source validation','classifier agreement is a recovery proxy, not independent biological confirmation','only annotated structures fully inside SR test blocks are evaluated downstream']))
print(summary.to_string(),flush=True);print(pd.DataFrame(rows).to_string(index=False),flush=True)

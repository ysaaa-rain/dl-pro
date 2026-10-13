"""Shape-only anomaly scan, coordinate exclusions and independent-repeat checks."""
import copy,warnings,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sklearn.isotonic import IsotonicRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dlpro.experiment_core import Autoencoder,ShapeTransform,DensityInvariantTransform,Classifier,configure,save_json,LENGTH,LABELS,hits,centered_indices

parser=argparse.ArgumentParser();parser.add_argument('--reuse-autoencoder');parser.add_argument('--score-mode',choices=['amplitude_retained','amplitude_invariant','noise_adjusted'],default='noise_adjusted');parser.add_argument('--output',default='results/discovery');parser.add_argument('--no-figures',action='store_true');args=parser.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'heatmaps').mkdir(exist_ok=True)
known=pd.read_csv('data/processed/known_windows.csv');scan=pd.read_csv('data/processed/scan_windows.csv')
r1=np.load('data/processed/scan_rep1.npz')['raw'];r2=np.load('data/processed/scan_rep2.npz')['raw']
configure(42)
blocks=scan.loc[scan.training_background,'background_block'].unique()
order=np.random.default_rng(42).permutation(blocks);train_blocks=set(order[:int(.7*len(order))]);val_blocks=set(order[int(.7*len(order)):])
train=scan.training_background.to_numpy()&scan.background_block.isin(train_blocks).to_numpy()
val=scan.training_background.to_numpy()&scan.background_block.isin(val_blocks).to_numpy()
if train.sum()<10 or val.sum()<5:raise ValueError('Insufficient separated unannotated background')
transform=(ShapeTransform() if args.score_mode=='amplitude_retained' else DensityInvariantTransform()).fit(r1[train]);transform.save(out/'transform.npz')
x1=transform.transform(r1)[:,:1];x2=transform.transform(r2)[:,:1]
invariance_error=float(np.max(abs(transform.transform(r1[:10])-transform.transform(r1[:10]*10))))
if args.score_mode!='amplitude_retained' and invariance_error>1e-5:raise ValueError('Density nuisance invariance failed')
model=Autoencoder();optim=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(42)
best=float('inf');state=None;history=[];tx=torch.from_numpy(x1)
if args.reuse_autoencoder:
    prior=ShapeTransform.load(Path(args.reuse_autoencoder)/'transform.npz')
    if type(prior)!=type(transform) or not np.allclose(prior.expected,transform.expected) or not np.allclose(prior.scale,transform.scale):raise ValueError('Reused AE transform differs from requested mode')
    checkpoint=torch.load(Path(args.reuse_autoencoder)/'model.pt',weights_only=True)
    state=checkpoint['state_dict'];selected_epoch=checkpoint['selected_epoch']
    history=pd.read_csv(Path(args.reuse_autoencoder)/'history.csv').to_dict('records')
else:
    for epoch in range(30):
        model.train();losses=[]
        for ids in np.array_split(rng.permutation(np.flatnonzero(train)),int(np.ceil(train.sum()/16))):
            optim.zero_grad();loss=F.mse_loss(model(tx[ids]),tx[ids]);loss.backward();optim.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():vl=float(F.mse_loss(model(tx[val]),tx[val]))
        history.append(dict(epoch=epoch+1,train_mse=float(np.mean(losses)),val_mse=vl))
        if vl<best:best=vl;state=copy.deepcopy(model.state_dict());selected_epoch=epoch+1
        if epoch%5==0:print(f'AE epoch {epoch+1}; val MSE {vl:.6f}',flush=True)
model.load_state_dict(state);model.eval();torch.save(dict(state_dict=state,selected_epoch=selected_epoch,seed=42,architecture='Autoencoder'),out/'model.pt')
pd.DataFrame(history).to_csv(out/'history.csv',index=False)
with torch.no_grad():
    reconstructed=np.concatenate([model(z).numpy() for z in tx.split(64)])
    embeddings=np.concatenate([model.encoder(z).numpy() for z in tx.split(64)])
    reconstructed2=np.concatenate([model(z).numpy() for z in torch.from_numpy(x2).split(64)])
raw_error=((reconstructed-x1)**2).mean(axis=(1,2,3));raw_error2=((reconstructed2-x2)**2).mean(axis=(1,2,3))
sample_table=pd.read_csv('data/annotations/samples.csv');wt=sample_table[sample_table.condition=='37C'].sort_values('replicate')
libraries=wt.library_pairs.astype(int).tolist()
effective_count=[r[:,0].mean(axis=(1,2))*libraries[i]/1e6*100 for i,r in enumerate([r1,r2])]
if args.score_mode=='noise_adjusted':
    density=[np.log(np.maximum(v,1e-12)) for v in effective_count]
    log_error=[np.log(np.maximum(v,1e-12)) for v in [raw_error,raw_error2]]
    null=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(density[0][train],log_error[0][train])
    edges=np.quantile(density[0][train],[.2,.4,.6,.8]);bin_train=np.searchsorted(edges,density[0][train])
    residual=log_error[0][train]-null.predict(density[0][train]);location=[];scatter=[]
    for b in range(5):
        values=residual[bin_train==b];center=float(np.median(values));location.append(center);scatter.append(max(float(1.4826*np.median(abs(values-center))),.001))
    location=np.asarray(location);scatter=np.asarray(scatter)
    def calibrated(d,e):
        bucket=np.searchsorted(edges,d);return (e-null.predict(d)-location[bucket])/scatter[bucket]
    score=calibrated(density[0],log_error[0]);score2=calibrated(density[1],log_error[1])
    if not np.isfinite(score).all() or not np.isfinite(score2).all():raise ValueError('Nonfinite noise-adjusted score')
    np.savez(out/'conditional_null.npz',density_thresholds=null.X_thresholds_,error_thresholds=null.y_thresholds_,density_edges=edges,location=location,scatter=scatter)
else:
    score=raw_error;score2=raw_error2
centers=scan.center.to_numpy();known_center=known.center.to_numpy();dist=abs(centers[:,None]-known_center[None,:]);covered=np.minimum(dist,LENGTH-dist)<=6400
train_known=known.split.eq('train').to_numpy();wanted=int(np.ceil(.9*train_known.sum()))
coverage=np.zeros(len(known),bool);order=np.argsort(-score);stop=len(order)
for j,idx in enumerate(order):
    coverage|=covered[idx]
    if (coverage&train_known).sum()>=wanted:stop=j+1;break
selected=order[:stop];threshold=float(score[selected[-1]])
retained=[]
for idx in selected:
    if all(min(abs(centers[idx]-centers[j]),LENGTH-abs(centers[idx]-centers[j]))>=12800 for j in retained):retained.append(int(idx))
retained=np.asarray(retained,int)
# Calibration uses mismatched unannotated windows across repeats, not candidate outcomes.
bg=np.flatnonzero(scan.training_background.to_numpy());pairs=[]
triangle=np.triu_indices(128,k=2)
expected=transform.expected[0,abs(np.arange(128)[:,None]-np.arange(128)[None,:])]
def corr(i,j):
    a=(r1[i,0]/expected)[triangle];b=(r2[j,0]/expected)[triangle]
    if a.std()==0 or b.std()==0:return None
    return float(np.corrcoef(a,b)[0,1])
for i in bg:
    eligible=bg[np.minimum(abs(centers[bg]-centers[i]),LENGTH-abs(centers[bg]-centers[i]))>25600]
    if len(eligible):
        value=corr(i,int(rng.choice(eligible)))
        if value is not None:pairs.append(value)
repro_threshold=max(.5,float(np.quantile(pairs,.95))) if pairs else .5
with warnings.catch_warnings(record=True) as numeric_warnings:
    warnings.simplefilter('always')
    pca=PCA(n_components=min(12,len(retained)-1,embeddings.shape[1]),svd_solver='full').fit(embeddings[retained].astype(float))
    compressed=pca.transform(embeddings[retained].astype(float))
    kmeans=KMeans(n_clusters=min(6,len(retained)),n_init=20,random_state=42).fit(compressed)
if not np.isfinite(compressed).all() or not np.isfinite(kmeans.cluster_centers_).all():raise ValueError('Nonfinite clustering result')
np.savez_compressed(out/'embedding.npz',latent=embeddings,pca_components=pca.components_,pca_mean=pca.mean_,centers=kmeans.cluster_centers_)
cls_transform=ShapeTransform.load('results/classification/transform.npz');cls=Classifier();cls.load_state_dict(torch.load('results/classification/model_seed42.pt',weights_only=True)['state_dict']);cls.eval()
with torch.no_grad():cls_prob=np.concatenate([torch.softmax(cls(z),1).numpy() for z in torch.from_numpy(cls_transform.transform(r1)).split(64)])
records=[]
for position,idx in enumerate(retained):
    center=int(centers[idx]);start_index=centered_indices(center)[0]%((LENGTH+99)//100);count=min(128,(LENGTH+99)//100-start_index)
    a=start_index*100;b=min((start_index+count)*100,LENGTH);segments=[(a,b)]
    if count<128:segments.append((0,(128-count)*100))
    matched=known.loc[hits(center,12800,known,buffer=200),'structure_id'].tolist();repeat_corr=corr(idx,idx)
    records.append(dict(candidate_id=f'C{position+1:04d}',scan_index=int(idx),chrom='NC_000913.3',start=segments[0][0],end=segments[-1][1],center=center,length_bp=sum(v-u for u,v in segments),wraps_origin=len(segments)>1,
        cluster=int(kmeans.labels_[position]),shape_score=float(score[idx]),rep2_shape_score=float(score2[idx]),density=float(r1[idx,0].mean()),repeat_oe_correlation=repeat_corr,
        reproducible=repeat_corr is not None and repeat_corr>=repro_threshold,known_overlap=bool(matched),known_ids=';'.join(matched),classifier_label=LABELS[cls_prob[idx].argmax()],classifier_max_probability=float(cls_prob[idx].max()),
        distinction='within buffered known annotation' if matched else 'no known annotation within window + 200 bp buffer; biological novelty unconfirmed'))
f=pd.DataFrame(records);clusters=[]
for cluster,part in f.groupby('cluster'):
    reproducible_members=int(part.reproducible.sum());known_members=int(part.known_overlap.sum())
    novel=known_members==0 and reproducible_members>=5
    clusters.append(dict(cluster=int(cluster),members=len(part),known_members=known_members,reproducible_members=reproducible_members,tentative_new_cluster=novel,representative=part.sort_values('shape_score',ascending=False).candidate_id.iloc[0]))
cluster_frame=pd.DataFrame(clusters);f['tentative_new_cluster']=f.cluster.isin(cluster_frame.loc[cluster_frame.tentative_new_cluster,'cluster'])
f.to_csv(out/'candidates.csv',index=False)
f[f.reproducible&~f.known_overlap].to_csv(out/'unannotated_reproducible.csv',index=False)
cluster_frame.to_csv(out/'clusters.csv',index=False)
f[f.tentative_new_cluster&f.reproducible&~f.known_overlap].to_csv(out/'novel_candidates.csv',index=False)
scan['raw_reconstruction_mse']=raw_error;scan['effective_coarse_counts']=effective_count[0];scan['shape_score_rep1']=score;scan['shape_score_rep2']=score2;scan['density_rep1']=r1[:,0].mean(axis=(1,2));scan['selected_before_nms']=np.isin(np.arange(len(scan)),selected);scan['retained']=np.isin(np.arange(len(scan)),retained)
scan.to_csv(out/'all_windows.csv',index=False)
recall=[]
for split in ['train','val','test','all']:
    mask=np.ones(len(known),bool) if split=='all' else known.split.eq(split).to_numpy()
    for method,ids in [('shape_before_nms',selected),('shape_after_nms',retained)]:
        hit=covered[ids].any(axis=0)
        recall.append(dict(split=split,method=method,n_structures=int(mask.sum()),n_recalled=int((hit&mask).sum()),center_recall=float(hit[mask].mean())))
pd.DataFrame(recall).to_csv(out/'known_recall.csv',index=False)
# Held-out known-vs-background separation, with full input-window purging.
known_raw=np.load('data/processed/known_rep1.npz')['raw'];known_x=transform.transform(known_raw)[:,:1]
with torch.no_grad():known_z=np.concatenate([model.encoder(z).numpy() for z in torch.from_numpy(known_x).split(64)])
protected_test=known[known.split!='train'][['window_start','window_end']].rename(columns={'window_start':'start','window_end':'end'})
protected_train=known[known.split=='train'][['window_start','window_end']].rename(columns={'window_start':'start','window_end':'end'})
bt=np.asarray([i for i in np.flatnonzero(train) if not hits(int(centers[i]),12800,protected_test).any()])
bv=np.asarray([i for i in np.flatnonzero(val) if not hits(int(centers[i]),12800,protected_train).any()])
separation=dict(n_background_train=len(bt),n_background_holdout=len(bv),window_purging=True)
if len(bt)>=5 and len(bv)>=5:
    ktrain=known.split.eq('train').to_numpy();ktest=known.split.eq('test').to_numpy()
    bx=np.concatenate([known_z[ktrain],embeddings[bt]]);by=np.r_[np.ones(ktrain.sum()),np.zeros(len(bt))]
    vx=np.concatenate([known_z[ktest],embeddings[bv]]);vy=np.r_[np.ones(ktest.sum()),np.zeros(len(bv))]
    lr=LogisticRegression(class_weight='balanced',max_iter=3000,random_state=42).fit(bx.astype(float),by);p=lr.predict_proba(vx.astype(float))[:,1]
    zz=np.einsum('nd,kd->nk',vx.astype(float),lr.coef_,optimize=False).ravel()+lr.intercept_[0]
    ref=1/(1+np.exp(-zz));delta=float(abs(ref-p).max())
    if delta>1e-6:raise ValueError('Separation predictor failed independent arithmetic')
    separation['independent_probability_max_error']=delta
    if not np.isfinite(p).all():raise ValueError('Nonfinite separation model probabilities')
    separation.update(balanced_accuracy=float(balanced_accuracy_score(vy,p>=.5)),roc_auc=float(roc_auc_score(vy,p)))
else:separation['limitation']='Too few fully purged held-out backgrounds; no metric claimed'
save_json(out/'representation_check.json',separation)
for row in ([] if args.no_figures else f.itertuples()):
    idx=row.scan_index;fig,axes=plt.subplots(1,2,figsize=(7,3.2),constrained_layout=True)
    for ax,matrix,title in zip(axes,[x1[idx,0],x2[idx,0]],['WT rep1','WT rep2']):
        im=ax.imshow(matrix,cmap='magma',vmin=0,vmax=1,origin='lower');ax.set_title(title);ax.set_xlabel('Relative 100 bp bin')
    axes[0].set_ylabel('Relative 100 bp bin');fig.colorbar(im,ax=axes,label='Training-scaled log(1+O/E)')
    fig.suptitle(f'{row.candidate_id}; center {row.center:,}; r={row.repeat_oe_correlation:.3f}' if row.repeat_oe_correlation is not None else row.candidate_id,fontsize=9)
    fig.savefig(out/'heatmaps'/f'{row.candidate_id}.png',dpi=110);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5));im=ax.scatter(compressed[:,0],compressed[:,1],c=f.cluster,cmap='tab10',s=25);ax.set(xlabel='PC1',ylabel='PC2',title='Non-overlapping shape candidates');fig.colorbar(im,ax=ax,label='Cluster');fig.tight_layout();fig.savefig(out/'clusters.png',dpi=130);plt.close(fig)
save_json(out/'protocol.json',dict(score_mode=args.score_mode,seed=42,background_train=int(train.sum()),background_val=int(val.sum()),background_blocks_disjoint=True,normalization_fit='Only WT rep1 unannotated training backgrounds; '+('amplitude retained' if args.score_mode=='amplitude_retained' else 'per-window O/E mean cancels amplitude'),autoencoder_epochs=30,selected_epoch=selected_epoch,
    n_genome_windows=len(scan),threshold=threshold,shape_score_units=('Standardized log reconstruction error conditional on training-background coarse counts' if args.score_mode=='noise_adjusted' else 'Reconstruction MSE'),noise_control=('Decreasing isotonic conditional mean plus 5 training-count bins of robust residual scale; no validation/test fit' if args.score_mode=='noise_adjusted' else 'none'),threshold_rule='Largest noise-adjusted shape-score threshold reaching 90% center recall on classification-training known structures only',selected_before_nms=len(selected),retained_after_nms=len(retained),selected_genome_fraction=len(selected)/len(scan),nms_min_center_separation_bp=12800,
    repeat_correlation='Pearson of untransformed O/E upper triangle k>=2; distance expectation from WT rep1 training backgrounds',repeat_threshold=repro_threshold,mismatched_background_pairs=len(pairs),minimum_reproducible_cluster_members=5,known_buffer_bp=200,
    tentative_new_clusters=int(cluster_frame.tentative_new_cluster.sum()),density_shape_spearman=float(spearmanr(score,r1[:,0].mean(axis=(1,2))).statistic),numeric_warning_count=len(numeric_warnings),numeric_outputs_finite=True,density_multiplication_invariance_max_error=invariance_error,
    limitations=['unannotated training background may contain unknown structures','classifier probability is closed-set resemblance, not a calibrated novelty score','conditional null assumes Poisson-like count sparsity effects are monotone; empirical validation remains necessary','threshold calibrated on training labels; held-out recall reported separately','absence of annotations plus reproducibility does not confirm biological novelty']))
print('discovery',len(f),'candidates;',int(cluster_frame.tentative_new_cluster.sum()),'tentative new clusters; representation',separation,flush=True)

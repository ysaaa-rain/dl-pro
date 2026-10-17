"""All-window, all-stratum repeat agreement, with full per-block diagnostics."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from dlpro.experiment_core import ContactReader
from dlpro.method_review import load_config, sha256, write_json
from dlpro.sr_review import load_data
from dlpro.targeted_review import pearson_all

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config)
    out=Path(cfg['output'])/'noise';out.mkdir(exist_ok=True);assert not (out/'stratified_repeat_agreement.csv').exists()
    cache=Path('data/processed')/cfg['version'];cache.mkdir(exist_ok=True)
    plan,raw,depth=load_data(cfg);use=plan[plan.split.isin(['train','val'])]
    n=cfg['sr']['patch_bins']//cfg['sr']['factor'];f=cfg['sr']['factor'];i,j=np.triu_indices(n);diagonal=i==j
    distances=(j-i)*cfg['sr']['native_bp']*f
    storage=np.lib.format.open_memmap(cache/'block_repeat_diagnostics.npy',mode='w+',dtype='float32',shape=(len(use),len(i),4))
    distribution=[];checks=[]
    for pos,(idx,row) in enumerate(use.iterrows()):
        images=[np.asarray(x[idx],dtype='float32')*m for x,m in zip(raw,depth)]
        blocks=[x.reshape(n,f,n,f).transpose(0,2,1,3)[i,j] for x in images]
        masses=[x.sum((1,2)) for x in blocks];low=[v/f**2 for v in masses]
        q=[x/v[:,None,None].clip(1e-12) for x,v in zip(blocks,masses)]
        # On diagonal coarse blocks, compare only unique native upper pixels.
        unique=np.ones_like(blocks[0],bool);unique[diagonal]=np.triu(np.ones((f,f),bool))
        unique_q=[np.where(unique,x,0) for x in q]
        for k in (0,1):unique_q[k]/=unique_q[k].sum((1,2))[:,None,None].clip(1e-12)
        mse=((unique_q[0]-unique_q[1])**2).sum((1,2))/unique.sum((1,2))
        cosine=(unique_q[0]*unique_q[1]).sum((1,2))/np.sqrt((unique_q[0]**2).sum((1,2))*(unique_q[1]**2).sum((1,2))).clip(1e-12)
        storage[pos]=np.stack([masses[0],masses[1],mse,cosine],axis=1)
        # Count strata use paired mean mass, not selected high-count blocks only.
        average=(masses[0]+masses[1])/2
        for lower,upper in zip(cfg['diagnostic']['distance_edges_bp'][:-1],cfg['diagnostic']['distance_edges_bp'][1:]):
            band=(distances>=lower)&(distances<upper)
            for ca,cb in zip(cfg['diagnostic']['count_edges'][:-1],cfg['diagnostic']['count_edges'][1:]):
                keep=band&(average>=ca)&(average<cb);both=keep&(masses[0]>0)&(masses[1]>0)
                record=dict(sample_index=int(idx),start=int(row.start),group=int(row.group),split=row.split,distance_lower_bp=lower,distance_upper_bp=upper,count_lower=ca,count_upper=cb,blocks=int(keep.sum()),both_positive_blocks=int(both.sum()))
                if keep.any():
                    v1,v2=blocks[0][keep],blocks[1][keep]
                    record.update(input40_pearson=pearson_all(low[0][keep],low[1][keep]),normalized_detail_pearson_common_positive=pearson_all(unique_q[0][both][unique[both]],unique_q[1][both][unique[both]]) if both.any() else None,proportion_mse_all_blocks=float(mse[keep].mean()),source1_zero_block_fraction=float((masses[0][keep]==0).mean()),source2_zero_block_fraction=float((masses[1][keep]==0).mean()),native1_zero_fraction=float((v1[unique[keep]]==0).mean()),native2_zero_fraction=float((v2[unique[keep]]==0).mean()),native_repeat_squared_difference=float(((v1-v2)[unique[keep]]**2).mean()),mean_pair_count=float(average[keep].mean()))
                distribution.append(record)
        if row.split=='val':
            # Direct raw Cooler versus native average checks on every validation window.
            samples=pd.read_csv('data/annotations/samples.csv');wt=samples[samples.condition=='37C'].sort_values('replicate')
            for r,sample in enumerate(wt.itertuples()):
                reader=ContactReader(sample.path)
                direct=reader.window(int(row.start)//40,n=n,bin_bp=40,depth_normalize=False)*depth[r]
                aggregate=images[r].reshape(n,f,n,f).mean((1,3))
                error=float(abs(aggregate-direct).max());assert np.allclose(aggregate,direct,rtol=1e-6,atol=1e-6)
                checks.append(dict(sample_index=int(idx),source_repeat=r+1,max_abs_difference=error))
        if pos%30==0:print('All-count noise strata',pos+1,'/',len(use),flush=True)
    storage.flush();pd.DataFrame(distribution).to_csv(out/'stratified_repeat_agreement.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'direct_cooler_aggregation.csv',index=False)
    use.to_csv(out/'per_block_window_index.csv',index=False)
    write_json(out/'protocol.json',dict(windows=len(use),all_spatial_train_val_windows_included=True,per_block_cache=(cache/'block_repeat_diagnostics.npy').as_posix(),per_block_sha256=sha256(cache/'block_repeat_diagnostics.npy'),per_block_shape=list(storage.shape),per_block_fields=['source1_depth_normalized_image_mass','source2_depth_normalized_image_mass','unique_probability_mse','unique_probability_cosine'],coarse_unique_triangle_order='numpy.triu_indices(256)',direct_raw_data_checks=len(checks),selection='All distance/count cells, including empty and zero-only cells; pair mean count strata',caveat='Empirical repeatability; not a theoretical recoverability bound. Biological variation and count noise are not separable with two repeats.',test_data_read=False))

if __name__=='__main__':main()

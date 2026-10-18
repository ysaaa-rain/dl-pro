"""Two matched NLL arms isolate repeat exposure, not architecture or loss changes."""
import argparse
import copy
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from dlpro.experiment_core import configure
from dlpro.method_review import load_config, sha256, write_json
from dlpro.sr_review import load_data, make_batch, make_model
from dlpro.targeted_review import same_objective, supervision_repeats

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=load_config(a.config)
    sr=cfg['sr'];out=Path(cfg['output'])/sr['result_subdir'];out.mkdir(exist_ok=True)
    plan,raw,depth=load_data(cfg);train=np.flatnonzero(plan.split.eq('train'));val=np.flatnonzero(plan.split.eq('val'))
    params=np.load(Path(cfg['previous_output'])/'sr_corrected/normalization.npz')
    assert np.array_equal(params['training_indices'],train)
    scale=float(params['scale']);profile=params['profile'];torch.set_num_threads(cfg['threads'])
    # Same fitted parameters as every fifth-round comparator; never refit on held-out inputs.
    parameter_hash=sha256(Path(cfg['previous_output'])/'sr_corrected/normalization.npz')
    begin=time.monotonic();configure(sr['train_seeds'][0]);torch.set_num_threads(cfg['threads'])
    model=make_model(sr['primary_arm'],scale,profile,sr);optim=torch.optim.Adam(model.parameters(),lr=sr['learning_rate'])
    high,_=make_batch(raw,depth,train[:sr['batch_size']],np.zeros(sr['batch_size'],int))
    for _ in range(sr['resource_probe_steps']):
        optim.zero_grad();loss=same_objective(model,high,sr);assert torch.isfinite(loss);loss.backward();optim.step()
    seconds=(time.monotonic()-begin)/sr['resource_probe_steps']
    write_json(out/'resource_probe.json',dict(seconds_per_step=seconds,estimated_training_seconds=seconds*sr['steps']*len(sr['arms'])*len(sr['train_seeds']),parameters=sum(x.numel() for x in model.parameters()),raw_cache_bytes=sum(x.nbytes for x in raw)))
    print('Resource estimate seconds',seconds*sr['steps']*len(sr['arms'])*len(sr['train_seeds']),flush=True)
    for seed in sr['train_seeds']:
        for arm in sr['arms']:
            path=out/f'{arm}_seed{seed}.pt';assert not path.exists()
            configure(seed);torch.set_num_threads(cfg['threads']);model=make_model(arm,scale,profile,sr)
            optim=torch.optim.Adam(model.parameters(),lr=sr['learning_rate'])
            # Separate RNG streams keep spatial batches and augmentation identical across arms.
            spatial=np.random.default_rng(seed);repeat_rng=np.random.default_rng(seed+cfg['seed'])
            best=float('inf');history=[];losses=[];begin=time.monotonic()
            for step in range(1,sr['steps']+1):
                model.train();ids=spatial.choice(train,sr['batch_size'],replace=False);reverse=spatial.random(len(ids))<.5
                repeats=supervision_repeats(arm,repeat_rng,len(ids));high,_=make_batch(raw,depth,ids,repeats,reverse)
                optim.param_groups[0]['lr']=sr['min_learning_rate']+(sr['learning_rate']-sr['min_learning_rate'])*.5*(1+math.cos(math.pi*(step-1)/sr['steps']))
                optim.zero_grad();loss=same_objective(model,high,sr);assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),sr['gradient_clip']);optim.step();losses.append(float(loss.detach()))
                if step%sr['validate_every']==0:
                    model.eval();values=[]
                    # Both arms selected by the SAME paired-repeat validation objective.
                    with torch.no_grad():
                        for r in (0,1):
                            for i in val:
                                high,_=make_batch(raw,depth,[i],[r]);values.append(float(same_objective(model,high,sr)))
                    value=float(np.mean(values));history.append(dict(step=step,train_loss=float(np.mean(losses)),validation_objective=value,elapsed_seconds=time.monotonic()-begin));losses=[]
                    if value<best:best=value;selected=step;state=copy.deepcopy(model.state_dict())
                    pd.DataFrame(history).to_csv(out/f'{arm}_seed{seed}_history.csv',index=False)
                    print(arm,seed,step,value,flush=True)
            torch.save(dict(state_dict=state,seed=seed,arm=arm,scale=scale,selected_step=selected,validation_objective=best,training_indices=train.tolist(),normalization_sha256=parameter_hash,config_sha256=sha256(a.config,True)),path)
    write_json(out/'training_status.json',dict(completed_models=len(sr['arms'])*len(sr['train_seeds']),normalization_sha256=parameter_hash,train_windows=len(train),train_spatial_groups=int(plan.loc[train,'group'].nunique()),source_semantics='mean16 native symmetric image counts',test_read_during_training=False,paired_validation_repeats=[1,2]))

if __name__=='__main__':main()

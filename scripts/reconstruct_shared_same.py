"""Apply any constrained validation-locked branch from the focused review."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from dlpro.method_review import load_config,sha256
from dlpro.sr_review import make_model,infer_model
from evaluate_allocation_review import simple_predictions

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/targeted_review_v1.json');p.add_argument('--input-low',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',default='validation_selected');p.add_argument('--seed',type=int)
    a=p.parse_args();cfg=load_config(a.config);sr=cfg['sr'];root=Path(cfg['output']);ev=root/'evaluation'
    if a.output.exists():raise FileExistsError(a.output)
    lock=json.loads((ev/'test_lock.json').read_text(encoding='utf-8'))
    selection=json.loads((ev/'validation_selection.json').read_text(encoding='utf-8'))
    assert sha256(ev/'validation_selection.json')==lock['selection_sha256']
    old=Path(cfg['previous_output'])/'sr_corrected';params=np.load(old/'normalization.npz')
    assert sha256(old/'normalization.npz')==lock['parameter_sha256']
    method=selection['selected_method'] if a.method=='validation_selected' else a.method
    seed=sr['train_seeds'][0] if a.seed is None else a.seed
    low=np.load(a.input_low,allow_pickle=False)
    if low.ndim!=2 or low.shape[0]!=low.shape[1] or low.shape[0]*sr['factor']>len(params['profile']):raise ValueError('Input exceeds frozen square patch geometry')
    if not np.isfinite(low).all() or low.min()<0 or not np.allclose(low,low.T,rtol=1e-5,atol=0):raise ValueError('Expected symmetric nonnegative finite40bp image means')
    torch.set_num_threads(cfg['threads']);x=torch.from_numpy(low.astype('float32'))[None,None]
    checkpoint_hash=None
    with torch.no_grad():
        if method.startswith(('allocation_','legacy_','context_')):
            directory=root/sr['result_subdir'] if method in sr['arms'] else old
            path=directory/f'{method}_seed{seed}.pt';checkpoint_hash=sha256(path)
            assert checkpoint_hash==lock['checkpoints'][path.as_posix()]
            cp=torch.load(path,weights_only=True);model=make_model(method,float(params['scale']),params['profile'],sr).eval();model.load_state_dict(cp['state_dict'])
            pred=infer_model(model,method,x,float(params['scale']),sr)[0]
        else:
            if method.endswith('unprojected'):raise ValueError('This application requires exact source mass')
            pred=simple_predictions(x,params['profile'],sr['factor'])[method]
    error=float(((F.avg_pool2d(pred,sr['factor'])-x).abs()/x.clamp_min(1e-8)).max());symmetry=float((pred-pred.transpose(-1,-2)).abs().max()/pred.max().clamp_min(1e-8))
    assert error<1e-5 and symmetry<1e-5 and torch.isfinite(pred).all() and pred.min()>=0
    metadata=dict(method=method,seed=seed if checkpoint_hash else None,checkpoint_sha256=checkpoint_hash,input_sha256=sha256(a.input_low),block_mean_max_relative_error=error,symmetry_max_relative_error=symmetry,input_unit='40bp mean of16 native image pixels',output_unit='10bp expected image counts; not integer measured contacts')
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,counts=pred.numpy()[0,0],low_mean=low,metadata_json=np.asarray(json.dumps(metadata)))
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()

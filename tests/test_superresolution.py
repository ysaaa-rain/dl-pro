import json
import numpy as np
import torch
from dlpro.superresolution import block_plan, mean_pool, ResidualSR, scores
from dlpro.data import inspect_cool


def test_blocks_no_overlap_and_disjoint_groups():
    frame=block_plan(4641652)
    assert len(frame)==92
    assert frame.groupby('group').split.nunique().max()==1
    assert set(frame.split)=={'train','val','test'}
    assert (frame.sort_values('start').start.to_numpy()[1:] >= frame.sort_values('start').end.to_numpy()[:-1]).all()


def test_pool_preserves_counts_and_model_shape():
    a=np.arange(64,dtype=np.float32).reshape(8,8)
    low=mean_pool(a)
    assert np.isclose(low.sum()*16,a.sum())
    model=ResidualSR()
    with torch.no_grad(): y=model(torch.from_numpy(low)[None,None])
    assert y.shape==(1,1,8,8)
    torch.testing.assert_close(y,y.transpose(-1,-2))
    assert scores(a/63,a/63)['ssim']==1

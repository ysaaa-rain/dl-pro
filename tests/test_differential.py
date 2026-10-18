import numpy as np
from dlpro.differential import local_enrichment, repeat_contrast


def test_distance_matching_and_global_depth_invariance():
    centers=(np.arange(60)+.5)*100
    distance=abs(np.arange(60)[:,None]-np.arange(60)[None,:])
    a=1/(distance+1.)
    e,_,n=local_enrichment(a,centers,2000,4000,100)
    scaled,_,_=local_enrichment(a*.13,centers,2000,4000,100)
    assert n>20
    assert np.isclose(e,1) and np.isclose(scaled,e)
    core=(centers>=2000)&(centers<4000)
    a[np.ix_(core,core)]*=2
    enriched,_,_=local_enrichment(a,centers,2000,4000,100)
    assert np.isclose(enriched,2)


def test_all_unpaired_cross_contrasts():
    effect,consistent,minimum,_,_=repeat_contrast(np.array([1.,2.]),np.array([2.,4.]))
    assert np.isclose(effect,1)
    assert not consistent and minimum==0
    effect,consistent,minimum,_,_=repeat_contrast(np.array([1.,1.1]),np.array([2.,2.2]))
    assert consistent and minimum>0.8 and np.isclose(effect,1)

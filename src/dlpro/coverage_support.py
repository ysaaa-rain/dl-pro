"""Observed native-contact support; zero coverage is not assigned a biological cause."""
import numpy as np
from dlpro.experiment_core import LENGTH,centered_indices


def coarse_support(coverage,centers):
    ids=np.arange(len(coverage));width=np.minimum(10,LENGTH-ids*10);result=[]
    for channel,bp in enumerate([100,200]):
        bins=ids//(bp//10);total=(LENGTH+bp-1)//bp;valid=np.bincount(bins,weights=width*(coverage>0),minlength=total);area=np.bincount(bins,weights=width,minlength=total);fraction=valid/area
        starts=np.asarray([centered_indices(int(c))[channel] for c in centers]);indices=(starts[:,None]+np.arange(128)[None])%total;result.append(fraction[indices])
    return np.stack(result,1)

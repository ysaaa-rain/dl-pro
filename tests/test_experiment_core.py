import numpy as np
import pandas as pd
import cooler
from dlpro.experiment_core import ContactReader,DensityInvariantTransform,ShapeTransform,LENGTH


def test_circular_boundary_and_short_bin_area(tmp_path):
    start=np.arange(0,LENGTH,10)
    bins=pd.DataFrame({'chrom':'NC_000913.3','start':start,'end':np.minimum(start+10,LENGTH)})
    last=len(bins)-1
    pixels=pd.DataFrame({'bin1_id':[0,0,last],'bin2_id':[0,last,last],'count':[100,52,27]})
    path=tmp_path/'boundary.cool';cooler.create_cooler(str(path),bins,pixels)
    matrix=ContactReader(path).window(-1,n=2,bin_bp=100,depth_normalize=False)
    expected=np.array([[27/(5.2**2),52/(5.2*10)],[52/(5.2*10),100/100]])
    np.testing.assert_allclose(matrix,expected,rtol=1e-6)


def test_density_nuisance_removed_but_shape_change_retained(tmp_path):
    rng=np.random.default_rng(7);raw=rng.gamma(2,1,(5,2,128,128)).astype('float32');raw=(raw+raw.transpose(0,1,3,2))/2
    transform=DensityInvariantTransform().fit(raw[:3]);before=transform.expected.copy()
    np.testing.assert_allclose(transform.transform(raw),transform.transform(raw*17),atol=1e-5)
    changed=raw.copy();changed[:,:,20:36,70:86]*=8;changed[:,:,70:86,20:36]*=8
    assert np.max(abs(transform.transform(raw)-transform.transform(changed)))>.1
    np.testing.assert_array_equal(before,transform.expected)
    transform.save(tmp_path/'transform.npz');restored=ShapeTransform.load(tmp_path/'transform.npz')
    np.testing.assert_array_equal(restored.transform(raw),transform.transform(raw))

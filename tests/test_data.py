import numpy as np
import pandas as pd
import pytest
import cooler
from dlpro.data import read_window, local_oe, validate_annotations, overlap_groups, inventory, inspect_cool


def test_real_cool_format_and_memory_guard(tmp_path):
    path = str(tmp_path/'toy.cool')
    bins = pd.DataFrame({'chrom':['chr']*4,'start':[0,10,20,30],'end':[10,20,30,40]})
    pixels = pd.DataFrame({'bin1_id':[0,0,1,2,3], 'bin2_id':[0,1,1,2,3], 'count':[5,2,6,7,8]})
    cooler.create_cooler(path,bins,pixels)
    a, valid = read_window(path,'chr',0,20)
    np.testing.assert_array_equal(a, [[5,2],[2,6]])
    assert valid.all() and inspect_cool(path)['binsize'] == 10
    assert len(inventory(tmp_path)[0]['sha256']) == 64
    with pytest.raises(ValueError): read_window(path,'chr',0,40,max_bins=2)
    with pytest.raises(ValueError): read_window(path,'chr',-10,10)


def test_oe_missing_and_zero():
    a = np.array([[2.,0.,np.nan],[0.,4.,0.],[np.nan,0.,6.]])
    result = local_oe(a)
    np.testing.assert_allclose(np.diag(result), [.5,1,1.5])
    assert np.isnan(result[0,1]) and np.isnan(result[0,2])
    with pytest.raises(ValueError): local_oe(np.array([[1,-1],[-1,1]]))


def test_annotations_fail_closed():
    f = pd.DataFrame({'type':['CHIN'],'chrom':['chr'],'start':[0],'end':[20]})
    assert validate_annotations(f,{'chr':40}).center.iloc[0] == 10
    for bad in [f.assign(start=.5),f.assign(end=50),f.assign(type='unknown'),pd.concat([f,f])]:
        with pytest.raises(ValueError): validate_annotations(bad, {'chr':40})


def test_overlap_transitive_and_repeats():
    f = pd.DataFrame({'chrom':['chr']*5,'start':[0,5,14,24,0],'end':[10,15,20,30,10]})
    g = overlap_groups(f)
    assert g[0] == g[1] == g[2] == g[4]
    assert g[3] != g[0]

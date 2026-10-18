import numpy as np
import pytest
from sklearn.metrics import accuracy_score,f1_score
from dlpro.classification_review import metrics,paired_metric_intervals,matched_masks,check_spatial_partition,choose_bias,calibrate_bias,orientation_features

def test_group_interval_and_metrics_preserve_pairs():
    y=np.array([0,0,1,1,2,2]);a=np.array([0,1,1,1,2,0]);g=np.array([0,0,1,1,2,2])
    m=metrics(y,a,3);assert m['accuracy']==accuracy_score(y,a);assert m['macro_f1']==f1_score(y,a,average='macro')
    assert all(r['delta']==r['low']==r['high']==0 for r in paired_metric_intervals(y,a,a,g,3,12,100))

def test_context_overlap_rejected_across_partitions():
    with pytest.raises(ValueError):check_spatial_partition(np.array([0,5]),np.array([10,15]),np.array([0,1]))
    assert check_spatial_partition(np.array([0,5,20]),np.array([10,15,25]),np.array([0,0,1]))

def test_matched_occlusion_keeps_area_distance_and_support():
    n=32;d=abs(np.arange(n)[:,None]-np.arange(n)[None,:]);valid=d>=2;target=np.zeros((n,n),bool);target[12:20,12:20]=True
    roi,control=matched_masks(target,valid,8);assert control is not None and roi.sum()==control.sum();assert not (roi&control).any()
    np.testing.assert_array_equal(np.bincount(d[roi]),np.bincount(d[control]));assert np.array_equal(control,control.T)
    _,bad=matched_masks(valid,valid,8);assert bad is None

def test_calibration_is_label_symmetric_and_probabilities_normalized():
    p=np.array([[.3,.6,.1],[.1,.2,.7],[.6,.3,.1],[.2,.6,.2]]);y=np.array([0,2,0,1]);bias,_=choose_bias(p,y,[-.5,0,.5]);result=calibrate_bias(p,bias);np.testing.assert_allclose(result.sum(1),1)
    permutation=np.array([2,0,1]);inverse=np.argsort(permutation);by,_=choose_bias(p[:,permutation],inverse[y],[-.5,0,.5]);np.testing.assert_allclose(result,calibrate_bias(p[:,permutation],by)[:,inverse])

def test_directional_features_do_not_use_coordinates_and_reverse_consistently():
    rng=np.random.default_rng(5);raw=rng.random((2,1,32,32));raw=(raw+raw.transpose(0,1,3,2))/2;expected=np.ones((1,32));support=np.ones((2,1,32),bool)
    original=orientation_features(raw,expected,support);reverse=orientation_features(raw[:,:,::-1,::-1],expected,support[:,:,::-1]);np.testing.assert_allclose(original,reverse,atol=1e-10)

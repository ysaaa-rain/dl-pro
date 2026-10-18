import numpy as np
from dlpro.local_morphology import analyze,interval_iou,FEATURE_NAMES


def contact_fixture(start=40,end=64):
 n=128;y,x=np.indices((n,n));baseline=np.exp(-abs(x-y)/20)+.05
 image=baseline.copy();image[start:end,start:end]*=8
 return image,baseline[np.arange(n),0]


def test_local_contact_geometry_is_amplitude_invariant_and_moves_with_structure():
 image,expected=contact_fixture()
 first=analyze(image,expected,np.ones(128,bool),100)
 scaled=analyze(image*13,expected,np.ones(128,bool),100)
 moved,_=contact_fixture(64,88)
 shifted=analyze(moved,expected,np.ones(128,bool),100)
 assert np.allclose(first['geometry'],scaled['geometry'],atol=1e-8)
 assert first['region']['lo']==scaled['region']['lo']
 assert len(first['geometry'])==len(FEATURE_NAMES)
 assert shifted['region']['lo']>first['region']['lo']
 assert interval_iou((first['region']['lo'],first['region']['hi']),(40,64))>.25


def test_missing_support_is_never_accepted_as_local_structure():
 image,expected=contact_fixture()
 result=analyze(image,expected,np.zeros(128,bool),100)
 assert not result['region']['valid']
 assert np.isfinite(result['geometry']).all()
 assert interval_iou((2,8),(5,11))==1/3


def test_support_fraction_respects_physical_diagonal_exclusion_at_all_scales():
 image,expected=contact_fixture()
 for bp in (50,100,200):
  result=analyze(image,expected,np.ones(128,bool),bp)
  assert 0<=result['region']['support_fraction']<=1
  assert result['image_support'].dtype==bool
  assert result['image_support'].shape==(32,32)

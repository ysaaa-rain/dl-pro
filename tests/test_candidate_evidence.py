"""Mixed groups and singleton candidates remain eligible for individual analysis."""
import pandas as pd
from dlpro.candidate_evidence import assess_inventory


def test_known_members_and_singletons_never_exclude_individually_supported_candidates():
 base=dict(quality_pass=True,known_buffer_overlap=False,known_full_context_overlap=False,
  both_repeats_beyond_known_distance=True,top3_geometry_difference_directions_agree=3,
  matched_background_count=40,matched_background_localization_tail=.02,
  local_repeat_boundary_iou=.8,minimum_cross_scale_interval_iou=.7)
 frame=pd.DataFrame([
  dict(base,candidate_id='known',cluster=0,known_buffer_overlap=True),
  dict(base,candidate_id='unannotated_in_mixed_group',cluster=0),
  dict(base,candidate_id='singleton',cluster=1),
  dict(base,candidate_id='poor_quality',cluster=2,quality_pass=False),
 ])
 out=assess_inventory(frame).set_index('candidate_id')
 assert len(out)==len(frame)
 assert out.loc['unannotated_in_mixed_group','tentative_new_shape']
 assert out.loc['singleton','tentative_new_shape']
 assert not out.loc['known','tentative_new_shape']
 assert not out.loc['poor_quality','tentative_new_shape']
 assert '基础有效像素' in out.loc['poor_quality','novelty_evidence_assessment']

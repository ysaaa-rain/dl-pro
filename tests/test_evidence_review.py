import numpy as np
import pytest
from dlpro.evidence_review import anonymous_id,context_relation,append_review,block_statistics

def test_identity_stable_and_keyed():
    assert anonymous_id(123,'12'*32)==anonymous_id(123,'12'*32)
    assert anonymous_id(123,'12'*32)!=anonymous_id(123,'34'*32)
    assert '123' not in anonymous_id(123,'12'*32)

def test_geometry_is_not_novelty_label():
    assert context_relation(50,0,100,[40],[60],2)=='center_inside_known_interval'
    assert context_relation(50,0,100,[80],[90],2)=='input_contains_known_interval'
    assert context_relation(50,0,100,[101],[110],2)=='known_context_or_neighbor'
    assert context_relation(50,0,100,[150],[160],2)=='outside_known_context'

def test_review_log_validation_append_and_tamper(tmp_path):
    path=tmp_path/'reviews.jsonl';valid=dict(case_id='BR-x',reviewer='independent-reviewer',assessment='insufficient',evidence='Both views inspected')
    with pytest.raises(ValueError):append_review(path,{**valid,'reviewer':''},{'BR-x'},{'insufficient'},'a'*64)
    append_review(path,valid,{'BR-x'},{'insufficient'},'a'*64)
    append_review(path,{**valid,'reviewer':'second'},{'BR-x'},{'insufficient'},'a'*64)
    path.write_text(path.read_text().replace('Both views','Altered views'),encoding='utf-8')
    with pytest.raises(ValueError):append_review(path,valid,{'BR-x'},{'insufficient'},'a'*64)

def test_stratum_counts_keep_empty_target_errors_and_unique_diagonal():
    p=np.ones((2,4,4));y=np.zeros_like(p);s=np.ones_like(p)
    unique=np.ones_like(p,bool);unique[0]=np.triu(np.ones((4,4),bool))
    metrics=block_statistics(p,y,s,unique)
    np.testing.assert_array_equal(metrics['raw_mse'],[1,1]);assert not metrics['common_positive'].any()
    p[0,3,0]=999
    np.testing.assert_array_equal(block_statistics(p,y,s,unique)['raw_mse'],[1,1])

import pandas as pd
import pytest

from dlpro.coverage_support import validate_candidate_table


def test_candidate_audit_accepts_different_valid_outcomes_and_rejects_wrong_flags():
    protocol = dict(repeat_correlation_min=.5, independent_spacing_bp=25600, min_cluster_members=5)
    for count in (1, 5, 7):
        frame = pd.DataFrame(dict(center=[30000*i for i in range(count)], supported_triangle_fraction=[1.]*count,
                                  wt_oe_correlation_supported=[.8]*count, repeat_supported=[True]*count,
                                  known_full_25_6kb_context_overlap=[False]*count, eligible_unannotated_context=[True]*count,
                                  independent_context_member=[True]*count, repeat2_consensus_near=[True]*count, cluster=[0]*count))
        clusters = pd.DataFrame([dict(cluster=0, total_members=count, independent_members=count, tentative_new_cluster=count >= 5)])
        result = validate_candidate_table(frame, clusters, protocol)
        assert result['independent'] == count
        changed = frame.copy()
        changed.loc[0, 'repeat_supported'] = False
        with pytest.raises(AssertionError):
            validate_candidate_table(changed, clusters, protocol)

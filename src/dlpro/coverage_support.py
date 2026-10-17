"""Observed native-contact support; zero coverage is not assigned a biological cause."""
import numpy as np
from dlpro.experiment_core import LENGTH,centered_indices

# Frozen repeat-QC requirement in candidate_audit_protocol.json: >=50% of the
# off-diagonal upper triangle must have common supported rows in both repeats.
MINIMUM_COMMON_TRIANGLE_FRACTION = .5


def validate_candidate_table(candidate, clusters, protocol):
    """Verify quality/cluster rules without asserting any discovery outcome."""
    supported = (candidate.supported_triangle_fraction.ge(MINIMUM_COMMON_TRIANGLE_FRACTION)
                 & candidate.wt_oe_correlation_supported.ge(protocol["repeat_correlation_min"]))
    assert candidate.repeat_supported.eq(supported).all()
    eligible = supported & ~candidate.known_full_25_6kb_context_overlap
    assert candidate.eligible_unannotated_context.eq(eligible).all()
    accepted = []
    for row in candidate[eligible].sort_values("center").itertuples():
        distance = np.abs(np.asarray(accepted) - row.center)
        if np.all(np.minimum(distance, LENGTH-distance) >= protocol["independent_spacing_bp"]):
            accepted.append(row.center)
    assert candidate.independent_context_member.eq(candidate.center.isin(accepted)).all()
    for row in clusters.itertuples():
        part = candidate[candidate.cluster == row.cluster]
        members = part[part.independent_context_member]
        expected = (row.cluster >= 0 and len(members) >= protocol["min_cluster_members"]
                    and members.repeat2_consensus_near.all()
                    and not part.known_full_25_6kb_context_overlap.any())
        assert row.total_members == len(part) and row.independent_members == len(members)
        assert row.tentative_new_cluster == bool(expected)
    return dict(independent=len(accepted), tentative_new_clusters=int(clusters.tentative_new_cluster.sum()))


def coarse_support(coverage,centers):
    ids=np.arange(len(coverage));width=np.minimum(10,LENGTH-ids*10);result=[]
    for channel,bp in enumerate([100,200]):
        bins=ids//(bp//10);total=(LENGTH+bp-1)//bp;valid=np.bincount(bins,weights=width*(coverage>0),minlength=total);area=np.bincount(bins,weights=width,minlength=total);fraction=valid/area
        starts=np.asarray([centered_indices(int(c))[channel] for c in centers]);indices=(starts[:,None]+np.arange(128)[None])%total;result.append(fraction[indices])
    return np.stack(result,1)

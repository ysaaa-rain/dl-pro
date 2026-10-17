import numpy as np

from dlpro.novelty import empirical_rank, grouped_auc, shape_vectors
from dlpro.refinement import morphology_features


def test_shape_vectors_cancel_amplitude_and_preserve_axis_symmetry():
    raw = np.random.default_rng(4).random((3, 2, 32, 32)).astype('float32')
    raw = (raw + raw.transpose(0, 1, 3, 2)) / 2
    support = np.ones((3, 2, 32), bool)
    expected = np.ones((2, 32))
    a, nuisance = shape_vectors(raw, support, expected, 8)
    b, changed = shape_vectors(raw * 10, support, expected, 8)
    assert np.allclose(a, b, atol=1e-6)
    assert not np.allclose(nuisance[:, 0], changed[:, 0])
    field = a.reshape(3, 2, 8, 8)
    assert np.allclose(field, field.transpose(0, 1, 3, 2))


def test_rank_and_group_bootstrap_compute_actual_values_without_expected_outcomes():
    assert empirical_rank(np.array([-1, 1, 4]), np.array([0, 1, 2])).tolist() == [0, .5, .75]
    result = grouped_auc([True, True, False, False], [.9, .2, .8, .1], [0, 1, 2, 3], 4, 50)
    assert result['auc'] == .75
    assert result['low'] <= result['auc'] <= result['high']


def test_batch_subset_optimization_does_not_change_any_used_shape_feature():
    raw = np.random.default_rng(2).random((6, 2, 32, 32)).astype('float32')
    support = np.ones((6, 2, 32), bool)
    expected = np.ones((2, 32))
    keep = np.array([1, 4])
    assert np.allclose(morphology_features(raw, expected, support)[keep],
                       morphology_features(raw[keep], expected, support[keep]))
    assert np.allclose(shape_vectors(raw, support, expected, 8)[0][keep],
                       shape_vectors(raw[keep], support[keep], expected, 8)[0])

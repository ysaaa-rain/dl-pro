import numpy as np
from dlpro.ridge_features import directional_features


def test_directional_shapes_are_finite_amplitude_invariant_and_axis_consistent():
    x, y = np.indices((128, 128))
    raw = (1 + 6 * np.exp(-((x + y - 127) / 3) ** 2))[None, None]
    support = np.ones((1, 1, 128), bool)
    expected = np.ones((1, 128))
    features = directional_features(raw, expected, support)
    assert features.shape == (1, 20)
    assert np.isfinite(features).all()
    assert np.allclose(features, directional_features(raw * 7, expected, support), atol=1e-6)
    # Swapping the two genomic axes preserves both diagonal directions and
    # swaps the horizontal/vertical curvature summaries.
    transposed = directional_features(raw.transpose(0, 1, 3, 2), expected, support)
    assert np.allclose(features, transposed, atol=1e-6)
    assert features[0, 0] > features[0, 1]
    zeros = directional_features(np.zeros_like(raw), expected, support)
    assert np.array_equal(zeros, np.zeros_like(zeros))

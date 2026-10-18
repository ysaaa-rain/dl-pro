"""Directional curvature of supported, amplitude-normalized contact shapes."""

import numpy as np
from scipy.ndimage import gaussian_filter


def directional_features(raw, expected, valid_rows):
    n = raw.shape[-1]
    distance = abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    valid = ((distance[None, None] >= 2)
             & valid_rows[:, :, :, None] & valid_rows[:, :, None, :]
             & (raw.sum(-1) > 0)[:, :, :, None]
             & (raw.sum(-1) > 0)[:, :, None, :])
    oe = raw / expected[:, distance][None]
    amplitude = (oe * valid).sum((-2, -1)) / np.maximum(valid.sum((-2, -1)), 1)
    field = np.where(valid, np.log1p(oe / np.maximum(amplitude[:, :, None, None], 1e-12)), 0)
    center = np.zeros((n, n), bool)
    center[n // 4:3 * n // 4, n // 4:3 * n // 4] = True
    features = []
    for channel in range(raw.shape[1]):
        mask = valid[:, channel].astype("float32")
        for sigma in (1, 2, 4, 8):
            support = gaussian_filter(mask, (0, sigma, sigma))
            smooth = gaussian_filter(field[:, channel], (0, sigma, sigma)) / np.maximum(support, 1e-6)
            gx, gy = np.gradient(smooth, axis=(1, 2))
            xx = np.gradient(gx, axis=1)
            yy = np.gradient(gy, axis=2)
            xy = (np.gradient(gx, axis=2) + np.gradient(gy, axis=1)) / 2
            trace = xx + yy
            discriminant = np.sqrt((xx - yy) ** 2 + 4 * xy ** 2)
            small = (trace - discriminant) / 2
            large = (trace + discriminant) / 2
            responses = (
                -(xx + 2 * xy + yy) / 2,
                -(xx - 2 * xy + yy) / 2,
                -xx, -yy,
                np.maximum(-small, 0) * np.exp(-abs(large) / np.maximum(abs(small), 1e-6)),
            )
            eligible = valid[:, channel] & center & (support >= .9)
            for response in responses:
                response = np.maximum(response * sigma ** 2, 0)
                features.append(np.asarray([
                    np.quantile(value[keep], .98) if keep.any() else 0
                    for value, keep in zip(response, eligible)
                ]))
    result = np.stack(features, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite directional morphology")
    return result

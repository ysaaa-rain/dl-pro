"""Conditional anomaly and shallow cross-repeat novelty representations."""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def shape_vectors(raw, support, expected, pooled_bins):
    n = raw.shape[-1]
    delta = abs(np.arange(n)[:, None] - np.arange(n)[None])
    valid = ((delta[None, None] >= 2) & support[:, :, :, None] & support[:, :, None, :])
    oe = raw / expected[:, delta][None]
    amplitude = (oe * valid).sum((-2, -1)) / valid.sum((-2, -1)).clip(1)
    field = np.where(valid, np.log1p(oe / amplitude[:, :, None, None].clip(1e-8)), 0)
    f = n // pooled_bins
    if n % pooled_bins:
        raise ValueError("Pooling must divide the frozen shape input")
    def pool(x):
        return x.reshape(len(x), raw.shape[1], pooled_bins, f, pooled_bins, f).mean((3, 5))
    vectors = (pool(field) / pool(valid).clip(1e-6)).reshape(len(raw), -1)
    nuisance = []
    for channel in range(raw.shape[1]):
        a, m = raw[:, channel], valid[:, channel]
        mean = (a * m).sum((-2, -1)) / m.sum((-2, -1)).clip(1)
        zeros = ((a == 0) & m).sum((-2, -1)) / m.sum((-2, -1)).clip(1)
        nuisance.extend([np.log(mean.clip(1e-12)), zeros, support[:, channel].mean(-1),
                         (a * (m & (delta[None] >= n // 4))).sum((-2, -1)) / (a * m).sum((-2, -1)).clip(1e-12)])
    return vectors, np.stack(nuisance, axis=1)


def reference_centroids(vectors, indices, groups):
    return np.stack([np.median(vectors[indices[groups[indices] == g]], axis=0)
                     for g in np.unique(groups[indices])])


def nearest(vectors, references):
    return cdist(vectors, references).min(1) / np.sqrt(vectors.shape[1])


def empirical_rank(values, calibration):
    calibration = np.asarray(calibration)
    if not len(calibration):
        raise ValueError("An empty calibration set cannot define a novelty rank")
    return np.searchsorted(np.sort(calibration), values, side="right") / (len(calibration) + 1)


def fit_routes(vectors, nuisance, train_bg, references, val_bg, val_refs, groups, cfg):
    """No labels accepted; all training/calibration IDs are supplied and recorded."""
    scaler = StandardScaler().fit(nuisance[1][train_bg])
    polynomial = PolynomialFeatures(degree=cfg["nuisance_degree"], include_bias=False)
    design = {r: polynomial.fit_transform(scaler.transform(nuisance[r])) for r in (1, 2)}
    conditional = Ridge(alpha=cfg["ridge_alpha"]).fit(design[1][train_bg], vectors[1][train_bg])
    residual = {r: vectors[r] - conditional.predict(design[r]) for r in (1, 2)}
    center = np.median(residual[1][train_bg], axis=0)
    spread = np.maximum(1.4826 * np.median(abs(residual[1][train_bg] - center), axis=0), cfg["scale_floor"])
    z = {r: (residual[r] - center) / spread for r in (1, 2)}
    training = np.unique(np.r_[train_bg, references])
    components = min(cfg["pca_components"], len(training) - 1, z[1].shape[1])
    pca = PCA(n_components=components, svd_solver="full").fit(z[1][training])
    latent = {r: pca.transform(z[r]) for r in (1, 2)}
    latent_scale = np.maximum(latent[1][training].std(0), cfg["scale_floor"])
    latent = {r: latent[r] / latent_scale for r in (1, 2)}
    denoiser = Ridge(alpha=cfg["ridge_alpha"]).fit(latent[1][training], latent[2][training])
    paired = {r: denoiser.predict(latent[r]) for r in (1, 2)}
    feature_sets = dict(conditional=z, pca=latent, paired=paired)
    scores = {r: {} for r in (1, 2)}
    diagnostics = dict(pca_components=components, representation_training_indices=training.tolist(),
                       background_training_indices=train_bg.tolist(), reference_indices=references.tolist(),
                       calibration_background_indices=val_bg.tolist(), calibration_reference_indices=val_refs.tolist(),
                       reference_groups=len(np.unique(groups[references])),
                       paired_training_mse=float(np.mean((paired[1][training] - latent[2][training]) ** 2)))
    for name, embedding in feature_sets.items():
        ref = reference_centroids(embedding[1], references, groups)
        if len(val_refs):
            calibration_novelty = nearest(embedding[1][val_refs], ref)
        elif len(ref) > 1:
            distances = cdist(ref, ref)
            np.fill_diagonal(distances, np.inf)
            calibration_novelty = distances.min(1) / np.sqrt(ref.shape[1])
        else:
            raise ValueError("Neither held-out allowed references nor multiple independent reference groups")
        background = reference_centroids(embedding[1], train_bg, groups)
        for r in (1, 2):
            anomaly = np.sqrt(np.mean(embedding[r] ** 2, axis=1)) if name == "conditional" else nearest(embedding[r], background)
            if r == 1:
                calibration_anomaly = anomaly[val_bg]
            anomaly_rank = empirical_rank(anomaly, calibration_anomaly)
            novelty_rank = empirical_rank(nearest(embedding[r], ref), calibration_novelty)
            scores[r][name + "_novelty"] = np.minimum(anomaly_rank, novelty_rank)
            if name == "conditional":
                scores[r]["conditional_anomaly"] = anomaly_rank
    per_scale = []
    width = cfg["pooled_bins"] ** 2
    for channel in range(len(cfg["input_scales_bp"])):
        zscale = {r: z[r][:, channel * width:(channel + 1) * width] for r in (1, 2)}
        ref = reference_centroids(zscale[1], references, groups)
        if len(val_refs):
            cal = nearest(zscale[1][val_refs], ref)
        else:
            distance = cdist(ref, ref)
            np.fill_diagonal(distance, np.inf)
            cal = distance.min(1) / np.sqrt(width)
        anomaly_cal = np.sqrt((zscale[1][val_bg] ** 2).mean(1))
        for r in (1, 2):
            per_scale.append(np.minimum(empirical_rank(np.sqrt((zscale[r] ** 2).mean(1)), anomaly_cal),
                                        empirical_rank(nearest(zscale[r], ref), cal)))
    joint = np.min(np.stack([*per_scale, scores[1]["paired_novelty"], scores[2]["paired_novelty"]]), axis=0)
    for r in (1, 2):
        scores[r]["joint_multiscale"] = joint
    return scores, diagnostics


def grouped_auc(labels, scores, groups, seed, draws):
    labels, scores, groups = np.asarray(labels, bool), np.asarray(scores), np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    positive, negative = np.flatnonzero(labels), np.flatnonzero(~labels)
    if not len(positive) or not len(negative):
        return dict(auc=None, low=None, high=None, bootstrap_valid=0)
    wins = np.zeros((len(unique), len(unique)))
    comparison = (scores[positive, None] > scores[negative]).astype(float) + .5 * (scores[positive, None] == scores[negative])
    np.add.at(wins, (inverse[positive, None], inverse[negative][None]), comparison)
    pos = np.bincount(inverse[positive], minlength=len(unique))
    neg = np.bincount(inverse[negative], minlength=len(unique))
    weights = np.random.default_rng(seed).multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=draws)
    numerator = np.einsum("bi,ij,bj->b", weights, wins, weights)
    denominator = (weights @ pos) * (weights @ neg)
    estimates = numerator[denominator > 0] / denominator[denominator > 0]
    low, high = np.quantile(estimates, [.025, .975])
    return dict(auc=float(comparison.mean()), low=float(low), high=float(high), bootstrap_valid=len(estimates))

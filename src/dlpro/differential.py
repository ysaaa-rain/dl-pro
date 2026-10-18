"""Distance-matched local contact enrichment and unpaired repeat contrasts."""
import numpy as np


def local_enrichment(matrix, centers, start, end, bin_bp, row_correct=False):
    a = np.asarray(matrix, dtype=float)
    if row_correct:
        marginal = a.mean(axis=1)
        denominator = np.sqrt(np.outer(marginal, marginal))
        a = np.divide(a, denominator, out=np.zeros_like(a), where=denominator > 0)
    core = (centers >= start) & (centers < end)
    left, right = centers < start, centers >= end
    signal = background = weight = 0.0
    for d in range(max(1, int(np.ceil(200 / bin_bp))), min(len(a), int((end-start)//bin_bp)+1)):
        values = np.diag(a, d)
        target = core[:-d] & core[d:]
        flank = (left[:-d] & left[d:]) | (right[:-d] & right[d:])
        if target.sum() and flank.sum():
            count = int(target.sum())
            signal += values[target].sum()
            background += values[flank].mean() * count
            weight += count
    if not weight or signal <= 0 or background <= 0:
        return np.nan, np.nan, int(weight)
    return signal / background, signal / weight, int(weight)


def repeat_contrast(wild, mutant):
    wild, mutant = np.log2(wild), np.log2(mutant)
    cross = (mutant[:, None] - wild[None, :]).ravel()
    effect = float(mutant.mean() - wild.mean())
    return effect, bool(np.all(cross > 0) or np.all(cross < 0)), float(np.min(np.abs(cross))), float(abs(wild[1]-wild[0])), float(abs(mutant[1]-mutant[0]))

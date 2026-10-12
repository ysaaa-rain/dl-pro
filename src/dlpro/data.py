"""Read local windows only; never materialize the genome-wide dense matrix."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import cooler


def inventory(root):
    rows = []
    for p in sorted(Path(root).rglob('*')):
        if not p.is_file():
            continue
        digest = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        size = p.stat().st_size
        rows.append(dict(path=str(p), bytes=size, sha256=digest.hexdigest(),
                         git_size_status='review-large-file' if size >= 50*1024**2 else 'small'))
    return rows


def inspect_cool(uri):
    c = cooler.Cooler(uri)
    return dict(uri=uri, binsize=int(c.binsize) if c.binsize is not None else None, chromosomes={k:int(v) for k,v in c.chromsizes.items()},
                nnz=int(c.info['nnz']), nbins=int(c.info['nbins']),
                has_weights='weight' in c.bins().columns,
                storage_mode=c.storage_mode)


def read_window(uri, chrom, start, end, balance=False, max_bins=2048):
    c = cooler.Cooler(uri)
    if chrom not in c.chromsizes or not 0 <= start < end <= c.chromsizes[chrom]:
        raise ValueError('Invalid region; origin-crossing windows must be split explicitly')
    left, right = c.extent((chrom, start, end))
    if right-left > max_bins:
        raise ValueError('Window exceeds dense allocation limit; coarsen or reduce region')
    matrix = c.matrix(balance=balance).fetch((chrom, start, end)).astype(float)
    return matrix, np.isfinite(matrix)


def local_oe(matrix):
    """Exploratory local O/E; NOT genome-wide expected or ICE balancing.

    Missing values remain missing; zero-expected diagonals are excluded.
    Final repeat comparisons must share a predeclared normalization protocol.
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.allclose(a, a.T, equal_nan=True):
        raise ValueError('Expected symmetric square matrix')
    if np.isinf(a).any() or np.any(a[np.isfinite(a)] < 0):
        raise ValueError('Counts must be nonnegative and finite or NaN')
    result = np.full_like(a, np.nan)
    for d in range(len(a)):
        i = np.arange(len(a)-d)
        values = a[i, i+d]
        valid = np.isfinite(values)
        expected = values[valid].mean() if valid.any() else 0
        if expected > 0:
            result[i, i+d] = values / expected
            result[i+d, i] = values / expected
    return result


def validate_annotations(frame, chromsizes):
    required = {'type', 'chrom', 'start', 'end'}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError('Need nonempty type,chrom,start,end annotations')
    f = frame.copy()
    if f[list(required)].isna().any().any() or not set(f.type) <= {'CHIN','OPCID','CHID'}:
        raise ValueError('Missing annotation fields or unknown structure type')
    for col in ['start', 'end']:
        vals = pd.to_numeric(f[col], errors='raise')
        if not np.isfinite(vals).all() or not (vals == np.floor(vals)).all():
            raise ValueError('Coordinates must be finite integers')
        f[col] = vals.astype('int64')
    for row in f.itertuples():
        if row.chrom not in chromsizes or not 0 <= row.start < row.end <= chromsizes[row.chrom]:
            raise ValueError('Annotation outside reference bounds')
    if f.duplicated(['type','chrom','start','end']).any():
        raise ValueError('Duplicate annotations')
    f['center'] = (f.start + f.end) // 2
    f['length_bp'] = f.end - f.start
    return f


def overlap_groups(windows):
    """Group connected overlapping input windows; all repeats inherit this group.

    Call on actual extracted window extents, not merely annotation extents.
    Half-open touching intervals do not overlap. Circular seam needs a separate
    grouping step before circular windows can be supported.
    """
    groups = pd.Series(index=windows.index, dtype='int64')
    group = -1
    for _, rows in windows.groupby('chrom', sort=True):
        right = -1
        for idx, row in rows.sort_values(['start','end']).iterrows():
            if row.start >= row.end or row.start < 0:
                raise ValueError('Invalid window')
            if row.start >= right:
                group += 1
                right = row.end
            else:
                right = max(right, row.end)
            groups.loc[idx] = group
    return groups.astype('int64')

"""Seeded percentile intervals; paired differences preserve observation pairing."""
import numpy as np


def bootstrap_ci(values, *, n_boot=2000, seed=42, alpha=0.05):
    values=np.asarray(values,dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected a nonempty finite one-dimensional sample')
    if n_boot < 1 or not 0 < alpha < 1:
        raise ValueError('Invalid bootstrap count or alpha')
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(values,len(values),replace=True).mean() for _ in range(n_boot)])
    lo,hi=np.quantile(means,[alpha/2,1-alpha/2])
    return float(values.mean()),float(lo),float(hi)


def paired_bootstrap_diff(a, b, *, n_boot=2000, seed=42, alpha=0.05):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    if a.shape != b.shape:
        raise ValueError('Paired samples must have identical lengths')
    return bootstrap_ci(a-b,n_boot=n_boot,seed=seed,alpha=alpha)

"""Shared synthetic spatial AnnData builders for offline pytest."""

from __future__ import annotations

import numpy as np
import scanpy as sc


def synthetic_domain_adata(n=1500, g=40, k=5, seed=0, ambig_frac=0.0):
    """Poisson marker-program spatial data: ``k`` domains, optional ambiguous blends.

    Used by ``test_pipeline.py`` (clean domains) and ``test_mapping.py`` (with ambiguous
    cells for confidence calibration). Returns ``(AnnData, k)``.
    """
    rng = np.random.default_rng(seed)
    mpr = g // k
    marker_of = {j: np.arange(j * mpr, (j + 1) * mpr) for j in range(k)}
    dom = rng.integers(0, k, n)
    lib = rng.lognormal(0.0, 0.3, n)

    ambig = np.zeros(n, dtype=bool)
    if ambig_frac > 0:
        n_ambig = int(ambig_frac * n)
        ambig[rng.choice(n, n_ambig, replace=False)] = True

    rate = np.full((n, g), 0.3)
    for c in range(n):
        rate[c, marker_of[dom[c]]] += rng.uniform(6, 14)
        if ambig[c]:
            other = (dom[c] + rng.integers(1, k)) % k
            rate[c, marker_of[other]] += rng.uniform(4, 9)
    X = rng.poisson(rate * lib[:, None]).astype("float32")

    ad = sc.AnnData(X)
    ad.var_names = [f"g{j:02d}" for j in range(g)]
    ad.obsm["spatial"] = rng.uniform(0, 40, size=(n, 2))
    if ambig_frac > 0:
        ad.obs["cell_type"] = [f"type{d}" for d in dom]
        ad.obs["cell_type"] = ad.obs["cell_type"].astype("category")
        ad.obs["is_ambiguous"] = ambig
    return ad, k

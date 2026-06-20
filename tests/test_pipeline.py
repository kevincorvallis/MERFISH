"""Tests for the MERFISH analysis pipeline.

- Offline tests run the real Scanpy pipeline on a small synthetic dataset (fast).
- Tests marked ``live`` download a REAL public MERFISH dataset over the network.

    pytest tests/                 # everything (live tests hit the network)
    pytest tests/ -m "not live"   # offline only
"""

import pytest

import live_test
from synthetic_fixtures import synthetic_domain_adata


def test_pipeline_runs_and_recovers_structure():
    ad, k = synthetic_domain_adata()
    m = live_test.run_pipeline(ad)
    assert m["n_clusters"] >= 2, "pipeline should find multiple clusters"
    assert m["n_clusters"] <= 2 * k, "should not wildly over-cluster clean data"
    assert "X_umap" in ad.obsm, "UMAP embedding should be computed"
    assert m["has_spatial"], "spatial coords should be preserved"


def test_pipeline_is_deterministic():
    a1, _ = synthetic_domain_adata(seed=1)
    a2, _ = synthetic_domain_adata(seed=1)
    assert live_test.run_pipeline(a1)["n_clusters"] == \
        live_test.run_pipeline(a2)["n_clusters"]


@pytest.mark.live
def test_real_merfish_pipeline():
    """Download real Moffitt 2018 hypothalamus MERFISH data and run the pipeline."""
    ad = live_test.load_dataset("merfish")
    m = live_test.run_pipeline(ad)
    assert m["n_cells"] > 1000, "real MERFISH dataset should have many cells"
    assert m["n_genes"] >= 100, "MERFISH panel should have ~100+ genes"
    assert m["n_clusters"] >= 3, "real tissue should yield several cell populations"
    assert m["has_spatial"], "MERFISH data must carry spatial coordinates"
    # unsupervised clustering should agree with the published cell types well
    # above chance (ARI ~ 0 for random labels)
    assert m["ari_vs_published"] > 0.2, (
        f"Leiden clusters should recover published cell types "
        f"(ARI={m['ari_vs_published']})")

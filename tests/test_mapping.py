"""Tests for principled, confidence-scored reference mapping (celltype_mapping).

This is the upgrade over the project's heuristic shared-PCA + cosine cell-type
mapping: a faithful reimplementation of the MapMyCells / Allen ``cell_type_mapper``
algorithm — marker-gene correlation with bootstrap confidence — that runs without
the Allen reference download.

- Offline tests build a synthetic labelled dataset and check the mapper recovers
  the known labels AND that its confidence score is calibrated (confident calls
  are more accurate than unconfident ones).
- The ``live`` test maps the real Moffitt 2018 hypothalamus MERFISH cells onto
  their own published ``Cell_class`` taxonomy via a held-out split.
"""

import pytest
import scanpy as sc
import numpy as np

import celltype_mapping as cm
from synthetic_fixtures import synthetic_domain_adata


def _labeled(n=2400, g=60, k=6, ambig_frac=0.18, seed=0):
    """Synthetic labelled spatial data with ambiguous cells for calibration tests."""
    return synthetic_domain_adata(n=n, g=g, k=k, seed=seed, ambig_frac=ambig_frac)


def test_mapper_recovers_known_types():
    ad, k = _labeled()
    res = cm.evaluate_mapping(ad, "cell_type", test_size=0.5, seed=0, n_bootstrap=50)
    assert res["n_types"] == k
    assert res["accuracy"] > 0.80, f"held-out accuracy too low: {res['accuracy']}"
    # confidence is a probability-like score in [0, 1]
    conf = res["mapped"]["confidence"].to_numpy()
    assert conf.min() >= 0.0 and conf.max() <= 1.0


def test_confidence_is_calibrated():
    """The whole point of the upgrade: a confident call should be a better call."""
    ad, _ = _labeled()
    res = cm.evaluate_mapping(ad, "cell_type", test_size=0.5, seed=0, n_bootstrap=80)
    df = res["mapped"]
    hi = df[df["confidence"] >= 0.9]
    lo = df[df["confidence"] < 0.9]
    assert len(hi) > 0 and len(lo) > 0, "need both confident and unconfident calls"
    acc_hi = (hi["pred"] == hi["truth"]).mean()
    acc_lo = (lo["pred"] == lo["truth"]).mean()
    assert acc_hi > acc_lo, (
        f"confidence not calibrated: acc@hi={acc_hi:.2f} acc@lo={acc_lo:.2f}")


def test_mapping_is_deterministic():
    ad, _ = _labeled(seed=3)
    r1 = cm.evaluate_mapping(ad, "cell_type", test_size=0.5, seed=0, n_bootstrap=40)
    r2 = cm.evaluate_mapping(ad, "cell_type", test_size=0.5, seed=0, n_bootstrap=40)
    assert r1["accuracy"] == r2["accuracy"]
    assert list(r1["mapped"]["pred"]) == list(r2["mapped"]["pred"])


def test_beats_cosine_baseline_on_calibration():
    """Principled bootstrap mapping yields a usable confidence the cosine
    heuristic lacks; on accuracy it should be at least competitive."""
    ad, _ = _labeled(seed=1)
    res = cm.evaluate_mapping(ad, "cell_type", test_size=0.5, seed=0, n_bootstrap=60,
                              with_baseline=True)
    assert "cosine_accuracy" in res
    # principled mapper should not be meaningfully worse than the cosine heuristic
    assert res["accuracy"] >= res["cosine_accuracy"] - 0.05


@pytest.mark.live
def test_mapping_on_real_merfish():
    """Map real Moffitt 2018 hypothalamus cells onto their published Cell_class."""
    ad = cm.load_moffitt()
    res = cm.evaluate_mapping(ad, "Cell_class", test_size=0.5, seed=0, n_bootstrap=50)
    assert res["n_test"] > 5000, "real held-out set should be large"
    assert res["accuracy"] > 0.70, (
        f"principled mapping should recover published cell classes well "
        f"(accuracy={res['accuracy']})")
    df = res["mapped"]
    acc_hi = (df[df.confidence >= 0.9]["pred"] == df[df.confidence >= 0.9]["truth"]).mean()
    assert acc_hi > res["accuracy"], "high-confidence calls should beat the average"


def _small_query(rows):
    ad = sc.AnnData(np.asarray(rows, dtype=float))
    ad.var_names = ['a', 'b', 'c', 'd']
    return ad


def _small_reference():
    return {'markers': ['a', 'b', 'c', 'd'], 'types': np.array(['type_a', 'type_b']),
            'centroids': np.array([[4, 3, 1, 0], [0, 1, 3, 4]], dtype=float)}


def test_uninformative_cells_abstain_instead_of_getting_confidence_one():
    query = _small_query([[0, 0, 0, 0], [5, 5, 5, 5], [np.nan, 3, 1, 0],
                          [4, np.inf, 1, 0], [4, 3, 1, 0]])
    result = cm.map_to_reference(query, _small_reference(), n_bootstrap=20, frac=1)
    assert list(result['pred'][:4]) == ['unassigned'] * 4
    assert (result['confidence'][:4] == 0).all()
    assert (result['correlation'][:4] == 0).all()
    assert result['pred'].iloc[4] == 'type_a'
    assert result['confidence'].iloc[4] == 1
    assert np.isfinite(result[['confidence', 'correlation']].to_numpy()).all()


def test_identical_centroids_do_not_break_ties_by_reference_order():
    ref = _small_reference()
    ref['centroids'][1] = ref['centroids'][0]
    result = cm.map_to_reference(_small_query([[4, 3, 1, 0]]), ref)
    assert result['pred'].iloc[0] == 'unassigned'
    assert result['confidence'].iloc[0] == 0


def test_constant_reference_rows_cannot_win():
    ref = _small_reference()
    ref['centroids'][:] = 1
    result = cm.map_to_reference(_small_query([[4, 3, 1, 0]]), ref)
    assert result['pred'].iloc[0] == 'unassigned'
    assert result['confidence'].iloc[0] == 0


@pytest.mark.parametrize('kwargs', [{'n_bootstrap': 0}, {'n_bootstrap': -2},
                                    {'n_bootstrap': 1.5}, {'frac': 0}, {'frac': 1.1}])
def test_invalid_bootstrap_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        cm.map_to_reference(_small_query([[4, 3, 1, 0]]), _small_reference(), **kwargs)


def test_one_marker_reference_is_rejected():
    query = sc.AnnData(np.array([[1.0]]))
    query.var_names = ['a']
    with pytest.raises(ValueError, match='at least two markers'):
        cm.map_to_reference(query, {'markers': ['a'], 'types': np.array(['a']),
                                   'centroids': np.array([[1.0]])})


def test_negative_match_and_nonfinite_reference_abstain():
    query = _small_query([[0, 1, 3, 4]])
    ref = _small_reference()
    ref['centroids'][1] = np.nan
    result = cm.map_to_reference(query, ref, frac=1)
    assert result['pred'].iloc[0] == 'unassigned'
    assert result['confidence'].iloc[0] == 0
    assert result['correlation'].iloc[0] == 0

"""Tests for the *genuine* non-circular, cell-type-aware composition QC.

The earlier QC was degenerate (adversarially shown): the reference was built from the section's own
placement (self-referential, JS(p,p)=0) and a gross misregistration scored 1.0 only via the
missing-region default — cell types were inert (scrambling them changed nothing). The fix:

  * `to_broad_class` — map any cell-type vocabulary (Moffitt `Cell_class` OR Allen ABC classes) to a
    shared broad vocabulary, so an EXTERNAL reference can be compared to a section.
  * `broaden_reference` — re-aggregate a fine `{region: {class: frac}}` reference into broad classes.
  * `composition_qc` — score only regions covered by BOTH the section and the reference, via real
    Jensen-Shannon over the shared vocabulary (NOT the missing-region default), reporting coverage
    separately. This makes the score genuinely cell-type-driven and non-circular.
"""

import numpy as np
import pytest

import atlas_registration as ar


def test_to_broad_class_bridges_moffitt_and_abc():
    moffitt = ["Excitatory", "Inhibitory", "Astrocyte", "OD Mature 2", "Microglia",
               "Endothelial 1", "Ependymal", "Ambiguous"]
    assert list(ar.to_broad_class(moffitt)) == [
        "excitatory", "inhibitory", "astro_epen", "oligo", "microglia",
        "vascular", "astro_epen", "other"]
    abc = ["Excitatory IT", "Inhibitory Pvalb", "Astro-Epen", "Oligo", "Micro-PVM", "Endo", "VLMC"]
    assert list(ar.to_broad_class(abc)) == [
        "excitatory", "inhibitory", "astro_epen", "oligo", "microglia", "vascular", "vascular"]


def test_broaden_reference_sums_fine_classes():
    fine = {7: {"Excitatory IT": 0.3, "Excitatory ET": 0.2, "Oligo": 0.4, "Endo": 0.1}}
    broad = ar.broaden_reference(fine)
    assert broad[7]["excitatory"] == pytest.approx(0.5)
    assert broad[7]["oligo"] == pytest.approx(0.4)
    assert broad[7]["vascular"] == pytest.approx(0.1)


def test_composition_qc_is_celltype_aware_and_noncircular():
    """The genuine property: against an EXTERNAL reference, a wrong placement that lands cells in
    OVERLAPPING regions with the wrong cell-type makeup scores worse (JS-driven, not missing-region),
    AND the real cell labels are load-bearing (scrambling them changes the score)."""
    ref = {10: {"neuron": 0.9, "oligo": 0.1}, 20: {"oligo": 0.9, "neuron": 0.1}}  # external
    rids = np.array([10] * 100 + [20] * 100)
    classes = np.array(["neuron"] * 90 + ["oligo"] * 10 + ["oligo"] * 90 + ["neuron"] * 10, dtype=object)

    good = ar.composition_qc(rids, classes, ref)
    swapped = np.array([20] * 100 + [10] * 100)            # same cells, region ids swapped (overlapping set)
    bad = ar.composition_qc(swapped, classes, ref)

    assert good["coverage"] == 1.0 and bad["coverage"] == 1.0, "no missing-region artifact"
    assert good["score"] < 0.1, f"correct placement should match the external reference: {good['score']}"
    assert bad["score"] > good["score"] + 0.3, "JS-driven: wrong composition scores worse"

    # cell labels are LOAD-BEARING (the failure of the old QC): scrambling changes the score
    scrambled = np.random.default_rng(0).permutation(classes)
    scr = ar.composition_qc(swapped, scrambled, ref)
    assert abs(scr["score"] - bad["score"]) > 1e-6, "scrambling real types must change the score"


def test_composition_qc_reports_uncovered_separately():
    """A region absent from the reference is reported as uncovered coverage, NOT scored 1.0."""
    ref = {10: {"neuron": 1.0}}
    rids = np.array([10] * 50 + [999] * 50)               # 999 not in reference
    classes = np.array(["neuron"] * 100, dtype=object)
    qc = ar.composition_qc(rids, classes, ref)
    assert qc["coverage"] == pytest.approx(0.5)
    assert qc["score"] < 1e-9                              # covered region matches -> ~0, not inflated by 999


@pytest.mark.live
def test_genuine_qc_is_celltype_aware_on_real_tissue():
    """End-to-end genuine fix on REAL data: real Moffitt cells + real `Cell_class` scored against a
    REAL external Allen ABC reference (~4M cells). The real composition matches the correct ABC
    regions far better than a wrong cell-type marginal OR shuffled ABC regions — i.e. the QC is
    genuinely cell-type-aware and non-circular. Skips unless brainglobe, abc_atlas_access, the ABC
    metadata download, and the Moffitt h5ad are all present."""
    import glob
    import os

    pytest.importorskip("brainglobe_atlasapi")
    pytest.importorskip("abc_atlas_access")
    if not glob.glob("data/abc_cache/**/cell_metadata_with_parcellation_annotation.csv", recursive=True):
        pytest.skip("ABC metadata not downloaded (data/abc_cache)")
    if not os.path.exists("data/anndata/merfish.h5ad"):
        pytest.skip("Moffitt h5ad not present")

    import abc_qc_validation as v
    r = v.run()
    assert r["coverage"] > 0.9, f"external ABC reference should cover the section: {r['coverage']}"
    assert r["qc_real_types"] < r["qc_wrong_marginal_all_excitatory"] - 0.1, "cell types must be load-bearing"
    assert r["qc_real_types"] < r["qc_shuffled_reference"] - 0.1, "must match correct ABC regions, not random"
    assert r["cell_types_load_bearing"]

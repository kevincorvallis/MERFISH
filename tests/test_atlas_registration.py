"""Tests for atlas registration: 2D MERFISH section -> Allen CCFv3 + per-cell region labels.

The pipeline (see ``docs/atlas-registration-2026.md``) goes:
  DeepSlice affine anchor -> STalign/ANTs deformable warp -> per-cell annotation lift-over
  -> calibrated uncertainty -> QC cross-check vs the ABC reference.

The heavy registration engines (DeepSlice/STalign/ANTs) and the real CCFv3 download are
stubbed/optional (mirroring ``celltype_mapping.map_with_cell_type_mapper``). What is tested
here is the *dependency-light core* that must be correct regardless of backend, on a
synthetic labelled atlas:

  * the anchored-plane geometry (incl. oblique cut angles),
  * per-cell label transfer (annotation lookup through the registration),
  * the calibrated per-cell confidence (registration-ensemble bootstrap) — confident calls
    are more accurate, and cells near region boundaries are less confident,
  * the QC cross-check — cell-type composition divergence *rises under misregistration*.

This is the same testing philosophy as ``test_mapping.py``: synthetic ground truth, and we
assert *behaviour* (recovery + calibration + QC sensitivity), not implementation details.
"""

import numpy as np
import pytest

import atlas_registration as ar


# --- synthetic ground truth --------------------------------------------------

def _atlas(shape=(40, 80, 80)):
    """A small synthetic CCF: nested, contiguous, labelled regions with real boundaries."""
    return ar.synthetic_ccf(shape)


def _cells(seed=0, n=4000, tilt=0.12):
    """Sample cells from an obliquely-cut coronal plane through the synthetic atlas.

    Returns the ground-truth plane, the cells (in-plane uv coords), their true region
    labels, and a region-dependent 'cell type' so the QC composition cross-check has signal.
    """
    annotation, ontology = _atlas()
    plane = ar.coronal_plane(annotation.shape, tilt=tilt)
    cells = ar.sample_section_cells(annotation, plane, n=n, seed=seed)
    return annotation, ontology, plane, cells


# --- geometry ----------------------------------------------------------------

def test_anchored_plane_maps_corners_including_oblique():
    """O/U/V anchoring must place the four image corners at O, O+U, O+V, O+U+V — and
    represent an oblique plane (the DeepSlice convention) where x varies across the section."""
    O = np.array([20.0, 0.0, 0.0])
    U = np.array([3.0, 79.0, 0.0])   # spans Y, tilts in X -> oblique
    V = np.array([2.0, 0.0, 79.0])   # spans Z, tilts in X -> oblique
    plane = ar.AnchoredPlane(O, U, V)

    corners = plane.to_ccf(np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]]))
    assert np.allclose(corners[0], O)
    assert np.allclose(corners[1], O + U)
    assert np.allclose(corners[2], O + V)
    assert np.allclose(corners[3], O + U + V)
    assert np.allclose(corners[4], O + 0.5 * U + 0.5 * V)
    # genuinely oblique: the plane's x-coordinate is not constant across the section
    assert np.ptp(corners[:, 0]) > 1.0


# --- per-cell label transfer -------------------------------------------------

def test_label_transfer_recovers_planted_regions():
    """Mapping cells through the *correct* registration and looking up the annotation
    volume must recover the region each cell was sampled from (catches axis-order/rounding
    bugs that silently corrupt every downstream label)."""
    annotation, ontology, plane, cells = _cells()
    reg = ar.PlaneRegistration(plane)
    out = ar.assign_regions(cells["uv"], reg, annotation, ontology)

    assert list(out.index) == list(range(len(cells["uv"])))
    acc = (out["region_id"].to_numpy() == cells["truth_region"]).mean()
    assert acc > 0.99, f"label transfer should be near-exact under the true plane: {acc:.3f}"
    # acronyms come from the ontology, not raw ids
    assert set(out["acronym"]) <= set(ontology.values())


def test_assignment_handles_out_of_bounds_as_background():
    """Cells whose registration lands outside the atlas must resolve to background, not crash."""
    annotation, ontology = _atlas()
    # a plane translated far outside the volume
    far = ar.AnchoredPlane(np.array([1e4, 0, 0]),
                           np.array([0, 79.0, 0]), np.array([0, 0, 79.0]))
    out = ar.assign_regions(np.array([[0.5, 0.5], [0.1, 0.9]]),
                            ar.PlaneRegistration(far), annotation, ontology)
    assert (out["region_id"] == 0).all()


# --- calibrated uncertainty --------------------------------------------------

def test_confidence_is_calibrated_under_registration_error():
    """The novel layer: under a realistic (slightly wrong) registration, the
    registration-ensemble bootstrap confidence must be *calibrated* — cells we get right
    should carry higher confidence than cells we get wrong."""
    annotation, ontology, true_plane, cells = _cells()
    # the *estimated* registration is the truth plus a small, fixed anchoring error
    est = true_plane.perturb(np.random.default_rng(7), angle_deg=2.5, shift=2.5)
    reg = ar.PlaneRegistration(est)

    conf = ar.region_confidence(cells["uv"], reg, annotation,
                                n_perturb=64, angle_deg=2.5, shift=2.5, seed=0)
    assert conf["confidence"].between(0.0, 1.0).all()

    correct = conf["region_id"].to_numpy() == cells["truth_region"]
    assert correct.sum() > 0 and (~correct).sum() > 0, "need both right and wrong calls"
    assert conf.loc[correct, "confidence"].mean() > conf.loc[~correct, "confidence"].mean()


def test_cells_near_region_boundaries_are_less_confident():
    """Boundary ambiguity must show up as lower confidence — a cell deep inside a region is
    more certain than one straddling a border."""
    annotation, ontology, true_plane, cells = _cells()
    reg = ar.PlaneRegistration(true_plane.perturb(np.random.default_rng(1),
                                                  angle_deg=2.0, shift=2.0))
    conf = ar.region_confidence(cells["uv"], reg, annotation,
                                n_perturb=48, angle_deg=2.0, shift=2.0, seed=0)
    near = conf["dist_to_boundary"] <= 1.5
    interior = conf["dist_to_boundary"] >= 5.0
    assert near.sum() > 20 and interior.sum() > 20
    assert conf.loc[near, "confidence"].mean() < conf.loc[interior, "confidence"].mean()


def test_region_confidence_is_deterministic():
    annotation, ontology, true_plane, cells = _cells()
    reg = ar.PlaneRegistration(true_plane.perturb(np.random.default_rng(2), shift=2.0))
    a = ar.region_confidence(cells["uv"], reg, annotation, n_perturb=32, seed=0)
    b = ar.region_confidence(cells["uv"], reg, annotation, n_perturb=32, seed=0)
    assert np.array_equal(a["region_id"].to_numpy(), b["region_id"].to_numpy())
    assert np.allclose(a["confidence"].to_numpy(), b["confidence"].to_numpy())


# --- QC cross-check ----------------------------------------------------------

def test_jensen_shannon_bounds():
    assert ar.jensen_shannon([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0, abs=1e-9)
    assert ar.jensen_shannon([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0, abs=1e-9)
    # symmetric and normalization-invariant
    assert ar.jensen_shannon([2, 0], [0, 5]) == pytest.approx(1.0, abs=1e-9)


def test_qc_composition_detects_misregistration():
    """The QC cross-check: a section's per-region cell-type composition should match the ABC
    reference under a good registration and *diverge* under a bad one. We stand in the ABC
    reference with the true-assignment composition; a badly-warped registration must score
    measurably worse."""
    annotation, ontology, true_plane, cells = _cells()
    reference = ar.region_composition(cells["truth_region"], cells["cell_type"])

    good = ar.assign_regions(cells["uv"], ar.PlaneRegistration(true_plane),
                             annotation, ontology)
    bad_plane = true_plane.perturb(np.random.default_rng(3), angle_deg=30.0, shift=18.0)
    bad = ar.assign_regions(cells["uv"], ar.PlaneRegistration(bad_plane),
                            annotation, ontology)

    qc_good = ar.section_qc(good["region_id"], cells["cell_type"], reference)
    qc_bad = ar.section_qc(bad["region_id"], cells["cell_type"], reference)

    assert qc_good["score"] < 0.05, f"good registration should match reference: {qc_good['score']:.3f}"
    assert qc_bad["score"] > qc_good["score"] + 0.1, (
        f"QC must flag misregistration: good={qc_good['score']:.3f} bad={qc_bad['score']:.3f}")


# --- backend seams (heavy engines are optional, like map_with_cell_type_mapper) ----

@pytest.mark.parametrize("fn", ["deepslice_anchor", "ants_register"])
def test_heavy_backends_are_stubbed_with_install_hint(fn):
    with pytest.raises(NotImplementedError):
        getattr(ar, fn)()


# --- live: real Allen CCFv3 via brainglobe (optional, networked) --------------

@pytest.mark.live
def test_stalign_recovers_known_ccf_slice():
    """The molecular-aware deformable backend on real data: STalign LDDMM (2D→3D) recovers a
    *known* coronal slice's position in the real CCFv3 — after fitting, the per-cell AP
    coordinate matches the true plane within ~2 voxels — and exposes the standard Registration
    interface (transform_points / perturb) so the rest of the core runs on it unchanged."""
    pytest.importorskip("torch")
    pytest.importorskip("STalign")
    pytest.importorskip("brainglobe_atlasapi")
    from brainglobe_atlasapi import BrainGlobeAtlas

    bg = BrainGlobeAtlas("allen_mouse_100um")
    ref = np.asarray(bg.reference, dtype=float)
    res = np.array(bg.resolution, dtype=float)
    nx = ref.shape
    ap = nx[0] // 2  # the known true coronal plane
    xA = [np.arange(n) * d - (n - 1) * d / 2.0 for n, d in zip(nx, res)]
    sl = ref[ap]
    ys, xs = np.where(sl > sl.mean() * 0.5)
    sel = np.random.default_rng(0).choice(len(xs), 1500, replace=False)
    cells_xy = np.stack([xA[2][xs[sel]], xA[1][ys[sel]]], axis=1)  # (x, y) microns

    reg = ar.stalign_register(ref, res, cells_xy, niter=30, a=200.0)
    assert hasattr(reg, "transform_points") and hasattr(reg, "perturb")
    vox = reg.transform_points(cells_xy)
    assert vox.shape == (len(cells_xy), 3)
    assert abs(np.median(vox[:, 0]) - ap) < 2.0, f"AP recovery off: {np.median(vox[:, 0]):.1f} vs {ap}"
    inb = ((vox >= 0).all(1) & (vox[:, 0] < nx[0]) & (vox[:, 1] < nx[1]) & (vox[:, 2] < nx[2]))
    assert inb.mean() > 0.95


@pytest.mark.live
def test_real_ccfv3_loads_via_brainglobe():
    """Smoke test that the real Allen CCFv3 annotation + ontology load through the same
    interface the synthetic atlas uses. Skipped unless brainglobe-atlasapi is installed."""
    pytest.importorskip("brainglobe_atlasapi")
    annotation, ontology = ar.load_ccf_brainglobe("allen_mouse_100um")
    assert annotation.ndim == 3 and annotation.size > 0
    assert len(ontology) > 100 and 0 in ontology  # CCFv3 has ~670 structures + background


@pytest.mark.live
def test_real_ccfv3_coarsening_recovers_calibrates_and_qc():
    """The whole core, on the *real* Allen CCFv3 with ground truth: at a sensible ontology
    depth, label transfer is exact under the true plane, the ensemble confidence is calibrated
    (high-confidence calls beat the average), and the QC cross-check flags misregistration."""
    pytest.importorskip("brainglobe_atlasapi")
    ann, ont = ar.load_ccf_brainglobe("allen_mouse_100um", depth=3)
    assert 20 < len(ont) < 200, "depth-3 roll-up should give a few-dozen major structures"

    plane = ar.coronal_plane(ann.shape)
    cells = ar.sample_section_cells(ann, plane, seed=0)
    true_assign = ar.assign_regions(cells["uv"], ar.PlaneRegistration(plane), ann, ont)
    assert (true_assign["region_id"].to_numpy() == cells["truth_region"]).mean() > 0.99

    reg = ar.PlaneRegistration(plane.perturb(np.random.default_rng(7), angle_deg=2.5, shift=2.5))
    conf = ar.region_confidence(cells["uv"], reg, ann, n_perturb=48,
                                angle_deg=2.5, shift=2.5, seed=0)
    ok = conf["region_id"].to_numpy() == cells["truth_region"]
    hi = (conf["confidence"] >= 0.9).to_numpy()
    assert hi.sum() > 20 and ok[hi].mean() > ok.mean(), "high-confidence calls must beat average"

    reference = ar.region_composition(cells["truth_region"], cells["cell_type"])
    bad_plane = plane.perturb(np.random.default_rng(3), angle_deg=30.0, shift=18.0)
    bad = ar.assign_regions(cells["uv"], ar.PlaneRegistration(bad_plane), ann, ont)
    good_qc = ar.section_qc(true_assign["region_id"], cells["cell_type"], reference)["score"]
    bad_qc = ar.section_qc(bad["region_id"], cells["cell_type"], reference)["score"]
    assert bad_qc > good_qc + 0.1

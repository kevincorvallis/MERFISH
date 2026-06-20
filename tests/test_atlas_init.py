"""Tests for the affine-anchoring stage — a training-free coarse AP search that finds where a
2D section sits along the atlas's anterior-posterior axis by image similarity.

This is the lightweight stand-in for DeepSlice's learned AP/angle estimate: it gives the
deformable backend (STalign LDDMM) a correct starting plane, fixing the init-sensitivity that
made posterior sections misconverge. Tested offline on a synthetic graded volume where every
coronal slice is distinct, so the search has a unique correct answer.
"""

import numpy as np
import pytest

import atlas_registration as ar


def _graded_volume(shape=(30, 40, 40)):
    """A grayscale volume whose content shifts along AP (axis 0) — each coronal slice is
    distinguishable, so a section taken at one AP matches exactly one plane."""
    d, h, w = shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    vol = np.zeros(shape, dtype=float)
    for ap in range(d):
        cy = 8 + 0.8 * ap          # blob center marches along y with AP
        sigma = 5 + 0.1 * ap
        vol[ap] = np.exp(-(((yy - cy) ** 2 + (xx - w / 2) ** 2) / (2 * sigma ** 2)))
    return vol


def test_coarse_ap_search_finds_known_plane():
    """A section cut from a known AP must be matched back to that plane (within 1 voxel)."""
    vol = _graded_volume()
    res = np.array([1.0, 1.0, 1.0])
    nx = vol.shape
    xA = [np.arange(n) * d - (n - 1) * d / 2.0 for n, d in zip(nx, res)]
    true_ap = 15
    section_img = vol[true_ap]                       # the section, on the atlas (y, x) grid
    best, ncc = ar.coarse_ap_search(vol, res, section_img, xA[2], xA[1])
    assert ncc.shape == (nx[0],)
    assert abs(best - true_ap) <= 1, f"AP search found {best}, expected ~{true_ap}"


def test_coarse_ap_search_peaks_uniquely():
    """The similarity curve should peak sharply at the true plane, not be flat/ambiguous."""
    vol = _graded_volume()
    res = np.array([1.0, 1.0, 1.0])
    nx = vol.shape
    xA = [np.arange(n) * d - (n - 1) * d / 2.0 for n, d in zip(nx, res)]
    true_ap = 20
    _, ncc = ar.coarse_ap_search(vol, res, vol[true_ap], xA[2], xA[1])
    # the true plane scores higher than planes a few slices away
    assert ncc[true_ap] > ncc[true_ap - 5] and ncc[true_ap] > ncc[true_ap + 5]


def test_coarse_ap_search_handles_partial_section():
    """A section covering only part of the field (zeros elsewhere) should still localize."""
    vol = _graded_volume()
    res = np.array([1.0, 1.0, 1.0])
    nx = vol.shape
    xA = [np.arange(n) * d - (n - 1) * d / 2.0 for n, d in zip(nx, res)]
    true_ap = 18
    sec = vol[true_ap].copy()
    sec[:, : sec.shape[1] // 3] = 0.0                # crop a third of the field
    best, _ = ar.coarse_ap_search(vol, res, sec, xA[2], xA[1])
    assert abs(best - true_ap) <= 2


@pytest.mark.live
def test_stalign_autoinit_recovers_posterior_scan():
    """The improvement, end to end on real data: a *posterior* thalamic section (AP 80) that
    misconverged from STalign's default init (AP error ~1.4 mm) is recovered to within ~3 voxels
    once ``init='auto'`` anchors the AP plane first via ``coarse_ap_search``."""
    pytest.importorskip("torch")
    pytest.importorskip("STalign")
    pytest.importorskip("tornado")
    pytest.importorskip("brainglobe_atlasapi")
    from brainglobe_atlasapi import BrainGlobeAtlas

    bg = BrainGlobeAtlas("allen_mouse_100um")
    ref = np.asarray(bg.reference, dtype=float)
    res = np.array(bg.resolution, dtype=float)
    nx = ref.shape
    ap = 80  # posterior thalamus — the case that failed from default init
    xA = [np.arange(n) * d - (n - 1) * d / 2.0 for n, d in zip(nx, res)]
    sl = ref[ap]
    ys, xs = np.where(sl > sl.mean() * 0.5)
    sel = np.random.default_rng(2).choice(len(xs), 2500, replace=False)
    cells = np.column_stack([xA[2][xs[sel]], xA[1][ys[sel]]])  # (x, y) microns

    # auto-init anchors the AP from the start, so few iterations suffice for the AP assertion
    reg = ar.stalign_register(ref, res, cells, niter=25, a=200.0)  # init='auto' (default)
    vox = reg.transform_points(cells)
    assert abs(np.median(vox[:, 0]) - ap) < 3.0, \
        f"auto-init AP recovery off: {np.median(vox[:, 0]):.1f} vs {ap}"

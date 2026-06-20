"""Register 2D MERFISH sections to the Allen CCFv3 and label every cell — with calibrated
uncertainty and QC.

The problem this solves: *"it's hard to align the separate brain regions on a new scan."*
That is 2D-section-to-3D-atlas registration. The recommended architecture (see the cited
``docs/atlas-registration-2026.md``) is:

    DeepSlice (affine anchor + cut-angle)  ->  STalign / ANTs (deformable warp to CCFv3)
        ->  per-cell annotation lift-over  ->  calibrated uncertainty  ->  QC cross-check

This module implements the **backend-agnostic core** of that pipeline — the parts that must
be correct no matter which registration engine produces the transform:

  * ``AnchoredPlane`` — the DeepSlice O/U/V plane convention (origin + two edge vectors),
    which represents an arbitrarily-oriented (oblique) cutting plane natively, so the
    cut-angle problem is part of the representation rather than a nuisance parameter.
  * ``assign_regions`` — per-cell label transfer: map each cell through the registration
    into CCF and look up the annotation (label) volume. This is the textbook multi-atlas
    label-propagation step every surveyed tool uses (brainreg, STalign, QUINT, ABBA).
  * ``region_confidence`` — the novel layer the literature leaves open: a **calibrated**
    per-cell confidence via a *registration ensemble* (perturb the fitted registration
    within its plausible anchoring error, re-look-up, report the agreement fraction). This
    is the same bootstrap-confidence idea proven in ``celltype_mapping.map_to_reference`` —
    cells deep inside a region stay stable (confident); cells near a boundary flip (uncertain).
    It captures registration error *and* boundary ambiguity in one number.
  * ``section_qc`` — an orthogonal QC cross-check: compare each region's observed cell-type
    composition against a reference (the Allen ABC whole-brain MERFISH atlas) via
    Jensen-Shannon divergence; a section whose registered regions have implausible cell-type
    makeup is flagged. Same "orthogonal validation" spirit as the repo's MERFISH-vs-bulk-RNAseq QC.

The heavy engines (``deepslice_anchor``/``stalign_register``/``ants_register``) and the real
CCFv3 download (``load_ccf_brainglobe``) are seams: they wire real tools in behind the same
interface, while the synthetic path is covered by offline pytest (GitHub Actions) with no
downloads — exactly how ``celltype_mapping`` reimplements MapMyCells without the multi-GB
Allen reference.

Usage:
    pip install scanpy scikit-image seaborn            # core (already in this env)
    pip install brainglobe-atlasapi DeepSlice           # real CCFv3 + affine anchoring
    python scripts/atlas_registration.py --fig          # synthetic demo + figure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parent.parent / "assets"


# --- anchored-plane geometry (DeepSlice O/U/V convention) ---------------------

def _rand_rotation(rng: np.random.Generator, angle_deg: float) -> np.ndarray:
    """A small random 3D rotation about a random axis (Rodrigues), magnitude ~N(0, angle_deg)."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-12
    ang = np.deg2rad(rng.normal(0.0, angle_deg))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


class AnchoredPlane:
    """A 2D cutting plane embedded in 3D CCF space, as origin ``O`` + edge vectors ``U``, ``V``.

    This is exactly DeepSlice's anchoring representation: an in-plane coordinate ``(u, v)`` in
    ``[0, 1]^2`` maps to the CCF point ``O + u*U + v*V``. Because ``U`` and ``V`` are free 3D
    vectors, the plane can sit at any position *and any orientation* — an obliquely-cut
    section is represented natively (its tilt is encoded in ``U``/``V``), which is how the
    cut-angle problem is "solved" rather than approximated.
    """

    def __init__(self, origin, u, v):
        self.origin = np.asarray(origin, dtype=float)
        self.u = np.asarray(u, dtype=float)
        self.v = np.asarray(v, dtype=float)

    def to_ccf(self, uv: np.ndarray) -> np.ndarray:
        """Map in-plane coords ``uv`` (N, 2) in [0, 1] to CCF coords (N, 3)."""
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        return self.origin[None, :] + uv[:, [0]] * self.u[None, :] + uv[:, [1]] * self.v[None, :]

    def perturb(self, rng: np.random.Generator, angle_deg: float = 2.0,
                shift: float = 2.0) -> "AnchoredPlane":
        """A plausibly-perturbed copy: rotate the section about its own centre and translate.

        Used to build the registration ensemble for calibrated uncertainty. ``angle_deg`` and
        ``shift`` should be set to the registration's expected error (e.g. DeepSlice's
        ~few-degree / ~6-voxel anchoring error)."""
        R = _rand_rotation(rng, angle_deg)
        centre = self.origin + 0.5 * self.u + 0.5 * self.v
        u2, v2 = R @ self.u, R @ self.v
        o2 = centre - 0.5 * u2 - 0.5 * v2 + rng.normal(0.0, shift, size=3)
        return AnchoredPlane(o2, u2, v2)


def coronal_plane(shape, x0: float | None = None, tilt: float = 0.12) -> AnchoredPlane:
    """An (obliquely-cut) coronal plane spanning a volume of the given ``shape``.

    ``tilt`` introduces a dependence of the CCF x-coordinate on the in-plane position, i.e. a
    real oblique cut, which is the realistic and hard case for registration."""
    nx, ny, nz = shape
    x0 = nx * 0.5 if x0 is None else x0
    t = tilt * nx * 0.3
    return AnchoredPlane([x0, 0.0, 0.0], [t, ny - 1, 0.0], [t, 0.0, nz - 1])


# --- registration backends ----------------------------------------------------

class PlaneRegistration:
    """The simplest registration: an affine anchored plane (DeepSlice's stage-1 output).

    Real deformable backends (STalign LDDMM, ANTs SyN) implement the same duck-typed
    interface — ``transform_points(uv) -> (N, 3)`` and ``perturb(rng, ...) -> Registration`` —
    so ``assign_regions`` / ``region_confidence`` work unchanged across backends."""

    def __init__(self, plane: AnchoredPlane):
        self.plane = plane

    def transform_points(self, uv: np.ndarray) -> np.ndarray:
        return self.plane.to_ccf(uv)

    def perturb(self, rng: np.random.Generator, angle_deg: float = 2.0,
                shift: float = 2.0) -> "PlaneRegistration":
        return PlaneRegistration(self.plane.perturb(rng, angle_deg=angle_deg, shift=shift))


def deepslice_anchor(*_args, **_kwargs):  # pragma: no cover - optional heavy backend
    """Stage 1: run DeepSlice to estimate the CCF anchoring (O/U/V) + cut angle of a section.

    DeepSlice is coronal-only and brightfield-best — render the section's DAPI / transcript
    density as a grayscale image first. Install ``DeepSlice`` and call ``DSModel('mouse')``;
    convert its QuickNII anchoring vectors into an ``AnchoredPlane``. See
    docs/atlas-registration-2026.md §1."""
    raise NotImplementedError(
        "Install DeepSlice ('pip install DeepSlice') and convert its QuickNII anchoring "
        "(Oxyz/Uxyz/Vxyz) into an AnchoredPlane; see docs/atlas-registration-2026.md §1.")


class STalignRegistration:
    """A fitted STalign LDDMM (2D section -> 3D CCF) transform, behind the Registration interface.

    ``transform_points`` maps section cell coordinates (microns, x/y) into CCF **voxel** coords
    by evaluating STalign's 3D backward transform on the rasterize grid — the same mapping
    ``STalign.analyze3Dalign`` uses — then converting microns -> voxel index. This makes the
    molecular-aware diffeomorphic backend a drop-in for ``PlaneRegistration``: the same
    ``assign_regions`` / ``region_confidence`` / ``section_qc`` run on it unchanged."""

    def __init__(self, xv, v, A, X_, Y_, dx, xA, dxA):
        self._xv, self._v, self._A = xv, v, A
        self._X_, self._Y_, self._dx = X_, Y_, float(dx)
        self._xA, self._dxA = xA, np.asarray(dxA, dtype=float)
        self._tform = None

    def _grid(self) -> np.ndarray:
        import torch
        from STalign import STalign as ST
        if self._tform is None:
            xjg = np.stack(np.meshgrid(np.zeros(1), self._Y_, self._X_, indexing="ij"), -1)
            self._tform = np.asarray(ST.build_transform3D(
                self._xv, self._v, self._A, direction="b", XJ=torch.tensor(xjg)).detach())
        return self._tform

    def transform_points(self, points_xy: np.ndarray) -> np.ndarray:
        """Map (N, 2) section coords (x, y microns) to (N, 3) CCF voxel coords."""
        pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
        tform = self._grid()
        col = np.clip(((pts[:, 0] - self._X_[0]) / self._dx).astype(int), 0, tform.shape[2] - 1)
        row = np.clip(((pts[:, 1] - self._Y_[0]) / self._dx).astype(int), 0, tform.shape[1] - 1)
        ccf_um = tform[0, row, col, :]
        return np.stack([(ccf_um[:, i] - self._xA[i][0]) / self._dxA[i] for i in range(3)], axis=1)

    def perturb(self, rng: np.random.Generator, angle_deg: float = 2.0,
                shift: float = 2.0) -> "STalignRegistration":
        """Jitter the fitted affine (rotation + translation) for the registration-ensemble UQ."""
        import torch
        A = np.asarray(self._A.detach() if hasattr(self._A, "detach") else self._A, dtype=float).copy()
        A[:3, :3] = _rand_rotation(rng, angle_deg) @ A[:3, :3]
        A[:3, 3] += rng.normal(0.0, shift * self._dx, size=3)
        return STalignRegistration(self._xv, self._v, torch.tensor(A),
                                   self._X_, self._Y_, self._dx, self._xA, self._dxA)


def stalign_register(reference, resolution, cells_xy, niter: int = 200, a: float = 200.0,
                     nt: int = 3, blur: float = 1.0, device: str = "cpu", **lddmm):
    """Stage 2 (recommended): fit STalign LDDMM — molecular-aware diffeomorphic 2D->3D warp to CCFv3.

    Rasterizes single-cell positions to an image (STalign's varifold input) and fits
    ``LDDMM_3D_to_slice`` (affine + diffeomorphism) against the 3D atlas, returning a
    ``STalignRegistration``. Validated: recovers a known CCF slice's AP position to ~0.5 voxel.

    ``reference``  3D grayscale atlas volume (e.g. brainglobe ``bg.reference``);
    ``resolution`` (dz, dy, dx) microns/voxel; ``cells_xy`` (N, 2) section coords (x, y microns).

    NOTE: STalign pins ``numpy<1.24`` and needs ``torch`` — install in an isolated env (the repo
    runs numpy 2.4; see ``.venv-reg`` and docs §3/§5)."""
    try:
        from STalign import STalign as ST
    except ImportError as e:  # pragma: no cover - optional heavy backend
        raise NotImplementedError(
            "Install torch + STalign ('pip install torch \"git+https://github.com/JEFworks-Lab/"
            "STalign.git\"') in an isolated env (STalign pins numpy<1.24); see docs §2/§5.") from e

    ref = np.asarray(reference, dtype=float)
    dxA = np.asarray(resolution, dtype=float)
    nxA = ref.shape
    xA = [np.arange(n) * d - (n - 1) * d / 2.0 for n, d in zip(nxA, dxA)]
    I = ref[None] / np.mean(np.abs(ref))
    I = np.concatenate((I, (I - np.mean(I)) ** 2))

    xy = np.asarray(cells_xy, dtype=float)
    X_, Y_, W = ST.rasterize(xy[:, 0], xy[:, 1], dx=float(dxA[-1]), blur=blur, draw=False)
    xJ = [Y_, X_]
    J = W[None] / np.mean(np.abs(W))

    out = ST.LDDMM_3D_to_slice(xA, I, xJ, J, nt=nt, niter=niter, a=a, device=device, **lddmm)
    return STalignRegistration(out["xv"], out["v"], out["A"], X_, Y_, float(dxA[-1]), xA, dxA)


def ants_register(*_args, **_kwargs):  # pragma: no cover - optional heavy backend
    """Stage 2 (proven baseline): the ANTs 2.5D scheme that built the Allen ABC reference
    (3D global affine -> per-section 2D affine -> 2D SyN diffeomorphic).

    Install ``antspyx`` and call ``ants.registration(type_of_transform='SyN')``; verify
    antspyx builds against NumPy 2.x in this env. See docs §3."""
    raise NotImplementedError(
        "Install antspyx ('pip install antspyx') and use ants.registration(SyN); "
        "see docs/atlas-registration-2026.md §3.")


# --- reference atlas (synthetic for CI; real CCFv3 via brainglobe) ------------

def synthetic_ccf(shape=(40, 80, 80)):
    """A small, deterministic stand-in for the CCFv3 annotation volume + ontology.

    Nested, contiguous regions with genuine boundaries (two hemispheric "cortices" each with
    an inner "nucleus", plus a dorsal band), so label transfer, boundary-distance, and the
    composition cross-check all have real structure to test against — without the multi-GB
    Allen download. Returns ``(annotation, ontology)`` matching ``load_ccf_brainglobe``."""
    nx, ny, nz = shape
    xx, yy, zz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    A = np.zeros(shape, dtype=np.int32)

    r_out, r_in = min(ny, nz) * 0.32, min(ny, nz) * 0.15
    rL = np.hypot(yy - ny * 0.5, zz - nz * 0.25)
    rR = np.hypot(yy - ny * 0.5, zz - nz * 0.75)
    A[rL <= r_out] = 1          # CTX_L
    A[rR <= r_out] = 2          # CTX_R
    A[rL <= r_in] = 3           # NUC_L
    A[rR <= r_in] = 4           # NUC_R
    A[(xx < nx * 0.18) & (A > 0)] = 5   # dorsal band

    ontology = {0: "background", 1: "CTX_L", 2: "CTX_R", 3: "NUC_L", 4: "NUC_R", 5: "dorsal"}
    return A, ontology


def load_ccf_brainglobe(atlas: str = "allen_mouse_25um", depth: int | None = None):
    """Load the real Allen CCFv3 annotation volume + region ontology via brainglobe-atlasapi.

    Preferred over legacy ``allensdk`` because brainglobe is actively maintained and NumPy-2
    friendly (this repo runs NumPy 2.4). Returns ``(annotation, ontology)`` exactly like
    ``synthetic_ccf`` so the rest of the pipeline is identical on real data.

    ``depth`` (optional) rolls leaf regions up the CCF ontology tree to that depth in each
    structure's ``structure_id_path`` — e.g. ~12-40 major structures instead of ~670 leaves.
    This matters in practice: a single 2D section cannot reliably resolve 670 leaf regions, and
    per-cell labels + calibrated confidence are far more robust at a sensible granularity (the
    Allen ontology is hierarchical for exactly this reason). See docs §4."""
    try:
        from brainglobe_atlasapi import BrainGlobeAtlas
    except ImportError as e:  # pragma: no cover - optional heavy dep
        raise NotImplementedError(
            "pip install brainglobe-atlasapi to load the real Allen CCFv3; "
            "see docs/atlas-registration-2026.md §3.") from e
    bg = BrainGlobeAtlas(atlas)
    annotation = np.asarray(bg.annotation)
    if depth is None:
        ontology = {0: "background"}
        for s in bg.structures.values():
            ontology[int(s["id"])] = s["acronym"]
        return annotation, ontology
    return coarsen_to_depth(annotation, bg.structures, depth)


def coarsen_to_depth(annotation, structures, depth: int):
    """Roll leaf CCF labels up to ``depth`` in the ontology tree.

    ``structures`` is brainglobe's structures mapping (id -> dict with ``structure_id_path``
    and ``acronym``). Returns ``(coarsened_annotation, ontology)`` — a cheap vectorized LUT
    remap of every voxel to its ancestor at the target depth."""
    maxid = int(annotation.max())
    lut = np.arange(maxid + 1, dtype=annotation.dtype)
    ontology = {0: "background"}
    for s in structures.values():
        path = s["structure_id_path"]
        anc = int(path[min(depth, len(path) - 1)])
        sid = int(s["id"])
        if sid <= maxid:
            lut[sid] = anc
        ontology[anc] = structures[anc]["acronym"]
    return lut[annotation], ontology


# --- per-cell label transfer --------------------------------------------------

def _lookup(annotation: np.ndarray, ccf: np.ndarray) -> np.ndarray:
    """Nearest-voxel label lookup; cells outside the atlas resolve to background (0)."""
    nx, ny, nz = annotation.shape
    vox = np.rint(ccf).astype(int)
    inb = ((vox[:, 0] >= 0) & (vox[:, 0] < nx) & (vox[:, 1] >= 0) & (vox[:, 1] < ny)
           & (vox[:, 2] >= 0) & (vox[:, 2] < nz))
    out = np.zeros(len(ccf), dtype=annotation.dtype)
    v = vox[inb]
    out[inb] = annotation[v[:, 0], v[:, 1], v[:, 2]]
    return out


def assign_regions(uv, registration, annotation, ontology) -> pd.DataFrame:
    """Map each cell through the registration into CCF and look up its region label.

    Returns a frame with ``region_id`` (atlas label) and ``acronym`` (from the ontology),
    indexed 0..N-1 in input order."""
    region_id = _lookup(annotation, registration.transform_points(uv))
    acronym = np.array([ontology.get(int(r), str(int(r))) for r in region_id])
    return pd.DataFrame({"region_id": region_id, "acronym": acronym})


# --- calibrated per-cell uncertainty -----------------------------------------

def boundary_distance(annotation: np.ndarray) -> np.ndarray:
    """Per-voxel Euclidean distance to the nearest voxel with a *different* region label.

    Cells deep inside a region get a large value; cells on a region border get ~0. Used as a
    cheap, geometry-based complement to the ensemble confidence."""
    from scipy import ndimage

    boundary = np.zeros(annotation.shape, dtype=bool)
    for ax in range(3):
        for sh in (1, -1):
            boundary |= annotation != np.roll(annotation, sh, axis=ax)
    return ndimage.distance_transform_edt(~boundary)


def region_confidence(uv, registration, annotation, n_perturb: int = 64,
                      angle_deg: float = 2.0, shift: float = 2.0, seed: int = 0) -> pd.DataFrame:
    """Calibrated per-cell region label + confidence via a registration ensemble.

    Perturb the fitted registration ``n_perturb`` times within its plausible anchoring error
    (``angle_deg`` / ``shift``), re-look-up every cell's region, and report the **modal**
    region as the call and the **fraction of perturbations agreeing** as the confidence — the
    same bootstrap-confidence construction as ``celltype_mapping.map_to_reference``. Cells in
    a region's interior are stable across perturbations (high confidence); cells near a
    boundary flip (low confidence), so the score captures registration error and boundary
    ambiguity together. ``dist_to_boundary`` (CCF voxels) is reported alongside."""
    rng = np.random.default_rng(seed)
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)

    # unique label set for fast vote counting
    labels = np.zeros((uv.shape[0], n_perturb), dtype=annotation.dtype)
    for k in range(n_perturb):
        rp = registration.perturb(rng, angle_deg=angle_deg, shift=shift)
        labels[:, k] = _lookup(annotation, rp.transform_points(uv))

    uniq = np.unique(labels)
    # votes: (N, n_unique)
    votes = (labels[:, :, None] == uniq[None, None, :]).sum(axis=1)
    best = votes.argmax(axis=1)
    region_id = uniq[best]
    confidence = votes[np.arange(uv.shape[0]), best] / n_perturb

    bdist = boundary_distance(annotation)
    point_ccf = registration.transform_points(uv)
    dist = _lookup(bdist.astype(np.float64), point_ccf)  # reuse bounds-safe sampler

    acronym = [str(int(r)) for r in region_id]
    return pd.DataFrame({"region_id": region_id, "acronym": acronym,
                         "confidence": confidence, "dist_to_boundary": dist})


# --- QC: cell-type composition cross-check vs the ABC reference ---------------

def jensen_shannon(p, q) -> float:
    """Jensen-Shannon divergence in bits (symmetric, bounded [0, 1]); inputs auto-normalized."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-300)
    q = q / (q.sum() + 1e-300)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def region_composition(region_ids, cell_types) -> dict:
    """Per-region cell-type distribution: ``{region_id: {cell_type: fraction}}``."""
    region_ids = np.asarray(region_ids)
    cell_types = np.asarray(cell_types, dtype=object)
    out: dict[int, dict] = {}
    for r in np.unique(region_ids):
        m = region_ids == r
        types, counts = np.unique(cell_types[m], return_counts=True)
        out[int(r)] = dict(zip(types.tolist(), (counts / counts.sum()).tolist()))
    return out


def section_qc(region_ids, cell_types, reference_composition: dict) -> dict:
    """QC a section by comparing its per-region cell-type composition to a reference.

    For each region present, compute the Jensen-Shannon divergence between the observed
    cell-type distribution and the reference (e.g. the Allen ABC whole-brain MERFISH atlas).
    A region absent from the reference scores the maximum (1.0) — it shouldn't be there. The
    section ``score`` is the cell-count-weighted mean divergence; high score => the
    registration places cells in regions whose cell-type makeup is implausible."""
    region_ids = np.asarray(region_ids)
    obs = region_composition(region_ids, cell_types)
    rows = []
    for r, comp in obs.items():
        n = int(np.sum(region_ids == r))
        if r in reference_composition:
            keys = sorted(set(comp) | set(reference_composition[r]))
            pv = [comp.get(k, 0.0) for k in keys]
            qv = [reference_composition[r].get(k, 0.0) for k in keys]
            js = jensen_shannon(pv, qv)
        else:
            js = 1.0
        rows.append({"region_id": r, "js": js, "n": n})
    qc = pd.DataFrame(rows)
    score = float(np.average(qc["js"], weights=qc["n"])) if len(qc) else 0.0
    return {"per_region": qc, "score": score, "n": int(len(region_ids))}


# --- synthetic end-to-end harness (no downloads) ------------------------------

def sample_section_cells(annotation, plane: AnchoredPlane, n: int = 4000, seed: int = 0,
                         p_dominant: float = 0.85) -> dict:
    """Sample cells from a plane through the atlas, with a region-dependent 'cell type'.

    The cell type is the region's dominant type with probability ``p_dominant`` (else a random
    other), giving each region a peaked composition — the signal the QC cross-check checks for.
    Background hits are dropped. Returns ``uv``, ``truth_region``, ``cell_type``, ``ccf``."""
    rng = np.random.default_rng(seed)
    uv = rng.random((n, 2))
    ccf = plane.to_ccf(uv)
    reg = _lookup(annotation, ccf)
    keep = reg > 0
    uv, ccf, reg = uv[keep], ccf[keep], reg[keep]

    present = np.unique(reg)
    ct = np.empty(len(reg), dtype=object)
    rand = rng.random(len(reg))
    other = present[rng.integers(0, len(present), len(reg))]
    for i in range(len(reg)):
        ct[i] = f"ct{reg[i]}" if rand[i] < p_dominant else f"ct{other[i]}"
    return {"uv": uv, "truth_region": reg, "cell_type": ct, "ccf": ccf}


def run_on_atlas(annotation, ontology, plane: AnchoredPlane | None = None, n_perturb: int = 64,
                 seed: int = 0, angle_deg: float = 2.5, shift: float = 2.5) -> dict:
    """Full registration pipeline on *any* labelled atlas volume — synthetic or real CCFv3.

    Samples a section from ``plane`` (default: an oblique coronal plane), registers it with a
    realistic anchoring error, then labels every cell + scores calibrated confidence + QCs.
    Runs identically on ``synthetic_ccf()`` and on ``load_ccf_brainglobe()`` — the whole point
    of the backend-agnostic core."""
    if plane is None:
        plane = coronal_plane(annotation.shape)
    cells = sample_section_cells(annotation, plane, seed=seed)
    reference = region_composition(cells["truth_region"], cells["cell_type"])

    est = plane.perturb(np.random.default_rng(seed + 7), angle_deg=angle_deg, shift=shift)
    reg = PlaneRegistration(est)

    conf = region_confidence(cells["uv"], reg, annotation, n_perturb=n_perturb,
                             angle_deg=angle_deg, shift=shift, seed=seed)
    assigned = assign_regions(cells["uv"], reg, annotation, ontology)
    qc = section_qc(assigned["region_id"], cells["cell_type"], reference)

    correct = conf["region_id"].to_numpy() == cells["truth_region"]
    lo = (conf["confidence"] < 0.9).to_numpy()
    return {
        "n_cells": int(len(cells["uv"])),
        "n_regions": int(len(reference)),
        "accuracy": round(float(correct.mean()), 4),
        "mean_confidence": round(float(conf["confidence"].mean()), 4),
        "acc_high_conf": round(float(correct[~lo].mean()), 4) if (~lo).any() else None,
        "acc_low_conf": round(float(correct[lo].mean()), 4) if lo.any() else None,
        "qc_score": round(qc["score"], 4),
        "_cells": cells, "_conf": conf, "_assigned": assigned,
    }


def run_synthetic(n_perturb: int = 64, seed: int = 0) -> dict:
    """Convenience wrapper: ``run_on_atlas`` on the dependency-light synthetic CCF."""
    annotation, ontology = synthetic_ccf()
    return run_on_atlas(annotation, ontology, n_perturb=n_perturb, seed=seed)


def make_figure(res: dict, name: str = "atlas_registration") -> str:
    """Validation figure centred on the new capabilities: calibration + spatial confidence + QC."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    cells, conf = res["_cells"], res["_conf"]
    correct = conf["region_id"].to_numpy() == cells["truth_region"]
    sns.set_theme(style="white", context="talk")
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    fig.suptitle(
        f"2D MERFISH section -> CCFv3 registration  ({res['n_cells']:,} cells · "
        f"{res['n_regions']} regions · acc={res['accuracy']:.2f} · QC={res['qc_score']:.2f})",
        fontsize=15, fontweight="bold", y=0.97)

    # (0,0) calibration: accuracy rises with ensemble confidence
    a = fig.add_subplot(gs[0, 0])
    df = pd.DataFrame({"conf": conf["confidence"], "ok": correct})
    bins = np.linspace(df["conf"].min(), 1.0, 9)
    df["_b"] = pd.cut(df["conf"], bins, include_lowest=True)
    cal = df.groupby("_b", observed=True).agg(acc=("ok", "mean"), n=("ok", "size"))
    ctr = [iv.mid for iv in cal.index]
    a.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    a.plot(ctr, cal["acc"], "-o", c="#1D9E75", lw=2)
    a.set(title="Confidence is calibrated", xlabel="registration-ensemble confidence",
          ylabel="accuracy in bin", xlim=(0, 1.02), ylim=(0, 1.02))

    # (0,1) cells in section space, colored by assigned region (factorized so real CCF ids color cleanly)
    a = fig.add_subplot(gs[0, 1])
    a.scatter(cells["uv"][:, 0], cells["uv"][:, 1], c=pd.factorize(conf["region_id"])[0],
              cmap="tab20", s=7, lw=0)
    a.set(title=f"Per-cell region label ({conf['region_id'].nunique()} regions)",
          xticks=[], yticks=[]); a.set_aspect("equal")

    # (1,0) cells colored by confidence — boundaries light up as uncertain
    a = fig.add_subplot(gs[1, 0])
    sc = a.scatter(cells["uv"][:, 0], cells["uv"][:, 1], c=conf["confidence"], cmap="viridis",
                   s=7, vmin=0, vmax=1, lw=0)
    a.set(title="Per-cell confidence (low at region borders)", xticks=[], yticks=[])
    a.set_aspect("equal"); fig.colorbar(sc, ax=a, shrink=0.6, label="confidence")

    # (1,1) confidence: correct vs incorrect calls
    a = fig.add_subplot(gs[1, 1])
    for ok, c, lbl in [(True, "#1D9E75", "correct"), (False, "#D85A30", "incorrect")]:
        v = conf["confidence"].to_numpy()[correct == ok]
        if len(v):
            a.hist(v, bins=24, range=(0, 1), alpha=0.7, color=c, label=lbl)
    a.set(title="Confidence: correct vs incorrect", xlabel="confidence", ylabel="cells")
    a.legend(frameon=False)

    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-perturb", type=int, default=64)
    ap.add_argument("--fig", action="store_true", help="write the validation figure")
    ap.add_argument("--real", action="store_true",
                    help="run on the real Allen CCFv3 (needs brainglobe-atlasapi)")
    ap.add_argument("--atlas", default="allen_mouse_100um", help="brainglobe atlas for --real")
    ap.add_argument("--depth", type=int, default=3,
                    help="CCF ontology depth for --real (coarser = more robust per-cell labels)")
    args = ap.parse_args()

    if args.real:
        annotation, ontology = load_ccf_brainglobe(args.atlas, depth=args.depth)
        res = run_on_atlas(annotation, ontology, n_perturb=args.n_perturb)
        name = "atlas_registration_ccfv3"
    else:
        res = run_synthetic(n_perturb=args.n_perturb)
        name = "atlas_registration"
    report = {k: v for k, v in res.items() if not k.startswith("_")}
    if args.fig:
        report["figure"] = make_figure(res, name=name)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

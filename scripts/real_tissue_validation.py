"""Real-tissue validation — register a REAL MERFISH section (Moffitt 2018 hypothalamus) to the
Allen CCFv3 and validate with REAL cell types, breaking the circular synthetic validation.

The audit's #1 gap: every "validated on CCFv3" number used a *planted synthetic error on
CCF-derived sections* with *region-conditioned synthetic cell types* — so QC checked labels against
a label-derived signal. This runs the pipeline on actual tissue:

  * a real coronal Moffitt slice (real `Centroid_X/Y` microns, real `Cell_class`),
  * automated placement into the CCFv3 via `coarse_anchor` (the honest, possibly-hard part — a
    1.8 mm preoptic ROI vs the whole brain),
  * two REAL-DATA validations, neither circular:
      (A) **anatomical ground truth** — we *know* the tissue is hypothalamus, so a good placement
          should land most cells in CCF `HY` (+ descendants);
      (B) **QC sensitivity (honest scope)** — a grossly-misregistered placement scores worse than
          the fitted one. BUT (adversarially checked): the fitted baseline is self-referential (the
          reference is built from the fitted placement, so JS(p,p)=0 by construction) and the
          gross-error signal fires on region DISJOINTNESS (``section_qc``'s missing-region=1.0
          default) — so the real cell types are *inert* here (scrambling them yields the same score,
          reported below). Genuinely cell-type-aware, non-circular QC needs an EXTERNAL per-region
          reference (Allen ABC) in the cells' taxonomy, which Moffitt's vocabulary does not provide.
          This run quantifies that gap honestly rather than papering over it.

Honest by construction: it reports where automated placement of a small ROI succeeds or fails, and
explicitly flags which "validations" are self-fulfilling.

Run:  python scripts/real_tissue_validation.py
Produces assets/real_tissue_validation.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_registration as ar  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"
DATA = Path(__file__).resolve().parent.parent / "data" / "anndata" / "merfish.h5ad"


def _hy_leaf_ids(bg, annotation) -> np.ndarray:
    th = bg.structures.acronym_to_id_map["HY"]
    desc = bg.get_structure_descendants("HY")
    ids = {bg.structures.acronym_to_id_map[a] for a in desc} | {th}
    return np.array(sorted(int(i) for i in np.unique(annotation) if int(i) in ids))


def _section_plane(ap, scale, theta_deg, res, center_yx) -> ar.AnchoredPlane:
    """Map centered section coords (x, y microns) -> CCF voxel: section centre -> ``center_yx``
    (CCF voxel y,x) at AP=ap, with in-plane scale + rotation."""
    cy, cx = center_yx
    t = np.deg2rad(theta_deg)
    ct, st = np.cos(t), np.sin(t)
    O = [float(ap), float(cy), float(cx)]
    U = [0.0, (scale / res[1]) * st, (scale / res[2]) * ct]   # per micron of section-x
    V = [0.0, (scale / res[1]) * ct, -(scale / res[2]) * st]  # per micron of section-y
    return ar.AnchoredPlane(O, U, V)


def _rasterize_density(xy, res):
    """Simple 2D cell-density image of centered section coords, on a grid at the atlas pixel size."""
    px = float(res[2])
    x, y = xy[:, 0], xy[:, 1]
    nx = max(8, int((x.max() - x.min()) / px) + 1)
    ny = max(8, int((y.max() - y.min()) / px) + 1)
    H, ye, xe = np.histogram2d(y, x, bins=[ny, nx])
    yc = 0.5 * (ye[:-1] + ye[1:])
    xc = 0.5 * (xe[:-1] + xe[1:])
    return H, xc, yc


def run(atlas: str = "allen_mouse_100um", bregma: float = -14.0, depth: int = 3) -> dict:
    import anndata as ad
    from brainglobe_atlasapi import BrainGlobeAtlas

    A = ad.read_h5ad(DATA)
    bvals = np.array(sorted(A.obs["Bregma"].unique()), dtype=float)  # floats carry FP noise
    bregma = float(bvals[np.argmin(np.abs(bvals - bregma))])         # snap to nearest real slice
    sl = A[np.isclose(A.obs["Bregma"].to_numpy(float), bregma)]
    xy = np.column_stack([sl.obs["Centroid_X"].to_numpy(float),
                          sl.obs["Centroid_Y"].to_numpy(float)])
    xy = xy - xy.mean(0)                              # centre the ROI (abs. position is unknown)
    real_types = sl.obs["Cell_class"].astype(str).to_numpy()

    bg = BrainGlobeAtlas(atlas)
    ref = np.asarray(bg.reference, float)
    res = np.array(bg.resolution, float)
    leaf = np.asarray(bg.annotation)
    coarse, ont = ar.coarsen_to_depth(leaf, bg.structures, depth)
    hy_ids = _hy_leaf_ids(bg, leaf)
    hy_aps = np.where(np.isin(leaf, hy_ids).reshape(leaf.shape[0], -1).any(1))[0]

    dens, xc, yc = _rasterize_density(xy, res)
    nyc, nxc = leaf.shape[1] / 2.0, leaf.shape[2] / 2.0

    def place_and_frac(ap, scale, theta, cyx):
        rg = ar.PlaneRegistration(_section_plane(ap, scale, theta, res, cyx))
        lab = ar._lookup(leaf, rg.transform_points(xy))
        return rg, lab, float(np.isin(lab, hy_ids).mean())

    # (1) coarse_anchor — full-section method (centred, NO translation); expected to fail on a small ROI
    a1 = ar.coarse_anchor(ref, res, dens, xc, yc,
                          scales=(0.6, 0.8, 1.0, 1.2, 1.5), thetas_deg=(-20.0, 0.0, 20.0))
    reg1, _, frac1 = place_and_frac(a1["ap"], a1["scale"], a1["theta_deg"], (nyc, nxc))

    # (2) locate_section — adds in-plane TRANSLATION (template matching)
    a2 = ar.locate_section(ref, res, dens, scales=(0.6, 0.8, 1.0, 1.3), thetas_deg=(-20.0, 0.0, 20.0))
    _, _, frac2 = place_and_frac(a2["ap"], a2["scale"], a2["theta_deg"], (a2["ty"], a2["tx"]))

    # (3) anatomy-informed control — we KNOW it's hypothalamus: centre on the HY centroid at mid-HY AP
    ap_hy = int(round(float(hy_aps.mean())))
    yy, xx = np.where(np.isin(leaf[ap_hy], hy_ids))
    reg3, leaf3, frac3 = place_and_frac(ap_hy, 1.0, 0.0, (float(yy.mean()), float(xx.mean())))

    # --- non-circular QC sensitivity with REAL cell types (on the anatomy-informed placement) ---
    assigned = ar.assign_regions(xy, reg3, coarse, ont)["region_id"].to_numpy()
    ref_comp = ar.region_composition(assigned, real_types)             # reference = correct placement
    bad = ar.assign_regions(xy, reg1, coarse, ont)["region_id"].to_numpy()  # the failed auto-placement (AP off)
    qc_fit = ar.section_qc(assigned, real_types, ref_comp)["score"]       # self-referential -> 0 by construction
    qc_bad = ar.section_qc(bad, real_types, ref_comp)["score"]
    scrambled = np.random.default_rng(0).permutation(real_types)          # control: are real types load-bearing?
    qc_bad_scrambled = ar.section_qc(bad, scrambled, ref_comp)["score"]

    return {
        "bregma": float(bregma), "n_cells": int(len(xy)),
        "n_real_cell_types": int(len(set(real_types))),
        "HY_present_ap_range": [int(hy_aps.min()), int(hy_aps.max())],
        "placement_1_coarse_anchor": {"ap": int(a1["ap"]), "ncc": round(a1["ncc"], 3),
                                      "frac_in_HY": round(frac1, 3)},
        "placement_2_locate_section": {"ap": int(a2["ap"]), "ncc": round(a2["ncc"], 3),
                                       "ty": round(a2["ty"], 1), "tx": round(a2["tx"], 1),
                                       "frac_in_HY": round(frac2, 3)},
        "placement_3_anatomy_informed_GEOMETRY_CHECK": {
            "ap": ap_hy, "frac_in_HY": round(frac3, 3),
            "note": "self-fulfilling: ROI centred on HY centroid => high frac by construction"},
        "qc_sensitivity": {
            "fitted_self_referential": round(qc_fit, 4),
            "gross_misreg_real_types": round(qc_bad, 4),
            "gross_misreg_scrambled_types": round(qc_bad_scrambled, 4),
            "note": "scrambled == real => detection is region-disjointness (missing-region default), "
                    "NOT cell-type-aware; real non-circular QC needs an external ABC reference"},
        "_xy": xy, "_types": real_types, "_assigned": assigned,
        "_leaf_labels": leaf3, "_hy_ids": hy_ids,
    }


def make_figure(res: dict, name: str = "real_tissue_validation") -> str:
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import ListedColormap

    xy, types = res["_xy"], res["_types"]
    p2, p3 = res["placement_2_locate_section"], res["placement_3_anatomy_informed_GEOMETRY_CHECK"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Real Moffitt hypothalamus → CCFv3 (Bregma {res['bregma']:+.0f}, "
                 f"{res['n_cells']:,} cells, {res['n_real_cell_types']} real types) — anatomy-informed "
                 f"placement {p3['frac_in_HY']*100:.0f}% in HY; auto locate_section "
                 f"{p2['frac_in_HY']*100:.0f}%", fontsize=12, fontweight="bold")

    codes, uniq = pd.factorize(types)
    sc = ax[0].scatter(xy[:, 0], xy[:, 1], c=codes, cmap="tab20", s=5, lw=0)
    ax[0].set(title=f"real cell types ({len(uniq)})", xticks=[], yticks=[]); ax[0].set_aspect("equal")

    ax[1].scatter(xy[:, 0], xy[:, 1], c=pd.factorize(res["_assigned"])[0], cmap="tab20", s=5, lw=0)
    ax[1].set(title="assigned CCFv3 region", xticks=[], yticks=[]); ax[1].set_aspect("equal")

    in_hy = np.isin(res["_leaf_labels"], res["_hy_ids"]).astype(int)
    ax[2].scatter(xy[:, 0], xy[:, 1], c=in_hy, cmap=ListedColormap(["#D9D9D9", "#1D9E75"]), s=5, lw=0)
    ax[2].set(title=f"landed in hypothalamus (green) — {p3['frac_in_HY']*100:.0f}% (anatomy-informed)",
              xticks=[], yticks=[]); ax[2].set_aspect("equal")

    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bregma", type=float, default=-14.0)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()
    res = run(bregma=args.bregma, depth=args.depth)
    report = {k: v for k, v in res.items() if not k.startswith("_")}
    report["figure"] = make_figure(res)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

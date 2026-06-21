"""Automated placement of a small *dissected ROI* into the CCFv3 — the last open problem.

A 1.8 mm Moffitt hypothalamus ROI is too small/non-distinctive to localize in the whole brain by
image matching (whole-brain search lands at the wrong AP — 0% of cells in HY). The decisive fix is a
coarse **anatomical AP-range prior** — restrict the search to the structure's range
(`locate_section(..., ap_range=...)`), which any experimenter knows for a dissection.

Adversarial control (reported below): WITH the prior the section is recovered (~75% of cells in HY)
using EITHER the CCF Nissl reference OR a same-modality ABC cell-density target
(`abc_density_volume`). So the **prior — not the target modality — is the lever** (the cross-
modality hypothesis did not pan out; both targets work once the AP search is constrained).

Reports 4 placements (fraction of cells in hypothalamus `HY`): {Nissl, ABC-density} × {whole-brain,
HY-prior}.

Needs brainglobe + the ABC metadata (build once, ~1.6 GB) + the cached Moffitt h5ad.
Run:  python scripts/roi_placement_demo.py        Produces assets/roi_placement_demo.png.
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
ABC_CACHE = Path(__file__).resolve().parent.parent / "data" / "abc_cache"


def run(bregma: float = -14.0) -> dict:
    import anndata as ad
    from brainglobe_atlasapi import BrainGlobeAtlas
    from scipy.ndimage import gaussian_filter

    bg = BrainGlobeAtlas("allen_mouse_100um")
    ann = np.asarray(bg.annotation)
    res = np.array(bg.resolution, dtype=float)
    ref_nissl = np.asarray(bg.reference, dtype=float)
    hy_ids = np.array(sorted({bg.structures.acronym_to_id_map[a]
                              for a in bg.get_structure_descendants("HY")}
                             | {bg.structures.acronym_to_id_map["HY"]}))
    hy_aps = np.where(np.isin(ann, hy_ids).reshape(ann.shape[0], -1).any(1))[0]
    lo, hi = int(hy_aps.min()), int(hy_aps.max())

    density = ar.abc_density_volume(ABC_CACHE, ann.shape, res)  # same-modality target

    A = ad.read_h5ad(DATA)
    bvals = np.array(sorted(A.obs["Bregma"].unique()), dtype=float)
    bregma = float(bvals[np.argmin(np.abs(bvals - bregma))])
    sl = A[np.isclose(A.obs["Bregma"].to_numpy(float), bregma)]
    xy = np.column_stack([sl.obs["Centroid_X"].to_numpy(float),
                          sl.obs["Centroid_Y"].to_numpy(float)])
    xy = xy - xy.mean(0)
    px = float(res[2])
    nx = int((xy[:, 0].max() - xy[:, 0].min()) / px) + 1
    ny = int((xy[:, 1].max() - xy[:, 1].min()) / px) + 1
    sec_img = gaussian_filter(np.histogram2d(xy[:, 1], xy[:, 0], bins=[ny, nx])[0], 1.0)

    def frac_hy(loc):
        plane = ar.AnchoredPlane([loc["ap"], loc["ty"], loc["tx"]],
                                 [0, 0, loc["scale"] / res[2]], [0, loc["scale"] / res[1], 0])
        lab = ar._lookup(ann, ar.PlaneRegistration(plane).transform_points(xy))
        return float(np.isin(lab, hy_ids).mean())

    scales, thetas = (0.6, 0.8, 1.0, 1.3), (-20.0, 0.0, 20.0)

    def place(target, prior):
        return ar.locate_section(target, res, sec_img, scales=scales, thetas_deg=thetas, ap_range=prior)

    def rec(loc):
        return {"ap": loc["ap"], "ncc": round(loc["ncc"], 3), "frac_in_HY": round(frac_hy(loc), 3)}

    # 2x2: {Nissl, ABC-density} x {whole-brain, HY-prior}. The PRIOR is the decisive factor, not the
    # target modality (Nissl+prior ~ ABC-density+prior), so both controls are reported honestly.
    nissl_full, abc_full = place(ref_nissl, None), place(density, None)
    nissl_prior, abc_prior = place(ref_nissl, (lo, hi)), place(density, (lo, hi))
    placements = {
        "nissl_whole_brain": rec(nissl_full),
        "abc_density_whole_brain": rec(abc_full),
        "nissl_with_HY_prior": rec(nissl_prior),
        "abc_density_with_HY_prior": rec(abc_prior),
    }
    best_prior = nissl_prior if frac_hy(nissl_prior) >= frac_hy(abc_prior) else abc_prior
    return {"bregma": bregma, "n_cells": int(len(xy)), "HY_ap_range": [lo, hi],
            "placements": placements, "_xy": xy, "_ann": ann, "_res": res,
            "_hy_ids": hy_ids, "_best_prior": best_prior}


def make_figure(res: dict, name: str = "roi_placement_demo") -> str:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    p = res["placements"]
    keys = ["nissl_whole_brain", "abc_density_whole_brain", "nissl_with_HY_prior", "abc_density_with_HY_prior"]
    methods = ["Nissl\nwhole-brain", "ABC-dens\nwhole-brain", "Nissl\n+ HY prior", "ABC-dens\n+ HY prior"]
    fracs = [p[k]["frac_in_HY"] for k in keys]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    bars = ax[0].bar(methods, fracs, color=["#D85A30", "#D85A30", "#1D9E75", "#1D9E75"], width=0.7)
    ax[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=11)
    ax[0].set(ylabel="fraction of cells in hypothalamus (HY)", ylim=(0, 1),
              title="Whole-brain search fails; the AP-range prior recovers it\n(either target modality)")
    ax[0].tick_params(axis="x", labelsize=9)
    # the winning prior placement: cells colored by in/out HY
    xy = res["_xy"]
    loc = res["_best_prior"]
    plane = ar.AnchoredPlane([loc["ap"], loc["ty"], loc["tx"]],
                             [0, 0, loc["scale"] / res["_res"][2]], [0, loc["scale"] / res["_res"][1], 0])
    inhy = np.isin(ar._lookup(res["_ann"], ar.PlaneRegistration(plane).transform_points(xy)), res["_hy_ids"])
    ax[1].scatter(xy[:, 0], xy[:, 1], c=inhy.astype(int), cmap=ListedColormap(["#D9D9D9", "#1D9E75"]), s=6, lw=0)
    ax[1].set(title=f"AP-prior placement — {inhy.mean()*100:.0f}% in HY (AP {loc['ap']})",
              xticks=[], yticks=[]); ax[1].set_aspect("equal"); ax[1].invert_yaxis()
    fig.suptitle(f"Small dissected ROI → CCFv3: whole-brain search fails (too non-distinctive); a coarse "
                 f"AP-range anatomical prior recovers it (Moffitt Bregma {res['bregma']:+.0f}, "
                 f"{res['n_cells']:,} cells)", fontsize=11, fontweight="bold")
    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bregma", type=float, default=-14.0)
    args = ap.parse_args()
    res = run(bregma=args.bregma)
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    out["figure"] = make_figure(res)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

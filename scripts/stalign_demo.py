"""Real STalign LDDMM alignment of a 2D section to the 3D Allen CCFv3 — on real atlas data.

This exercises the *real* deformable backend (not the synthetic core): it fits
``STalign.LDDMM_3D_to_slice`` to register a 2D coronal section to the 3D CCFv3, maps every
cell into the atlas, and lifts over region labels — then checks recovery against ground truth
(the section was cut from a *known* CCF plane, so we know the right answer).

Heavy deps (torch + STalign, which pins numpy<1.24) — run in the isolated env:

    .venv-reg/bin/python scripts/stalign_demo.py --niter 100

Produces ``assets/atlas_registration_stalign.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_registration as ar  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def run(atlas: str = "allen_mouse_100um", niter: int = 100, depth: int = 3,
        n_cells: int = 4000, seed: int = 0) -> dict:
    from brainglobe_atlasapi import BrainGlobeAtlas

    bg = BrainGlobeAtlas(atlas)
    ref = np.asarray(bg.reference, dtype=float)
    res = np.array(bg.resolution, dtype=float)
    annotation, ontology = ar.coarsen_to_depth(np.asarray(bg.annotation), bg.structures, depth)

    nx = ref.shape
    ap = nx[0] // 2  # the KNOWN coronal plane the "section" is cut from
    xA = [np.arange(n) * d - (n - 1) * d / 2.0 for n, d in zip(nx, res)]
    sl = ref[ap]
    ys, xs = np.where(sl > sl.mean() * 0.5)
    sel = np.random.default_rng(seed).choice(len(xs), min(n_cells, len(xs)), replace=False)
    cells_xy = np.stack([xA[2][xs[sel]], xA[1][ys[sel]]], axis=1)  # (x, y) microns
    truth = annotation[ap, ys[sel], xs[sel]]

    t0 = time.time()
    reg = ar.stalign_register(ref, res, cells_xy, niter=niter, a=200.0)
    vox = reg.transform_points(cells_xy)
    assigned = ar.assign_regions(cells_xy, reg, annotation, ontology)
    fit_s = time.time() - t0

    pred = assigned["region_id"].to_numpy()
    return {
        "atlas": atlas, "niter": niter, "depth": depth, "n_cells": int(len(cells_xy)),
        "true_ap_index": int(ap),
        "recovered_ap_index_median": round(float(np.median(vox[:, 0])), 2),
        "ap_error_microns": round(float(np.median(np.abs(vox[:, 0] - ap)) * res[0]), 1),
        "region_agreement_vs_truth": round(float((pred == truth).mean()), 3),
        "fit_seconds": round(fit_s, 1),
        "_cells_xy": cells_xy, "_vox": vox, "_pred": pred, "_truth": truth,
        "_ref_slice": sl, "_ap": ap, "_res": res,
    }


def make_figure(res: dict, name: str = "atlas_registration_stalign") -> str:
    import matplotlib.pyplot as plt

    cells, vox, pred, truth = res["_cells_xy"], res["_vox"], res["_pred"], res["_truth"]
    fig = plt.figure(figsize=(16, 4.6))
    fig.suptitle(
        f"Real STalign LDDMM: 2D section → 3D Allen CCFv3  "
        f"(niter={res['niter']} · {res['n_cells']:,} cells · AP error "
        f"{res['ap_error_microns']:.0f} µm · region agreement {res['region_agreement_vs_truth']:.2f})",
        fontsize=13, fontweight="bold")

    # 1: the input section (the "new scan"), colored by true region
    a = fig.add_subplot(1, 3, 1)
    a.scatter(cells[:, 0], cells[:, 1], c=__import__("pandas").factorize(truth)[0],
              cmap="tab20", s=5, lw=0)
    a.set(title="Input 2D section (truth regions)", xticks=[], yticks=[]); a.set_aspect("equal")
    a.invert_yaxis()

    # 2: STalign-assigned regions after warping into the CCF
    a = fig.add_subplot(1, 3, 2)
    a.scatter(cells[:, 0], cells[:, 1], c=__import__("pandas").factorize(pred)[0],
              cmap="tab20", s=5, lw=0)
    a.set(title="Regions assigned via STalign→CCFv3", xticks=[], yticks=[]); a.set_aspect("equal")
    a.invert_yaxis()

    # 3: recovered AP position vs the known true plane
    a = fig.add_subplot(1, 3, 3)
    a.hist(vox[:, 0], bins=40, color="#1D9E75", alpha=0.85)
    a.axvline(res["_ap"], color="#D85A30", lw=2, ls="--", label=f"true plane (idx {res['_ap']})")
    a.set(title="Recovered anterior–posterior position", xlabel="CCF AP voxel index", ylabel="cells")
    a.legend(frameon=False, fontsize=9)

    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atlas", default="allen_mouse_100um")
    ap.add_argument("--niter", type=int, default=100)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--fig", action="store_true", default=True)
    args = ap.parse_args()

    res = run(atlas=args.atlas, niter=args.niter, depth=args.depth)
    report = {k: v for k, v in res.items() if not k.startswith("_")}
    if args.fig:
        report["figure"] = make_figure(res)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

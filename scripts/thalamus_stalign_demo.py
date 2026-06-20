"""Thalamus example with the *real* STalign LDDMM deformable backend + auto AP-anchor.

Aligns three different coronal scans through the thalamus (anterior / mid / posterior) to the
3D Allen CCFv3 with ``STalign.LDDMM_3D_to_slice``, now initialized by the training-free
``coarse_ap_search`` AP-anchor. This is the improvement over the bare STalign fit, which
misconverged on the posterior section (AP error ~1.4 mm); the auto-anchor brings it to ~0.1 mm.

Run (unified env with torch + STalign installed):
    python scripts/thalamus_stalign_demo.py --niter 80
Produces ``assets/thalamus_stalign_alignment.png``.
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


def run(atlas: str = "allen_mouse_100um", aps=(64, 72, 80), niter: int = 80,
        n_cells: int = 3000, seed: int = 0) -> dict:
    from brainglobe_atlasapi import BrainGlobeAtlas

    bg = BrainGlobeAtlas(atlas)
    ref = np.asarray(bg.reference, dtype=float)
    ann = np.asarray(bg.annotation)
    res = np.array(bg.resolution, dtype=float)
    nx = ref.shape
    xA = [np.arange(n) * d - (n - 1) * d / 2.0 for n, d in zip(nx, res)]
    th = bg.structures.acronym_to_id_map["TH"]
    th_ids = np.array(sorted({bg.structures.acronym_to_id_map[a]
                              for a in bg.get_structure_descendants("TH")} | {th}))

    scans = []
    for k, ap in enumerate(aps):
        sl = ref[ap]
        ys, xs = np.where(sl > sl.mean() * 0.5)
        sel = np.random.default_rng(seed + k).choice(len(xs), min(n_cells, len(xs)), replace=False)
        cells = np.column_stack([xA[2][xs[sel]], xA[1][ys[sel]]])  # (x, y) microns
        truth_thal = np.isin(ann[ap, ys[sel], xs[sel]], th_ids)

        t0 = time.time()
        reg = ar.stalign_register(ref, res, cells, niter=niter, a=200.0)  # init='auto'
        vox = reg.transform_points(cells)
        pred_thal = np.isin(ar._lookup(ann, vox), th_ids)
        fit_s = time.time() - t0

        inter = int((pred_thal & truth_thal).sum())
        union = int((pred_thal | truth_thal).sum())
        scans.append({
            "ap": int(ap), "n_cells": int(len(cells)),
            "recovered_ap_median": round(float(np.median(vox[:, 0])), 1),
            "ap_error_um": round(float(np.median(np.abs(vox[:, 0] - ap)) * res[0]), 1),
            "thalamus_iou": round(inter / max(union, 1), 3),
            "thalamus_recall": round(inter / max(int(truth_thal.sum()), 1), 3),
            "fit_seconds": round(fit_s, 1),
            "_cells": cells, "_truth_thal": truth_thal, "_pred_thal": pred_thal,
        })
    return {"atlas": atlas, "n_scans": len(aps), "scans": scans}


def make_figure(res: dict, name: str = "thalamus_stalign_alignment") -> str:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    scans = res["scans"]
    n = len(scans)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 9))
    fig.suptitle("Thalamus across 3 scans — real STalign LDDMM 2D→3D with auto AP-anchor "
                 "(posterior section now aligns too)", fontsize=14, fontweight="bold", y=0.99)
    truth_cmap = ListedColormap(["#E2E2E2", "#C03050"])
    pred_cmap = ListedColormap(["#E2E2E2", "#1D9E75"])
    labels = ["anterior", "mid", "posterior"]

    for j, s in enumerate(scans):
        c = s["_cells"]
        axes[0, j].scatter(c[:, 0], c[:, 1], c=s["_truth_thal"].astype(int), cmap=truth_cmap, s=6, lw=0)
        axes[0, j].set(title=f"Scan {j + 1} ({labels[j] if j < 3 else ''}): AP={s['ap']}\n"
                             f"truth — thalamus (crimson)")
        axes[1, j].scatter(c[:, 0], c[:, 1], c=s["_pred_thal"].astype(int), cmap=pred_cmap, s=6, lw=0)
        axes[1, j].set(title=f"STalign→CCFv3 (green) — IoU={s['thalamus_iou']} · "
                             f"AP err {s['ap_error_um']:.0f} µm")
        for r in range(2):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
            axes[r, j].set_aspect("equal"); axes[r, j].invert_yaxis()

    ASSETS.mkdir(exist_ok=True)
    out = ASSETS / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atlas", default="allen_mouse_100um")
    ap.add_argument("--aps", type=int, nargs="+", default=[64, 72, 80])
    ap.add_argument("--niter", type=int, default=80)
    args = ap.parse_args()

    res = run(atlas=args.atlas, aps=tuple(args.aps), niter=args.niter)
    report = {k: v for k, v in res.items() if k != "scans"}
    report["scans"] = [{k: v for k, v in s.items() if not k.startswith("_")} for s in res["scans"]]
    report["figure"] = make_figure(res)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

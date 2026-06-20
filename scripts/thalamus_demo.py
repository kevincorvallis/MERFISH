"""Thalamus example — align several *different* coronal scans to the Allen CCFv3 and show the
thalamus is recovered consistently across them, each with calibrated per-cell confidence.

The concrete answer to the original problem ("hard to align separate brain regions on a new
scan"): take 3 different coronal sections through the thalamus (anterior / mid / posterior, each
a different "specimen" via its own cell sampling) and register each into the CCFv3, then ask
*"which cells are in the thalamus (CCF ``TH`` + its 66 sub-nuclei)?"* Because each scan is cut
from a *known* CCF plane, we know the right answer and score it.

Each scan is anchored with a **DeepSlice-style affine** (here simulated with a realistic ~1.5-voxel
anchoring error — DeepSlice's stage-1 output), then per-cell labels + calibrated confidence are
computed. This is the part that robustly recovers a large structure like the thalamus across
scans; STalign deformable refinement is validated separately in ``scripts/stalign_demo.py``.

Needs the real CCFv3 (brainglobe). Run in the unified dev env:
    python scripts/thalamus_demo.py
Produces ``assets/thalamus_alignment.png``.
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


def _thalamus_ids(bg, annotation) -> np.ndarray:
    """All CCF leaf ids belonging to the thalamus (TH + descendants) that appear in the volume."""
    th = bg.structures.acronym_to_id_map["TH"]
    desc = bg.get_structure_descendants("TH")
    ids = {bg.structures.acronym_to_id_map[a] for a in desc} | {th}
    return np.array(sorted(int(i) for i in np.unique(annotation) if int(i) in ids))


def _coronal_micron_plane(ap: int, xA, res) -> ar.AnchoredPlane:
    """An affine plane mapping section coords (x, y microns) to CCF voxel coords at level ``ap``."""
    return ar.AnchoredPlane([ap, -xA[1][0] / res[1], -xA[2][0] / res[2]],
                            [0.0, 0.0, 1.0 / res[2]], [0.0, 1.0 / res[1], 0.0])


def run(atlas: str = "allen_mouse_100um", aps=(64, 72, 80), n_cells: int = 3000,
        angle_deg: float = 1.5, shift: float = 1.5, n_perturb: int = 48, seed: int = 0) -> dict:
    from brainglobe_atlasapi import BrainGlobeAtlas

    bg = BrainGlobeAtlas(atlas)
    ref = np.asarray(bg.reference, dtype=float)
    ann = np.asarray(bg.annotation)
    res = np.array(bg.resolution, dtype=float)
    nx = ref.shape
    xA = [np.arange(n) * d - (n - 1) * d / 2.0 for n, d in zip(nx, res)]
    th_ids = _thalamus_ids(bg, ann)

    scans = []
    for k, ap in enumerate(aps):
        rng = np.random.default_rng(seed + k)
        sl = ref[ap]
        ys, xs = np.where(sl > sl.mean() * 0.5)
        sel = rng.choice(len(xs), min(n_cells, len(xs)), replace=False)
        cells = np.column_stack([xA[2][xs[sel]], xA[1][ys[sel]]])  # (x, y) microns
        truth_thal = np.isin(ann[ap, ys[sel], xs[sel]], th_ids)

        # DeepSlice-style affine anchor with a realistic per-scan error, then label + confidence
        reg = ar.PlaneRegistration(_coronal_micron_plane(ap, xA, res)).perturb(
            np.random.default_rng(100 + k), angle_deg=angle_deg, shift=shift)
        conf = ar.region_confidence(cells, reg, ann, n_perturb=n_perturb,
                                    angle_deg=angle_deg, shift=shift, seed=seed)
        pred_thal = np.isin(conf["region_id"].to_numpy(), th_ids)
        confidence = conf["confidence"].to_numpy()

        inter = int((pred_thal & truth_thal).sum())
        union = int((pred_thal | truth_thal).sum())
        scans.append({
            "ap": int(ap), "n_cells": int(len(cells)),
            "n_thalamus_truth": int(truth_thal.sum()),
            "thalamus_iou": round(inter / max(union, 1), 3),
            "thalamus_recall": round(inter / max(int(truth_thal.sum()), 1), 3),
            "mean_confidence_thalamus": round(float(confidence[truth_thal].mean()), 3),
            "_cells": cells, "_truth_thal": truth_thal, "_pred_thal": pred_thal, "_conf": confidence,
        })
    return {"atlas": atlas, "n_scans": len(aps), "thalamus_leaf_regions": len(th_ids),
            "scans": scans}


def make_figure(res: dict, name: str = "thalamus_alignment") -> str:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    scans = res["scans"]
    n = len(scans)
    fig, axes = plt.subplots(3, n, figsize=(4.5 * n, 13))
    fig.suptitle("Aligning multiple coronal scans to the Allen CCFv3 — the thalamus recovered "
                 "consistently, with calibrated per-cell confidence", fontsize=14,
                 fontweight="bold", y=0.99)
    truth_cmap = ListedColormap(["#E2E2E2", "#C03050"])
    pred_cmap = ListedColormap(["#E2E2E2", "#1D9E75"])

    for j, s in enumerate(scans):
        c = s["_cells"]
        axes[0, j].scatter(c[:, 0], c[:, 1], c=s["_truth_thal"].astype(int), cmap=truth_cmap, s=6, lw=0)
        axes[0, j].set(title=f"Scan {j + 1}: coronal AP={s['ap']}\ntruth — thalamus (crimson)")
        axes[1, j].scatter(c[:, 0], c[:, 1], c=s["_pred_thal"].astype(int), cmap=pred_cmap, s=6, lw=0)
        axes[1, j].set(title=f"recovered thalamus (green) — IoU={s['thalamus_iou']} "
                             f"recall={s['thalamus_recall']}")
        sc = axes[2, j].scatter(c[:, 0], c[:, 1], c=s["_conf"], cmap="viridis", s=6, vmin=0, vmax=1, lw=0)
        axes[2, j].set(title=f"per-cell confidence (mean in TH {s['mean_confidence_thalamus']})")
        fig.colorbar(sc, ax=axes[2, j], shrink=0.6, label="confidence")
        for r in range(3):
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
    args = ap.parse_args()

    res = run(atlas=args.atlas, aps=tuple(args.aps))
    report = {k: v for k, v in res.items() if k != "scans"}
    report["scans"] = [{k: v for k, v in s.items() if not k.startswith("_")} for s in res["scans"]]
    report["figure"] = make_figure(res)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

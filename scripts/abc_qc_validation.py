"""Genuine non-circular, cell-type-aware QC on REAL tissue.

Real Moffitt 2018 cells with their real `Cell_class`, scored against a REAL external Allen ABC
reference built from ~4M ABC whole-brain MERFISH cells (`build_abc_reference`). This resolves the
three degeneracies an adversarial check found in the first QC attempt:

  * EXTERNAL reference (real ABC) — not self-referential (the old baseline was JS(p,p)=0);
  * scored only over COVERED regions via Jensen-Shannon — not the missing-region=1.0 default;
  * cell types are LOAD-BEARING — demonstrated by two discriminating controls.

The naive "scramble the labels" control is weak for this dataset because the dissected hypothalamic
ROI lands almost entirely in ONE depth-3 region (region-homogeneity), so two sharper controls are used:

  * wrong marginal  — replace the real types with a single wrong class; JS must worsen sharply.
  * shuffled reference — compare the real composition to RANDOM ABC regions; JS must worsen (the
    real biology matches the CORRECT atlas regions, not random ones).

Needs brainglobe + the ABC metadata (downloaded once, ~1.6 GB) + the cached Moffitt h5ad.
Run:  python scripts/abc_qc_validation.py
Produces assets/abc_qc_validation.png.
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


def run(bregma: float = -14.0, depth: int = 3, seed: int = 0) -> dict:
    import anndata as ad
    from brainglobe_atlasapi import BrainGlobeAtlas

    bg = BrainGlobeAtlas("allen_mouse_100um")
    abc_ref = ar.build_abc_reference(ABC_CACHE, bg.structures, depth=depth)

    A = ad.read_h5ad(DATA)
    bvals = np.array(sorted(A.obs["Bregma"].unique()), dtype=float)
    bregma = float(bvals[np.argmin(np.abs(bvals - bregma))])
    sl = A[np.isclose(A.obs["Bregma"].to_numpy(float), bregma)]
    xy = np.column_stack([sl.obs["Centroid_X"].to_numpy(float),
                          sl.obs["Centroid_Y"].to_numpy(float)])
    xy = xy - xy.mean(0)
    real_broad = ar.to_broad_class(sl.obs["Cell_class"].astype(str).to_numpy())

    leaf = np.asarray(bg.annotation)
    res = np.array(bg.resolution, dtype=float)
    coarse, ont = ar.coarsen_to_depth(leaf, bg.structures, depth)
    hy_ids = np.array(sorted({bg.structures.acronym_to_id_map[a]
                              for a in bg.get_structure_descendants("HY")}
                             | {bg.structures.acronym_to_id_map["HY"]}))
    hy_aps = np.where(np.isin(leaf, hy_ids).reshape(leaf.shape[0], -1).any(1))[0]
    ap = int(round(float(hy_aps.mean())))
    yy, xx = np.where(np.isin(leaf[ap], hy_ids))
    plane = ar.AnchoredPlane([ap, float(yy.mean()), float(xx.mean())],
                             [0, 0, 1 / res[2]], [0, 1 / res[1], 0])
    region_ids = ar.assign_regions(xy, ar.PlaneRegistration(plane), coarse, ont)["region_id"].to_numpy()

    # the genuine QC + two discriminating controls
    q_real = ar.composition_qc(region_ids, real_broad, abc_ref)
    q_wrong = ar.composition_qc(region_ids, np.array(["excitatory"] * len(region_ids), dtype=object), abc_ref)
    rng = np.random.default_rng(seed)
    keys = list(abc_ref)
    vals = [abc_ref[k] for k in keys]
    perm = rng.permutation(len(keys))
    shuffled_ref = {k: vals[perm[i]] for i, k in enumerate(keys)}
    q_shuf = ar.composition_qc(region_ids, real_broad, shuffled_ref)

    u, c = np.unique(region_ids, return_counts=True)
    return {
        "bregma": bregma, "n_cells": int(len(xy)),
        "abc_reference_regions": int(len(abc_ref)),
        "section_distinct_regions": int(len(u)),
        "section_dominant_region": ont.get(int(u[c.argmax()]), int(u[c.argmax()])),
        "coverage": round(q_real["coverage"], 3),
        "qc_real_types": round(q_real["score"], 3),
        "qc_wrong_marginal_all_excitatory": round(q_wrong["score"], 3),
        "qc_shuffled_reference": round(q_shuf["score"], 3),
        "cell_types_load_bearing": bool(q_wrong["score"] > q_real["score"] + 0.1
                                        and q_shuf["score"] > q_real["score"] + 0.1),
    }


def make_figure(res: dict, name: str = "abc_qc_validation") -> str:
    import matplotlib.pyplot as plt

    labels = ["real types\n(correct regions)", "wrong marginal\n(all excitatory)",
              "real types\n(shuffled ABC regions)"]
    scores = [res["qc_real_types"], res["qc_wrong_marginal_all_excitatory"], res["qc_shuffled_reference"]]
    colors = ["#1D9E75", "#D85A30", "#D85A30"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, scores, color=colors, width=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=11)
    ax.set_ylabel("composition QC (Jensen-Shannon vs ABC)")
    ax.set_title(f"Genuine non-circular cell-type-aware QC on real tissue\n"
                 f"real Moffitt ({res['n_cells']:,} cells, Bregma {res['bregma']:+.0f}) vs real Allen ABC "
                 f"({res['abc_reference_regions']} regions) · coverage {res['coverage']*100:.0f}%",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(scores) * 1.25)
    ax.margins(x=0.05)
    fig.text(0.5, -0.02, "Lower = better. Real types match the correct ABC regions far better than a "
             "wrong cell-type marginal or shuffled regions → the QC is genuinely cell-type-aware.",
             ha="center", fontsize=9, style="italic")
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
    res["figure"] = make_figure(res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

"""Segmentation swap — why the segmentation method changes your cell types.

The 2026 methods review (docs/methods-review-2026.md §1) flagged cell segmentation as the
highest-leverage upstream change: it produces the cell-by-gene matrix that feeds
``PCA -> UMAP -> Leiden -> cell-type mapping``, and the 2025 "Segmentation Matters"
benchmark showed the choice measurably splits/merges/drops downstream clusters.

The production-quality MERSCOPE-ready tools are **proseg** (``proseg --merscope``),
**Cellpose-SAM**, **RNA2seg**, and **segger** — but they need raw MERSCOPE output
(``detected_transcripts.csv.gz`` and/or DAPI/membrane mosaics) that isn't in this repo.
So, exactly like ``demo_pipeline.py``, this script *simulates molecule-level MERFISH data*
with known ground truth and runs the real comparison end to end:

  1. nucleus-only        — count transcripts near the nucleus only (under-segments)
  2. Voronoi / expansion — assign each transcript to its nearest nucleus (the vendor-
                           default-style morphology baseline)
  3. transcript-aware    — the modern paradigm (Baysor / proseg / segger): assign each
                           transcript by spatial position *and* expression likelihood,
                           so boundary molecules go to the cell whose profile they match

It then builds a cell-by-gene matrix from each segmentation, runs the standard Scanpy
pipeline, and reports both transcript-assignment accuracy and downstream Leiden ARI vs
ground-truth cell types — demonstrating that better segmentation yields cleaner cell types.

For real data, swap in the genuine tools (see ``cellpose_sam_segment`` and
docs/methods-review-2026.md §1). Usage:
    python scripts/segmentation_demo.py --fig
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets"


# --- 1. simulate molecule-level MERFISH data ---------------------------------

def simulate_molecules(grid: int = 18, spacing: float = 10.0, k_types: int = 6,
                       n_genes: int = 60, mean_tx: int = 80, sigma_cyto: float = 3.6,
                       p_marker: float = 0.7, seed: int = 0) -> dict:
    """Simulate transcripts (x, y, gene, true_cell) for a grid of overlapping cells.

    Cytoplasmic spread (``sigma_cyto``) is a meaningful fraction of cell ``spacing`` so
    neighbouring cells' transcript clouds overlap — the dense-tissue regime where
    morphology-only segmentation misassigns boundary molecules and transcript identity is
    what disambiguates them.
    """
    rng = np.random.default_rng(seed)
    markers_per = n_genes // k_types
    marker_of = {t: np.arange(t * markers_per, (t + 1) * markers_per) for t in range(k_types)}

    gx, gy = np.meshgrid(np.arange(grid), np.arange(grid))
    centers = np.column_stack([gx.ravel(), gy.ravel()]).astype(float) * spacing
    centers += rng.normal(0, spacing * 0.12, centers.shape)
    n_cells = centers.shape[0]
    true_type = rng.integers(0, k_types, n_cells)

    pos, gene_idx, true_cell = [], [], []
    for c in range(n_cells):
        n = max(5, int(rng.poisson(mean_tx)))
        pos.append(centers[c] + rng.normal(0, sigma_cyto, (n, 2)))
        is_marker = rng.random(n) < p_marker
        g = np.where(is_marker,
                     rng.choice(marker_of[true_type[c]], n),
                     rng.integers(0, n_genes, n))
        gene_idx.append(g)
        true_cell.append(np.full(n, c))

    return {
        "centers": centers, "true_type": true_type, "n_cells": n_cells,
        "n_genes": n_genes, "k_types": k_types, "sigma_cyto": sigma_cyto,
        "pos": np.vstack(pos),
        "gene_idx": np.concatenate(gene_idx),
        "true_cell": np.concatenate(true_cell),
    }


# --- 2. segmentation strategies ----------------------------------------------

def _knn_centers(sim: dict, k: int = 5):
    from scipy.spatial import cKDTree
    tree = cKDTree(sim["centers"])
    dist, idx = tree.query(sim["pos"], k=k)
    return dist, idx


def segment_voronoi(sim: dict) -> np.ndarray:
    """Baseline: every transcript -> nearest nucleus centre (morphology expansion)."""
    _, idx = _knn_centers(sim, k=1)
    return idx.ravel()


def segment_nucleus_only(sim: dict, nucleus_radius: float = 2.2) -> np.ndarray:
    """Under-segmentation: keep only transcripts within the nucleus radius, else -1."""
    dist, idx = _knn_centers(sim, k=1)
    lab = idx.ravel().copy()
    lab[dist.ravel() > nucleus_radius] = -1
    return lab


def _profiles(sim: dict, labels: np.ndarray, pseudo: float = 1.0) -> np.ndarray:
    """Per-cell gene probability profiles from currently-assigned transcripts."""
    counts = np.full((sim["n_cells"], sim["n_genes"]), pseudo, dtype=float)
    m = labels >= 0
    np.add.at(counts, (labels[m], sim["gene_idx"][m]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def segment_transcript_aware(sim: dict, n_iter: int = 6, k: int = 5, lam: float = 1.0,
                             seed: int = 0) -> np.ndarray:
    """Modern paradigm (Baysor/proseg/segger-style): assign each transcript to the
    candidate cell maximising log P(gene | cell profile) + lam * spatial log-likelihood.

    Iterates EM-style: estimate per-cell expression profiles, reassign transcripts,
    repeat. Boundary molecules get pulled to the cell whose expression they actually
    match instead of merely the nearest nucleus.
    """
    dist, cand = _knn_centers(sim, k=k)                       # T×k candidates
    spatial = -(dist ** 2) / (2.0 * sim["sigma_cyto"] ** 2)   # T×k spatial log-lik
    gene_idx = sim["gene_idx"]
    labels = cand[:, 0].copy()                                # init = Voronoi
    rows = np.arange(sim["pos"].shape[0])

    for _ in range(n_iter):
        prof = _profiles(sim, labels)
        expr = np.log(prof[cand, gene_idx[:, None]] + 1e-9)   # T×k expression log-lik
        labels = cand[rows, (expr + lam * spatial).argmax(axis=1)]
    return labels


def has_cellpose() -> bool:
    import importlib.util
    return importlib.util.find_spec("cellpose") is not None


def cellpose_sam_segment(image: np.ndarray, diameter: float | None = None) -> np.ndarray:
    """Production image-based path: Cellpose-SAM nuclei/cell masks from a DAPI mosaic.

    Cellpose-SAM (Cellpose 4.x, SAM ViT-L backbone) is the vendor-endorsed swap for the
    default MERSCOPE Cellpose model. Guarded so the demo/CI runs without the heavy torch
    dependency; install ``cellpose>=4`` to enable. See docs/methods-review-2026.md §1.
    """
    if not has_cellpose():
        raise NotImplementedError(
            "Install cellpose>=4 (Cellpose-SAM) to run image-based segmentation; "
            "see docs/methods-review-2026.md §1. The transcript-aware demo needs no torch.")
    from cellpose import models  # pragma: no cover - optional heavy path
    model = models.CellposeModel(gpu=False)
    masks, _, _ = model.eval(image, diameter=diameter)
    return masks


# --- 3. downstream impact: cell-by-gene -> Leiden -> ARI ----------------------

def build_cell_by_gene(sim: dict, labels: np.ndarray):
    """Aggregate assigned transcripts into a cells × genes AnnData (the pipeline input)."""
    import scanpy as sc

    X = np.zeros((sim["n_cells"], sim["n_genes"]), dtype="float32")
    m = labels >= 0
    np.add.at(X, (labels[m], sim["gene_idx"][m]), 1.0)
    ad = sc.AnnData(X)
    ad.var_names = [f"g{j:02d}" for j in range(sim["n_genes"])]
    ad.obs["true_type"] = [str(t) for t in sim["true_type"]]
    ad.obsm["spatial"] = sim["centers"]
    return ad


def leiden_ari(ad, min_counts: int = 5, seed: int = 0) -> float:
    """Standard Scanpy pipeline on a cell-by-gene matrix; ARI of Leiden vs true type."""
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score

    sc.settings.verbosity = 0
    a = ad[np.asarray(ad.X.sum(1)).ravel() >= min_counts].copy()
    if a.n_obs < 10:
        return 0.0
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.scale(a, max_value=10)
    nc = int(min(30, a.n_vars - 1, a.n_obs - 1))
    sc.tl.pca(a, n_comps=nc, svd_solver="arpack")
    sc.pp.neighbors(a, n_neighbors=15, n_pcs=nc, random_state=seed)
    sc.tl.leiden(a, resolution=1.0, flavor="igraph", n_iterations=2, directed=False,
                 random_state=seed)
    return float(adjusted_rand_score(a.obs["true_type"], a.obs["leiden"]))


def assignment_accuracy(labels: np.ndarray, sim: dict) -> float:
    """Fraction of transcripts assigned to their true cell (-1 counts as wrong)."""
    return float(np.mean(labels == sim["true_cell"]))


def evaluate(seed: int = 0) -> dict:
    """Run all three segmentations; report assignment accuracy + downstream Leiden ARI."""
    t0 = time.time()
    sim = simulate_molecules(seed=seed)
    methods = {
        "nucleus_only": segment_nucleus_only(sim),
        "voronoi": segment_voronoi(sim),
        "transcript_aware": segment_transcript_aware(sim, seed=seed),
    }
    out = {}
    for name, lab in methods.items():
        out[name] = {
            "acc": round(assignment_accuracy(lab, sim), 4),
            "ari": round(leiden_ari(build_cell_by_gene(sim, lab), seed=seed), 4),
            "n_assigned": int(np.sum(lab >= 0)),
            "labels": lab,
        }
    out["_sim"] = sim
    out["_runtime_s"] = round(time.time() - t0, 1)
    return out


# --- 4. figure ---------------------------------------------------------------

def make_figure(res: dict, name: str = "segmentation_demo") -> str:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sim = res["_sim"]
    sns.set_theme(style="white", context="talk")
    methods = ["nucleus_only", "voronoi", "transcript_aware"]
    labels_pretty = ["nucleus-only", "Voronoi (default)", "transcript-aware (modern)"]
    pal = ["#888780", "#D85A30", "#1D9E75"]

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
    fig.suptitle("Segmentation swap — the method changes your cell types "
                 f"({sim['n_cells']} cells · {len(sim['true_cell']):,} transcripts)",
                 fontsize=16, fontweight="bold", y=0.96)

    # crop for the spatial panels
    lo, hi = 30, 80
    cm = ((sim["pos"][:, 0] > lo) & (sim["pos"][:, 0] < hi) &
          (sim["pos"][:, 1] > lo) & (sim["pos"][:, 1] < hi))

    # (0,0) ground truth: transcripts colored by true cell type
    a = fig.add_subplot(gs[0, 0])
    tt = sim["true_type"][sim["true_cell"][cm]]
    a.scatter(sim["pos"][cm, 0], sim["pos"][cm, 1], c=tt, cmap="tab10", s=6, lw=0)
    cc = ((sim["centers"][:, 0] > lo) & (sim["centers"][:, 0] < hi) &
          (sim["centers"][:, 1] > lo) & (sim["centers"][:, 1] < hi))
    a.scatter(sim["centers"][cc, 0], sim["centers"][cc, 1], c="k", s=18, marker="+")
    a.set(title="Ground truth (transcripts by cell type)", xticks=[], yticks=[])
    a.set_aspect("equal")

    # (0,1) transcript-assignment accuracy
    a = fig.add_subplot(gs[0, 1])
    accs = [res[m]["acc"] for m in methods]
    a.bar(labels_pretty, accs, color=pal, edgecolor="white", lw=1.5)
    for i, v in enumerate(accs):
        a.text(i, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")
    a.set(title="Transcript-assignment accuracy", ylabel="fraction correct", ylim=(0, 1.05))
    a.tick_params(axis="x", labelsize=11)

    # (1,0) downstream Leiden ARI
    a = fig.add_subplot(gs[1, 0])
    aris = [res[m]["ari"] for m in methods]
    a.bar(labels_pretty, aris, color=pal, edgecolor="white", lw=1.5)
    for i, v in enumerate(aris):
        a.text(i, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")
    a.set(title="Downstream Leiden ARI vs true cell types", ylabel="adjusted Rand index",
          ylim=(0, max(aris) * 1.2 + 0.05))
    a.tick_params(axis="x", labelsize=11)

    # (1,1) where transcript-aware rescues Voronoi mistakes (crop)
    a = fig.add_subplot(gs[1, 1])
    vor, aware, truth = res["voronoi"]["labels"], res["transcript_aware"]["labels"], sim["true_cell"]
    rescued = cm & (vor != truth) & (aware == truth)
    both_wrong = cm & (vor != truth) & (aware != truth)
    correct = cm & (vor == truth)
    a.scatter(sim["pos"][correct, 0], sim["pos"][correct, 1], c="#D3D1C7", s=5, lw=0,
              label="Voronoi correct")
    a.scatter(sim["pos"][both_wrong, 0], sim["pos"][both_wrong, 1], c="#D85A30", s=10, lw=0,
              label="both wrong")
    a.scatter(sim["pos"][rescued, 0], sim["pos"][rescued, 1], c="#1D9E75", s=14, lw=0,
              label="rescued by transcript-aware")
    a.set(title="Boundary transcripts rescued (crop)", xticks=[], yticks=[])
    a.set_aspect("equal")
    a.legend(fontsize=9, framealpha=0.9, loc="upper right")

    out = ASSETS / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fig", action="store_true")
    args = ap.parse_args()

    res = evaluate(seed=args.seed)
    report = {m: {k: v for k, v in res[m].items() if k != "labels"}
              for m in ("nucleus_only", "voronoi", "transcript_aware")}
    report["runtime_s"] = res["_runtime_s"]
    report["cellpose_available"] = has_cellpose()
    if args.fig:
        report["figure"] = make_figure(res)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

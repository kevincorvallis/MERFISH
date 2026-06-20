"""Live integration test — run the analysis pipeline on REAL public spatial data.

Unlike the synthetic demo, this downloads a genuine MERFISH dataset (Moffitt et al.
2018, mouse hypothalamic preoptic region) via ``squidpy.datasets`` — no Google Cloud
auth required — and runs the same Scanpy + squidpy pipeline end to end, then reports
metrics. Used both as a CLI smoke test and by the pytest suite in ``tests/``.

Usage:
    pip install scanpy squidpy leidenalg igraph seaborn
    python scripts/live_test.py --dataset merfish --fig
    python scripts/live_test.py --dataset seqfish        # metrics only

Datasets (all real, openly downloadable): merfish, seqfish, slideseqv2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scanpy as sc

sc.settings.verbosity = 0

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def load_dataset(name: str):
    """Download a real public spatial-transcriptomics dataset via squidpy."""
    import squidpy as sq

    loaders = {
        "merfish": sq.datasets.merfish,        # Moffitt 2018 hypothalamus MERFISH
        "seqfish": sq.datasets.seqfish,        # mouse embryo seqFISH
        "slideseqv2": sq.datasets.slideseqv2,  # mouse hippocampus Slide-seqV2
    }
    if name not in loaders:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(loaders)}")
    return loaders[name]()


def run_pipeline(ad, n_comps: int = 50, resolution: float = 1.0,
                 label_key: str | None = None) -> dict:
    """Run the standard MERFISH single-cell pipeline; return metrics. Mutates `ad`.

    If the dataset ships author cell-type labels (auto-detected as ``Cell_class``),
    also report the Adjusted Rand Index between unsupervised Leiden clusters and
    those labels — a quantitative check that the pipeline recovers known biology.
    """
    t0 = time.time()
    ad.var_names_make_unique()
    if label_key is None and "Cell_class" in ad.obs:
        label_key = "Cell_class"

    sc.pp.filter_cells(ad, min_counts=10)
    sc.pp.filter_genes(ad, min_cells=5)

    sc.pp.normalize_total(ad)
    sc.pp.log1p(ad)
    ad.raw = ad
    sc.pp.scale(ad, max_value=10)

    nc = int(min(n_comps, ad.n_vars - 1, ad.n_obs - 1))
    sc.tl.pca(ad, n_comps=nc, svd_solver="arpack")
    sc.pp.neighbors(ad, n_neighbors=15, n_pcs=nc)
    sc.tl.umap(ad)
    sc.tl.leiden(ad, resolution=resolution, flavor="igraph", n_iterations=2,
                 directed=False, random_state=0)

    metrics = {
        "n_cells": int(ad.n_obs),
        "n_genes": int(ad.n_vars),
        "n_pcs": nc,
        "n_clusters": int(ad.obs["leiden"].nunique()),
        "has_spatial": "spatial" in ad.obsm,
        "runtime_s": round(time.time() - t0, 1),
    }
    if label_key and label_key in ad.obs:
        from sklearn.metrics import adjusted_rand_score
        metrics["label_key"] = label_key
        metrics["n_published_types"] = int(ad.obs[label_key].nunique())
        metrics["ari_vs_published"] = round(
            float(adjusted_rand_score(ad.obs[label_key], ad.obs["leiden"])), 3)
    return metrics


def make_figure(ad, name: str, label_key: str | None = None) -> str:
    """Validation figure: Leiden vs published labels, in UMAP, space, and concordance."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import adjusted_rand_score

    if label_key is None and "Cell_class" in ad.obs:
        label_key = "Cell_class"
    sns.set_theme(style="white", context="talk")

    leiden_cats = sorted(ad.obs["leiden"].cat.categories, key=int)
    lpal = sns.color_palette("husl", len(leiden_cats))
    lmap = {c: lpal[i] for i, c in enumerate(leiden_cats)}
    lcolors = np.array([lmap[v] for v in ad.obs["leiden"].astype(str)])

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.18)
    title = (f"Live test on REAL MERFISH data — {name} "
             f"({ad.n_obs:,} cells · {ad.n_vars} genes · {len(leiden_cats)} Leiden clusters)")
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.95)

    # (0,0) UMAP by Leiden (unsupervised)
    a = fig.add_subplot(gs[0, 0])
    a.scatter(*ad.obsm["X_umap"].T, c=lcolors, s=3, lw=0)
    a.set(title=f"UMAP — {len(leiden_cats)} Leiden clusters (unsupervised)",
          xticks=[], yticks=[])

    if label_key and label_key in ad.obs:
        classes = list(ad.obs[label_key].astype("category").cat.categories)
        cpal = sns.color_palette("tab20", len(classes))
        cmap = {c: cpal[i] for i, c in enumerate(classes)}
        ccolors = np.array([cmap[v] for v in ad.obs[label_key].astype(str)])
        ari = adjusted_rand_score(ad.obs[label_key], ad.obs["leiden"])

        # (0,1) UMAP by published cell class
        a = fig.add_subplot(gs[0, 1])
        a.scatter(*ad.obsm["X_umap"].T, c=ccolors, s=3, lw=0)
        a.set(title=f"UMAP — {len(classes)} published cell classes (Moffitt 2018)",
              xticks=[], yticks=[])
        for c in classes:
            a.scatter([], [], c=[cmap[c]], s=24, label=c)
        a.legend(fontsize=6.5, ncol=2, loc="upper left", framealpha=0.85,
                 handletextpad=0.2, columnspacing=0.6, markerscale=1.2)

        # (1,0) one coronal slice in tissue space, by published class
        a = fig.add_subplot(gs[1, 0])
        if "Bregma" in ad.obs:
            best = ad.obs["Bregma"].value_counts().idxmax()
            sub = ad[ad.obs["Bregma"] == best]
            sc_colors = np.array([cmap[v] for v in sub.obs[label_key].astype(str)])
            a.scatter(*sub.obsm["spatial"].T, c=sc_colors, s=7, lw=0)
            a.set_title(f"One coronal slice (Bregma {best:+.0f}) — published classes")
        else:
            a.scatter(*ad.obsm["spatial"].T, c=ccolors, s=3, lw=0)
            a.set_title("Tissue space — published classes")
        a.set_aspect("equal"); a.set(xticks=[], yticks=[])

        # (1,1) concordance: Leiden x published class, row-normalized
        a = fig.add_subplot(gs[1, 1])
        ct = pd.crosstab(ad.obs["leiden"], ad.obs[label_key])
        ctn = ct.div(ct.sum(1), axis=0)
        ctn = ctn.iloc[ctn.values.argmax(1).argsort()]
        sns.heatmap(ctn, cmap="viridis", ax=a, yticklabels=False,
                    cbar_kws={"label": "fraction of Leiden cluster", "shrink": 0.6})
        a.set_title(f"Leiden ↔ published concordance  (ARI = {ari:.2f})")
        a.set(xlabel="published cell class", ylabel="Leiden cluster (rows)")
        a.tick_params(axis="x", labelsize=7)
    else:
        a = fig.add_subplot(gs[0, 1])
        a.scatter(*ad.obsm["spatial"].T, c=lcolors, s=3, lw=0)
        a.set_aspect("equal")
        a.set(title="Clusters in tissue space", xticks=[], yticks=[])

    out = ASSETS / f"live_{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="merfish",
                    choices=["merfish", "seqfish", "slideseqv2"])
    ap.add_argument("--fig", action="store_true", help="save a figure to assets/")
    args = ap.parse_args()

    ad = load_dataset(args.dataset)
    metrics = run_pipeline(ad)
    if args.fig:
        metrics["figure"] = make_figure(ad, args.dataset)
    print(json.dumps({"dataset": args.dataset, **metrics}))


if __name__ == "__main__":
    main()

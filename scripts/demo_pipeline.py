"""Runnable demo of the MERFISH single-cell clustering pipeline.

The real notebooks need Vizgen MERSCOPE data from private S3/GCS buckets, so this
script generates a small SYNTHETIC spatial-transcriptomics dataset and runs the
*exact same* Scanpy pipeline the notebooks use — normalize -> log1p -> scale ->
PCA -> neighbors -> UMAP -> Leiden — to demonstrate it end to end and produce
real, computed figures. The data is simulated and clearly labelled as such; only
the pipeline and plots are "real".

Usage:
    pip install scanpy squidpy leidenalg igraph seaborn
    python scripts/demo_pipeline.py     # -> assets/demo_pipeline.png (+ squidpy fig)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

RNG = np.random.default_rng(0)
sc.settings.verbosity = 0

# --- 1. Simulate a spatial transcriptomics dataset ---------------------------
# 6 cell-type "domains", each occupying a region of tissue and expressing its
# own marker-gene program on top of a shared low background.
N_CELLS, N_GENES, K, MARKERS_PER = 3000, 60, 6, 8

domain = RNG.integers(0, K, size=N_CELLS)
lib = RNG.lognormal(mean=0.0, sigma=0.35, size=N_CELLS)          # per-cell depth
marker_of = {k: np.arange(k * MARKERS_PER, (k + 1) * MARKERS_PER) for k in range(K)}

rate = np.full((N_CELLS, N_GENES), 0.3)                          # background
for c in range(N_CELLS):
    rate[c, marker_of[domain[c]]] += RNG.uniform(6, 14)         # marker program
counts = RNG.poisson(rate * lib[:, None]).astype(np.float32)

# spatial coordinates: each domain is a blob in tissue space
centers = RNG.uniform(0, 40, size=(K, 2))
spatial = centers[domain] + RNG.normal(0, 3.2, size=(N_CELLS, 2))

genes = [f"g{j:02d}" for j in range(N_GENES)]
ad = sc.AnnData(counts, obs=pd.DataFrame(index=[f"cell{i}" for i in range(N_CELLS)]),
                var=pd.DataFrame(index=genes))
ad.obsm["spatial"] = spatial
ad.obs["true_domain"] = pd.Categorical(domain.astype(str))

# --- 2. Standard Scanpy pipeline (mirrors the real notebooks) ----------------
sc.pp.normalize_total(ad, target_sum=1e4)
sc.pp.log1p(ad)
ad.raw = ad
sc.pp.scale(ad, max_value=10)
sc.tl.pca(ad, svd_solver="arpack", n_comps=30)
sc.pp.neighbors(ad, n_neighbors=15, n_pcs=30)
sc.tl.umap(ad)
sc.tl.leiden(ad, resolution=1.0, flavor="igraph", n_iterations=2,
             directed=False, random_state=0)
n_leiden = ad.obs["leiden"].nunique()
print(f"Leiden found {n_leiden} clusters from {K} simulated domains")

# --- 3. Figure: UMAP | spatial | marker heatmap ------------------------------
sns.set_theme(style="white", context="talk")
pal = sns.color_palette("tab10", n_leiden)
cats = sorted(ad.obs["leiden"].cat.categories, key=int)
cmap = {c: pal[i] for i, c in enumerate(cats)}
colors = np.array([cmap[v] for v in ad.obs["leiden"].astype(str)])

fig = plt.figure(figsize=(17, 5.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.28)
fig.suptitle("Synthetic demo — MERFISH single-cell clustering pipeline (Scanpy)",
             fontsize=18, fontweight="bold", y=1.02)

ax0 = fig.add_subplot(gs[0])
ax0.scatter(*ad.obsm["X_umap"].T, c=colors, s=8, linewidths=0)
ax0.set(title="UMAP — Leiden clusters", xlabel="UMAP1", ylabel="UMAP2",
        xticks=[], yticks=[])

ax1 = fig.add_subplot(gs[1])
ax1.scatter(*ad.obsm["spatial"].T, c=colors, s=10, linewidths=0)
ax1.set(title="Clusters in tissue space", xlabel="x (µm)", ylabel="y (µm)",
        xticks=[], yticks=[])
ax1.set_aspect("equal")

# marker heatmap: top-2 genes per cluster, mean log-norm expression, z-scored
sc.tl.rank_genes_groups(ad, "leiden", method="wilcoxon", use_raw=True)
top = []
for c in cats:
    for g in [n[c] for n in ad.uns["rank_genes_groups"]["names"][:2]]:
        if g not in top:
            top.append(g)
expr = pd.DataFrame(ad.raw[:, top].X, columns=top)
expr["leiden"] = ad.obs["leiden"].to_numpy()
mean_expr = expr.groupby("leiden", observed=True)[top].mean().reindex(cats)
z = (mean_expr - mean_expr.mean()) / (mean_expr.std() + 1e-9)

ax2 = fig.add_subplot(gs[2])
sns.heatmap(z.T, cmap="viridis", ax=ax2, cbar_kws={"label": "z-scored mean", "shrink": 0.6},
            xticklabels=True, yticklabels=True)
ax2.set(title="Cluster marker genes", xlabel="Leiden cluster", ylabel="")
ax2.tick_params(axis="y", labelsize=8)

for legc in cats:
    ax0.scatter([], [], c=[cmap[legc]], s=40, label=legc)
ax0.legend(title="Leiden", fontsize=9, title_fontsize=10, loc="upper right",
           frameon=False, ncol=2, handletextpad=0.2, columnspacing=0.6)

out = Path(__file__).resolve().parent.parent / "assets" / "demo_pipeline.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")

# --- 4. Modern extension: spatial neighborhood enrichment (squidpy) ----------
# Plain Scanpy clusters in expression space; squidpy adds spatial-aware analysis.
try:
    import squidpy as sq
except ImportError:
    print("squidpy not installed — skipping spatial neighborhood figure "
          "(pip install squidpy)")
else:
    ad.obs["leiden"] = ad.obs["leiden"].astype("category")
    sq.gr.spatial_neighbors(ad, coord_type="generic", n_neighs=6)
    sq.gr.nhood_enrichment(ad, cluster_key="leiden", seed=0, show_progress_bar=False)
    z = ad.uns["leiden_nhood_enrichment"]["zscore"]
    fig2, ax = plt.subplots(figsize=(6.4, 5.2))
    sns.heatmap(z, cmap="RdBu_r", center=0, square=True, linewidths=1,
                linecolor="white", xticklabels=cats, yticklabels=cats,
                cbar_kws={"label": "neighborhood enrichment (z)"}, ax=ax)
    ax.set(title="Spatial neighborhood enrichment (squidpy)",
           xlabel="Leiden cluster", ylabel="Leiden cluster")
    out2 = Path(__file__).resolve().parent.parent / "assets" / "demo_spatial_squidpy.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out2}")

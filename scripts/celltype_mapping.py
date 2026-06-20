"""Principled, confidence-scored cell-type mapping for MERFISH data.

This is the upgrade the methods review flagged as the lowest-friction, highest
-payoff change for this pipeline (see the README): replace the
heuristic *shared-PCA + cosine* label transfer onto the Zeisel taxonomy with the
**MapMyCells / Allen ``cell_type_mapper`` algorithm** — marker-gene correlation with
**bootstrap confidence**.

Why it's better than the cosine heuristic:
  * **Marker-driven** — correlates over differentially-expressed genes per type, not a
    global PCA where housekeeping variance dominates.
  * **Confidence-scored** — bootstraps the marker set (default 100×); the fraction of
    bootstraps agreeing on a call is a probability-like confidence the cosine method
    simply does not produce. Low-confidence cells flag where the panel can't resolve a
    type (exactly what you want to know before trusting a spatial cell-type map).
  * **Hierarchy-ready** — ``map_hierarchical`` maps coarse class first, then refines
    within the winning class (class -> subtype), mirroring the Allen taxonomy levels.

This reimplements the *algorithm* so it runs with no Allen reference download — it builds
the reference from any labelled AnnData. To use the genuine package against the
whole-mouse-brain atlas instead, see ``map_with_cell_type_mapper`` below and
the README for the ABC Atlas WMB reference (CCN20230722).

Usage:
    pip install scanpy squidpy scikit-learn seaborn
    python scripts/celltype_mapping.py --fig          # real Moffitt data + figure
    python scripts/celltype_mapping.py --hierarchical  # also do class->neuron-subtype
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

sc.settings.verbosity = 0
ASSETS = Path(__file__).resolve().parent.parent / "assets"
DATA = Path(__file__).resolve().parent.parent / "data" / "anndata" / "merfish.h5ad"


# --- core algorithm ----------------------------------------------------------

def _dense(x) -> np.ndarray:
    return np.asarray(x.todense()) if hasattr(x, "todense") else np.asarray(x)


def _row_normalize(M: np.ndarray) -> np.ndarray:
    """Center and unit-normalize each row, for fast Pearson via dot product."""
    Mc = M - M.mean(axis=1, keepdims=True)
    n = np.linalg.norm(Mc, axis=1, keepdims=True)
    return Mc / (n + 1e-12)


def _pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation between every row of A (n×m) and B (k×m) -> n×k."""
    return _row_normalize(A) @ _row_normalize(B).T


def log_normalize(ad) -> "sc.AnnData":
    """Total-count normalize + log1p on a copy (the scale the reference is built on)."""
    a = ad.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def build_reference(ad_ref, label_key: str, n_markers: int = 15,
                    method: str = "wilcoxon", min_cells: int = 2) -> dict:
    """Build a marker-gene reference from a labelled, log-normalized AnnData.

    Returns per-type centroids restricted to the union of the top ``n_markers``
    differentially-expressed genes per type — the genes that actually discriminate
    cell types, which is what the correlation should be computed over.

    Types with fewer than ``min_cells`` cells are dropped: a one-cell group has no
    within-group variance for a rank test and is too rare to define a stable centroid.
    """
    a = ad_ref.copy()
    a.obs[label_key] = a.obs[label_key].astype("category")
    counts = a.obs[label_key].value_counts()
    keep = counts[counts >= min_cells].index
    a = a[a.obs[label_key].isin(keep)].copy()
    a.obs[label_key] = a.obs[label_key].cat.remove_unused_categories()
    types = list(a.obs[label_key].cat.categories)

    sc.tl.rank_genes_groups(a, label_key, method=method, n_genes=n_markers)
    names = a.uns["rank_genes_groups"]["names"]
    markers = sorted({g for t in types for g in list(names[t])[:n_markers]})

    sub = a[:, markers]
    X = _dense(sub.X)
    lab = a.obs[label_key].to_numpy()
    centroids = np.vstack([X[lab == t].mean(axis=0) for t in types])
    return {"types": np.array(types), "markers": markers, "centroids": centroids}


def map_to_reference(ad_query, ref: dict, n_bootstrap: int = 100,
                     frac: float = 0.7, seed: int = 0) -> pd.DataFrame:
    """Map query cells onto a reference by bootstrapped marker correlation.

    For each of ``n_bootstrap`` iterations a random ``frac`` of the marker genes is
    drawn and every cell is assigned to its best-correlated type. The final label is
    the modal assignment; ``confidence`` is the fraction of bootstraps that agreed —
    a calibrated, probability-like score the cosine heuristic cannot give you.
    """
    markers = ref["markers"]
    types = ref["types"]
    X = _dense(ad_query[:, markers].X)
    C = ref["centroids"]
    rng = np.random.default_rng(seed)
    m = len(markers)
    k = int(max(2, round(frac * m)))

    votes = np.zeros((X.shape[0], len(types)), dtype=np.int32)
    for _ in range(n_bootstrap):
        idx = rng.choice(m, size=k, replace=False)
        win = _pearson(X[:, idx], C[:, idx]).argmax(axis=1)
        votes[np.arange(X.shape[0]), win] += 1

    assigned = votes.argmax(axis=1)
    confidence = votes[np.arange(X.shape[0]), assigned] / n_bootstrap
    corr_full = _pearson(X, C)
    corr_to_pred = corr_full[np.arange(X.shape[0]), assigned]
    return pd.DataFrame(
        {"pred": types[assigned], "confidence": confidence, "correlation": corr_to_pred},
        index=ad_query.obs_names,
    )


def cosine_baseline(ad_train, ad_query, label_key: str, n_pcs: int = 30) -> np.ndarray:
    """The project's current heuristic: shared-PCA space + cosine to class centroids.

    Reimplemented per-cell for an honest head-to-head against the principled mapper.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity

    Xtr, Xq = _dense(ad_train.X), _dense(ad_query.X)
    nc = int(min(n_pcs, Xtr.shape[1] - 1, Xtr.shape[0] - 1))
    pca = PCA(n_components=nc, svd_solver="arpack", random_state=0).fit(Xtr)
    Ptr, Pq = pca.transform(Xtr), pca.transform(Xq)

    lab = ad_train.obs[label_key].astype("category")
    types = list(lab.cat.categories)
    cents = np.vstack([Ptr[lab.to_numpy() == t].mean(axis=0) for t in types])
    return np.array(types)[cosine_similarity(Pq, cents).argmax(axis=1)]


def evaluate_mapping(ad, label_key: str, test_size: float = 0.5, seed: int = 0,
                     n_bootstrap: int = 100, n_markers: int = 15,
                     with_baseline: bool = False) -> dict:
    """Held-out evaluation: build the reference on a train split, map the test split.

    Using a held-out split (rather than mapping cells with a reference built from those
    same cells) keeps the accuracy honest — it measures generalization, not memorization.
    """
    from sklearn.model_selection import train_test_split

    t0 = time.time()
    a = log_normalize(ad)
    a.obs[label_key] = a.obs[label_key].astype("category")
    y = a.obs[label_key].to_numpy()
    idx = np.arange(a.n_obs)
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)

    ref = build_reference(a[tr], label_key, n_markers=n_markers)
    mapped = map_to_reference(a[te], ref, n_bootstrap=n_bootstrap, seed=seed)
    mapped.insert(0, "truth", y[te])
    mapped["correct"] = mapped["pred"] == mapped["truth"]

    out = {
        "label_key": label_key,
        "n_types": int(len(ref["types"])),
        "n_markers": int(len(ref["markers"])),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "accuracy": round(float(mapped["correct"].mean()), 4),
        "mean_confidence": round(float(mapped["confidence"].mean()), 4),
        "runtime_s": round(time.time() - t0, 1),
        "mapped": mapped,
    }
    if with_baseline:
        pred_b = cosine_baseline(a[tr], a[te], label_key)
        out["cosine_accuracy"] = round(float((pred_b == y[te]).mean()), 4)
    return out


def map_hierarchical(ad, class_key: str, subtype_key: str, test_size: float = 0.5,
                     seed: int = 0, n_bootstrap: int = 100) -> dict:
    """Two-level mapping: coarse class, then refine to subtype within the called class.

    Mirrors the Allen taxonomy levels (class -> subclass/supertype). Subtype accuracy is
    reported only over cells whose coarse class was called correctly.
    """
    from sklearn.model_selection import train_test_split

    a = log_normalize(ad)
    for key in (class_key, subtype_key):
        a.obs[key] = a.obs[key].astype("category")
    yc = a.obs[class_key].to_numpy()
    tr, te = train_test_split(np.arange(a.n_obs), test_size=test_size,
                              random_state=seed, stratify=yc)

    ref_c = build_reference(a[tr], class_key)
    called = map_to_reference(a[te], ref_c, n_bootstrap=n_bootstrap, seed=seed)
    called["truth_class"] = yc[te]
    class_acc = float((called["pred"] == called["truth_class"]).mean())

    # refine: within each predicted class, map to subtype using a class-specific ref
    a_tr, a_te = a[tr], a[te]
    sub_pred = pd.Series(index=called.index, dtype=object)
    sub_conf = pd.Series(index=called.index, dtype=float)
    for cls in ref_c["types"]:
        train_mask = a_tr.obs[class_key].to_numpy() == cls
        test_mask = (called["pred"] == cls).to_numpy()
        if test_mask.sum() == 0 or train_mask.sum() < 5:
            continue
        sub_tr = a_tr[train_mask]
        if sub_tr.obs[subtype_key].nunique() < 2:
            sub_pred[test_mask] = sub_tr.obs[subtype_key].iloc[0]
            sub_conf[test_mask] = 1.0
            continue
        ref_s = build_reference(sub_tr, subtype_key, n_markers=10)
        mp = map_to_reference(a_te[test_mask], ref_s, n_bootstrap=n_bootstrap, seed=seed)
        sub_pred[test_mask] = mp["pred"].to_numpy()
        sub_conf[test_mask] = mp["confidence"].to_numpy()

    called["sub_pred"] = sub_pred
    called["sub_conf"] = sub_conf
    called["truth_sub"] = a.obs[subtype_key].to_numpy()[te]
    ok = called["pred"] == called["truth_class"]
    sub_acc = float((called.loc[ok, "sub_pred"] == called.loc[ok, "truth_sub"]).mean())
    return {"class_accuracy": round(class_acc, 4), "subtype_accuracy": round(sub_acc, 4),
            "n_test": int(len(te)), "called": called}


def map_with_cell_type_mapper(*_args, **_kwargs):  # pragma: no cover - optional path
    """Stub: run the genuine Allen ``cell_type_mapper`` (MapMyCells) against the ABC
    Atlas whole-mouse-brain reference (CCN20230722).

    Install ``cell-type-mapper`` and download the WMB precomputed-stats reference, then
    call ``cell_type_mapper.cli.from_specified_markers``. See the README.
    This native reimplementation runs in offline pytest without that multi-GB download.
    """
    raise NotImplementedError(
        "Install the Allen 'cell-type-mapper' package and the ABC Atlas WMB reference; "
        "see the README.")


# --- data + figure -----------------------------------------------------------

def load_moffitt():
    """Real Moffitt 2018 hypothalamus MERFISH — cached h5ad if present, else squidpy."""
    import anndata as ad_io
    if DATA.exists():
        return ad_io.read_h5ad(DATA)
    import squidpy as sq
    return sq.datasets.merfish()


def make_figure(ad, res: dict, name: str = "celltype_mapping") -> str:
    """Validation figure centred on the new capability: confidence calibration."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = res["mapped"]
    sns.set_theme(style="white", context="talk")
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)
    sub = (f"principled MapMyCells-style mapping — {name} "
           f"({res['n_test']:,} held-out cells · {res['n_types']} types · "
           f"{res['n_markers']} markers · acc={res['accuracy']:.2f})")
    fig.suptitle(sub, fontsize=16, fontweight="bold", y=0.96)

    # (0,0) calibration curve: accuracy rises with confidence
    a = fig.add_subplot(gs[0, 0])
    bins = np.linspace(df["confidence"].min(), 1.0, 9)
    df["_bin"] = pd.cut(df["confidence"], bins, include_lowest=True)
    cal = df.groupby("_bin", observed=True).agg(acc=("correct", "mean"),
                                                n=("correct", "size"))
    centers = [iv.mid for iv in cal.index]
    a.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    a.scatter(centers, cal["acc"], s=np.clip(cal["n"] / cal["n"].max() * 320, 20, 320),
              c="#1D9E75", zorder=3)
    a.plot(centers, cal["acc"], c="#1D9E75", lw=2, zorder=2)
    a.set(title="Confidence is calibrated", xlabel="bootstrap confidence",
          ylabel="accuracy in bin", xlim=(0, 1.02), ylim=(0, 1.02))

    # (0,1) confidence distribution, correct vs incorrect
    a = fig.add_subplot(gs[0, 1])
    for ok, c, lbl in [(True, "#1D9E75", "correct"), (False, "#D85A30", "incorrect")]:
        v = df.loc[df["correct"] == ok, "confidence"]
        if len(v):
            a.hist(v, bins=24, range=(0, 1), alpha=0.7, color=c, label=lbl)
    a.set(title="Confidence: correct vs incorrect calls", xlabel="bootstrap confidence",
          ylabel="cells")
    a.legend(frameon=False)

    # (1,0) confusion matrix, row-normalized
    a = fig.add_subplot(gs[1, 0])
    ct = pd.crosstab(df["truth"], df["pred"])
    ct = ct.reindex(index=ct.sum(1).sort_values(ascending=False).index)
    ctn = ct.div(ct.sum(1), axis=0)
    sns.heatmap(ctn, cmap="viridis", ax=a, cbar_kws={"label": "fraction", "shrink": 0.6},
                vmin=0, vmax=1)
    a.set(title="Truth ↔ predicted (row-normalized)", xlabel="predicted", ylabel="published")
    a.tick_params(axis="x", labelsize=7, rotation=90)
    a.tick_params(axis="y", labelsize=7)

    # (1,1) spatial confidence map (real data with coords + Bregma)
    a = fig.add_subplot(gs[1, 1])
    if "spatial" in ad.obsm and "Bregma" in ad.obs:
        te_names = df.index
        sb = ad[te_names]
        best = sb.obs["Bregma"].value_counts().idxmax()
        m = (sb.obs["Bregma"] == best).to_numpy()
        sp = sb.obsm["spatial"][m]
        scat = a.scatter(sp[:, 0], sp[:, 1], c=df["confidence"].to_numpy()[m],
                         cmap="viridis", s=8, vmin=0, vmax=1, lw=0)
        fig.colorbar(scat, ax=a, shrink=0.6, label="confidence")
        a.set_aspect("equal")
        a.set(title=f"Spatial confidence (Bregma {float(best):+.0f})", xticks=[], yticks=[])
    else:
        a.scatter(df["confidence"], df["correlation"], s=6,
                  c=df["correct"].map({True: "#1D9E75", False: "#D85A30"}), lw=0)
        a.set(title="Confidence vs correlation", xlabel="confidence", ylabel="correlation")

    out = ASSETS / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="Cell_class", help="published label column")
    ap.add_argument("--n-bootstrap", type=int, default=100)
    ap.add_argument("--fig", action="store_true")
    ap.add_argument("--hierarchical", action="store_true",
                    help="also map Cell_class -> Neuron_cluster_ID")
    args = ap.parse_args()

    ad = load_moffitt()
    res = evaluate_mapping(ad, args.label, n_bootstrap=args.n_bootstrap, with_baseline=True)
    report = {k: v for k, v in res.items() if k != "mapped"}
    if args.fig:
        report["figure"] = make_figure(ad, res)
    if args.hierarchical and "Neuron_cluster_ID" in ad.obs:
        h = map_hierarchical(ad, "Cell_class", "Neuron_cluster_ID",
                             n_bootstrap=args.n_bootstrap)
        report["hierarchical"] = {k: v for k, v in h.items() if k != "called"}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

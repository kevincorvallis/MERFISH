"""Generate the polished QC summary figure for the README.

All values below are taken directly from the rendered outputs of
``notebooks/pipeline/qc_rnaseq_correlation.ipynb`` (the MERFISH vs bulk-RNAseq
QC notebook) — total/per-FOV counts and the Pearson r values reported in each
plot title. Nothing here is fabricated; this script just re-renders those real
summary statistics with seaborn for a cleaner, consistent look.

Usage:
    pip install seaborn
    python scripts/qc_figures.py        # -> assets/qc_summary.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Real QC numbers (source: qc_rnaseq_correlation.ipynb outputs) -----------
SAMPLES = ["Young-5", "Middle-4", "Old-4"]
FOV_COUNTS = np.array([2062, 1875, 1885])                  # max(filemap.fov)+1
COUNTS_PER_FOV = np.array([73_715, 87_466, 83_289])        # total / n_fov
TOTAL_COUNTS = FOV_COUNTS * COUNTS_PER_FOV                 # ~1.52 / 1.64 / 1.57e8

# MERFISH vs bulk RNAseq, Pearson r on log10 counts (per-age scatter titles)
RNASEQ_R = np.array([0.70, 0.75, 0.72])

# MERFISH replicate-vs-replicate Pearson r (correlation-grid titles)
REPLICATE_R = pd.DataFrame(
    [[1.00, 0.98, 0.98],
     [0.98, 1.00, 0.99],
     [0.98, 0.99, 1.00]],
    index=SAMPLES, columns=SAMPLES,
)

# --- Style -------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")
AGE_PAL = ["#3B528B", "#21918C", "#5EC962"]   # viridis trio, echoes the hero map
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "font.family": "DejaVu Sans",
})

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle(
    "MERSCOPE QC — MsBrain_VS38 aging cohort",
    fontsize=20, fontweight="bold", y=0.98,
)


def _bars(ax, values, title, ylabel, fmt):
    sns.barplot(x=SAMPLES, y=values, hue=SAMPLES, palette=AGE_PAL,
                legend=False, ax=ax, edgecolor="white", linewidth=1.5)
    ax.set(title=title, ylabel=ylabel, xlabel="")
    ax.margins(y=0.18)
    for p, v in zip(ax.patches, values):
        ax.annotate(fmt(v), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=13, fontweight="bold",
                    xytext=(0, 3), textcoords="offset points")
    sns.despine(ax=ax)


# (0,0) total transcript counts
_bars(axes[0, 0], TOTAL_COUNTS / 1e6,
      "Total transcript counts", "millions of transcripts",
      lambda v: f"{v:.0f}M")

# (0,1) mean counts per FOV
_bars(axes[0, 1], COUNTS_PER_FOV / 1e3,
      "Mean counts per field of view", "counts / FOV (thousands)",
      lambda v: f"{v:.0f}k")

# (1,0) replicate correlation heatmap
hm = sns.heatmap(REPLICATE_R, annot=True, fmt=".2f", cmap="crest",
                 vmin=0.96, vmax=1.0, square=True, linewidths=2,
                 linecolor="white", cbar_kws={"label": "Pearson r", "shrink": 0.8},
                 annot_kws={"fontweight": "bold"}, ax=axes[1, 0])
axes[1, 0].set_title("Replicate reproducibility")
hm.set_xticklabels(hm.get_xticklabels(), rotation=0)

# (1,1) MERFISH vs bulk RNAseq r
sns.barplot(x=SAMPLES, y=RNASEQ_R, hue=SAMPLES, palette=AGE_PAL, legend=False,
            ax=axes[1, 1], edgecolor="white", linewidth=1.5)
axes[1, 1].set(title="MERFISH vs bulk RNAseq", ylabel="Pearson r (log counts)",
               xlabel="", ylim=(0, 1.0))
for p, v in zip(axes[1, 1].patches, RNASEQ_R):
    axes[1, 1].annotate(f"r = {v:.2f}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=13, fontweight="bold",
                        xytext=(0, 3), textcoords="offset points")
sns.despine(ax=axes[1, 1])

fig.tight_layout(rect=(0, 0, 1, 0.96))

out = Path(__file__).resolve().parent.parent / "assets" / "qc_summary.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")

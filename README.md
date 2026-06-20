# 🧬 MERFISH Mouse Brain Atlas

> Single-cell-resolution spatial transcriptomics on the intact mouse brain — from raw Vizgen MERSCOPE output to interactive, cell-type-mapped spatial heatmaps.

![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![scanpy](https://img.shields.io/badge/scanpy-single--cell-1B998B)
![MERSCOPE](https://img.shields.io/badge/Vizgen-MERSCOPE-6E44FF)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**MERFISH** images individual RNA molecules *in situ* — reading out hundreds of genes per cell while keeping each transcript's exact position in intact tissue. This repo analyzes Vizgen **MERSCOPE** mouse-brain data end to end: load raw imagery + transcripts → QC against bulk RNAseq → **Scanpy** clustering (PCA / UMAP / Leiden) → cell-type mapping onto the Zeisel taxonomy → interactive **Observable** dashboards.

![Spatial transcript-density map of a coronal mouse-brain section](assets/hero_coronal_brain_spatial_map.png)

<p align="center"><em>A full coronal section reconstructed purely from spatial transcript positions — anatomy recovered from RNA alone.</em></p>

## 🔄 Pipeline

```mermaid
flowchart LR
    A["Raw MERSCOPE output<br/>mosaics · transcripts<br/>cell-by-gene · boundaries"] --> C["QC vs bulk RNAseq<br/>Pearson r on log counts"]
    A --> D["Build AnnData<br/>filter cells · drop Blanks"]
    D --> E["Scanpy clustering<br/>PCA · UMAP · Leiden"]
    E --> F["Cell-type mapping<br/>Zeisel taxonomy"]
    E --> G["Clustergrammer2 heatmap"]
    F --> H["Interactive Observable<br/>UMAP · spatial · heatmap"]
    G --> H
    C -. validates .-> E
```

## 📊 QC results

MERFISH counts track an orthogonal assay and reproduce across the `MsBrain_VS38` aging cohort (Young / Middle / Old).

| | |
|---|---|
| ![](assets/qc_merfish_vs_rnaseq.png) | ![](assets/qc_replicate_correlation.png) |
| **MERFISH vs bulk RNAseq** — Pearson `r = 0.70 / 0.75 / 0.72`. | **Replicate reproducibility** — `r = 0.98–1.00`, count ratio ≈ 1.0. |
| ![](assets/qc_total_counts_per_sample.png) | ![](assets/qc_counts_per_fov.png) |
| **Total counts per sample** — ~1.5–1.65 × 10⁸ transcripts each. | **Counts per FOV** — ~74k–87k, consistent yield. |

## 📓 Notebooks

| Notebook | What it does |
|---|---|
| [`notebooks/showcase_mouse_brain.ipynb`](notebooks/showcase_mouse_brain.ipynb) | Canonical end-to-end Vizgen showcase (public GCS data): AnnData 83,546 × 483, Leiden + cell-type map, Observable dashboards |
| [`notebooks/broad_local_adaptation.ipynb`](notebooks/broad_local_adaptation.ipynb) | Local (non-Colab) adaptation reading MERSCOPE output from external SSDs |
| [`notebooks/transcript_viz_prototype.ipynb`](notebooks/transcript_viz_prototype.ipynb) | Minimal prototype: load output → render selected gene transcripts in Observable |
| [`notebooks/pipeline/qc_rnaseq_correlation.ipynb`](notebooks/pipeline/qc_rnaseq_correlation.ipynb) | MERFISH ↔ bulk RNAseq + replicate correlation QC (the plots above) |
| [`notebooks/pipeline/umap_spatial_heatmap_v0.3.1.ipynb`](notebooks/pipeline/umap_spatial_heatmap_v0.3.1.ipynb) | Single-cell viz compiler: S3 matrix → UMAP/Leiden → embedded dashboard |
| [`notebooks/pipeline/transcripts_genes_of_interest_v0.2.0.ipynb`](notebooks/pipeline/transcripts_genes_of_interest_v0.2.0.ipynb) | Lightweight transcript viewer for hand-picked genes of interest |

## 🗂️ Structure

```text
MERFISH/
├── notebooks/
│   ├── showcase_mouse_brain.ipynb        # canonical end-to-end showcase
│   ├── broad_local_adaptation.ipynb      # local adaptation (external SSDs)
│   ├── transcript_viz_prototype.ipynb    # transcript-viz prototype
│   └── pipeline/                         # modular, versioned stages
│       ├── qc_rnaseq_correlation.ipynb
│       ├── umap_spatial_heatmap_v0.3.1.ipynb
│       └── transcripts_genes_of_interest_v0.2.0.ipynb
└── assets/                               # extracted figures
```

## 🚀 Quick start

```bash
pip install scanpy leidenalg loompy clustergrammer2 observable_jupyter \
            tifffile opencv-python h5py matplotlib pandas numpy scipy scikit-learn fsspec gcsfs
jupyter lab        # open notebooks/showcase_mouse_brain.ipynb
```

Demo data (Vizgen public release): `gs://public-datasets-vizgen-merfish/datasets/mouse_brain_map/BrainReceptorShowcase/`. Point each notebook's `base_path` / `dataset_path` at your local copy or bucket — raw MERSCOPE output (`*.tif`, `*.hdf5`, large `*.csv`) is git-ignored. The QC notebook also needs Vizgen's proprietary `merlin` / `encoder.abundance` packages.

## 🙏 Credits

Built on [Vizgen MERSCOPE](https://vizgen.com), the [Zeisel et al.](http://mousebrain.org) scRNAseq taxonomy, [Scanpy](https://scanpy.readthedocs.io), [Clustergrammer2](https://clustergrammer.readthedocs.io), and Observable. Released under the **MIT License**.

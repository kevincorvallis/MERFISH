# Methods review (2026)

Concise guide to the principled upgrades recommended for this MERFISH pipeline.
Each item has a **dependency-light core** (offline pytest, no large downloads) and an
optional **production backend** (real reference atlas or registration engine).

## Cell-type mapping

**Problem:** the production notebooks map Leiden clusters to the Zeisel taxonomy via
shared-PCA + cosine distance. That assigns a label but carries no confidence and can
mis-call clusters the panel cannot resolve.

**Upgrade:** MapMyCells / Allen `cell_type_mapper` — marker-gene correlation with
bootstrap confidence.

| Layer | Status | Where |
|---|---|---|
| Core algorithm (marker correlation + bootstrap confidence + hierarchical refine) | Implemented | [`scripts/celltype_mapping.py`](../scripts/celltype_mapping.py) |
| Genuine Allen package against ABC whole-mouse-brain reference | Stubbed (`map_with_cell_type_mapper`) | needs `cell-type-mapper` + multi-GB WMB stats |

**Use in notebooks:** after Leiden clustering, call `celltype_mapping.map_to_reference` or
`map_hierarchical` instead of the cosine heuristic in
[`broad_local_adaptation.ipynb`](../notebooks/broad_local_adaptation.ipynb).

Held-out validation on real Moffitt 2018 hypothalamus MERFISH: **0.77** vs **0.71**
accuracy vs the cosine heuristic, with calibrated per-cell confidence (mean **0.87**).

## Segmentation

**Problem:** cell segmentation builds the cell-by-gene matrix upstream of PCA/UMAP/Leiden.
Vendor-default Voronoi/expansion methods misassign boundary transcripts and shift downstream
clusters ([*Segmentation Matters*](https://www.biorxiv.org/content/10.1101/2025.08.25.672145v1)).

**Upgrade:** transcript-aware segmentation (Baysor / proseg / segger style) or deep-learning
methods (Cellpose-SAM).

| Layer | Status | Where |
|---|---|---|
| Simulated benchmark (nucleus / Voronoi / transcript-aware) | Implemented | [`scripts/segmentation_demo.py`](../scripts/segmentation_demo.py) |
| Cellpose-SAM on real DAPI mosaics | Hook only (`cellpose_sam_segment`) | needs `cellpose>=4` |
| Production MERSCOPE tools (proseg, RNA2seg, segger) | Documented, not wired | see README |

On simulated ground truth: transcript-aware segmentation improves transcript-assignment
accuracy (**0.76** vs **0.70**) and downstream Leiden ARI (**1.00** vs **0.96**).

## Atlas registration & per-cell region labels

**Problem:** placing a new 2D section in a common anatomical frame — "which Allen CCFv3
region is each cell in?"

**Upgrade:** DeepSlice affine anchor → STalign / ANTs deformable warp → per-cell annotation
lift-over, plus calibrated uncertainty and QC.

| Layer | Status | Where |
|---|---|---|
| Geometry, label transfer, ensemble confidence, QC cross-check | Implemented | [`scripts/atlas_registration.py`](../scripts/atlas_registration.py) |
| Real Allen CCFv3 volume load | Implemented (`load_ccf_brainglobe`) | needs `brainglobe-atlasapi` |
| DeepSlice affine anchoring | Stubbed (`deepslice_anchor`) | needs `DeepSlice` |
| STalign LDDMM / ANTs SyN deformable warp | Implemented / optional live test | unified dev env (`--no-deps` STalign install) |

Full design, citations, and **honest limitations**:
[`docs/atlas-registration-2026.md`](atlas-registration-2026.md).

> **Limitation:** validated accuracies on real CCFv3 geometry use synthetic anchoring error
> (ground truth known), not a real DeepSlice/STalign fit. The core layers are proven; end-to-end
> registration-engine accuracy is not yet benchmarked here.

## Testing

```bash
pip install -r requirements-dev.txt
pytest -m "not live"    # 20 offline tests — no network
pytest                  # + 5 live tests (squidpy download, brainglobe CCFv3, STalign)
```

Offline tests run in GitHub Actions (`.github/workflows/test.yml`).

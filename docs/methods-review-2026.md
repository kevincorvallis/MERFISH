# Methods review — what's new in spatial transcriptomics (2025–2026)

> Research pass over the newest experimental and computational methods relevant to this
> MERFISH mouse-brain pipeline. Each claim below was adversarially verified (3-vote: a
> finding is kept only if it survives skeptical re-checking) against the cited primary
> sources. Confidence and vote tallies are stated explicitly. Produced 2026-06-19.

This extends the README's **Spatial transcriptomics in context** / **Modern extensions**
sections (README lines 144–191). Most of the *platform* context there is confirmed and up
to date — the genuinely **new, actionable** material is the segmentation tooling, the
segmentation→clustering evidence, SPLIT, MapMyCells' algorithms, and the ABC Atlas
whole-mouse-brain taxonomy as a concrete Zeisel replacement.

---

## The two highest-leverage changes for *this* pipeline

1. **Swap cell segmentation** upstream of the count matrix → `proseg`, `Cellpose-SAM`, or
   `RNA2seg`, all of which natively accept MERSCOPE/MERFISH input.
2. **Re-map cell types** onto the **Allen Brain Cell (ABC) Atlas** whole-mouse-brain
   taxonomy via **MapMyCells**, replacing the heuristic Zeisel shared-PCA + cosine step in
   [`live_test.py`](../scripts/live_test.py) / the showcase notebook.

Everything else below is supporting context or a medium-confidence add-on.

---

## 1. Cell segmentation — the most concretely upgradeable step

`high confidence (3-0)`

> 🔬 **Demonstrated** in [`scripts/segmentation_demo.py`](../scripts/segmentation_demo.py)
> (tests in `tests/test_segmentation.py`). On simulated molecule-level data with known
> ground truth, the modern **transcript-aware** paradigm (Baysor/proseg/segger-style)
> beats the Voronoi/expansion baseline on transcript-assignment accuracy (**0.76** vs
> **0.70**) and downstream Leiden ARI (**1.00** vs **0.96**) — directly reproducing the
> "Segmentation Matters" finding. A guarded `cellpose_sam_segment()` hook runs the genuine
> Cellpose-SAM on real DAPI mosaics; production swap on real MERSCOPE output is
> `proseg --merscope`.

Segmentation sits *upstream* of your entire `PCA → UMAP → Leiden → cell-type mapping`
flow, and your pipeline currently inherits Vizgen's default Cellpose boundaries (or the
prebuilt cell-by-gene matrix). Four new generalist methods explicitly support
MERSCOPE/MERFISH:

| Method | What it is | MERSCOPE support | Status |
|---|---|---|---|
| **proseg** | Probabilistic / Bayesian-sampling segmentation ("cell simulation as cell segmentation") | native `proseg --merscope detected_transcripts.csv.gz` preset | *Nature Methods* 2025 (peer-reviewed) |
| **Cellpose-SAM** | Adapts SAM's ViT-L transformer backbone; near/super-human accuracy | vendor-endorsed swap inside Vizgen's VPT (plugin) | bioRxiv 2025.04 (shipping in Cellpose 4.x) |
| **RNA2seg** | Fuses RNA point clouds + DAPI/membrane stains; trained on 4M MERFISH+CosMx cells; ships a **brain checkpoint** | yes (MERFISH in training set) | *Genome Biology* 2025 (peer-reviewed) |
| **segger** | Attention GNN; segmentation as transcript→cell **link prediction**; **~30× faster than Baysor** | any subcellular iST incl. MERFISH | bioRxiv 2025.03 |

**Why it matters here (verified, `2-1`):** the *Segmentation Matters* benchmark (bioRxiv
2025.08) shows segmentation choice measurably reshapes downstream clusters — a manually
defined cluster was *split* under Cellpose/DBSCAN/YOLO but stayed 1:1 under a combined
method, and one cluster was *missed entirely* by YOLO. This is direct evidence that the
step feeding your count matrix changes your Leiden clusters and cell-type calls.

**Adoption friction (honest caveats):**
- `proseg` and `segger` **re-segment from raw transcripts** — adopting them moves your
  pipeline's *entry point* from the prebuilt cell-by-gene matrix to `detected_transcripts`.
- `Cellpose-SAM` is genuinely **drop-in** at the install/API level, but "drop-in" is not a
  guarantee of *better* results on dense/elongated brain cells — validate on your data.
- Most segmentation evidence comes from **10x Xenium / human tissue**, not MERSCOPE mouse
  brain. The *tools* accept MERSCOPE input; the relative *accuracy rankings* don't
  automatically transfer.
- RNA2seg's zero-shot brain result was on *hamster*; authors recommend cheap few-shot
  fine-tuning for a new tissue.

> **Do NOT claim** RNA2seg beats all of Baysor/Cellpose/ComSeg/GeneSegNet/proseg — that
> specific claim was **refuted (1-2)**.

Sources: [proseg](https://github.com/dcjones/proseg) ·
[Nature Methods 2025](https://www.nature.com/articles/s41592-025-02697-0) ·
[Cellpose-SAM (bioRxiv)](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1) ·
[RNA2seg (Genome Biology)](https://link.springer.com/article/10.1186/s13059-025-03908-9) ·
[segger (bioRxiv)](https://www.biorxiv.org/content/10.1101/2025.03.14.643160v1) ·
[Segmentation Matters](https://www.biorxiv.org/content/10.1101/2025.08.25.672145v1)

---

## 2. Re-map cell types onto the ABC Atlas, via MapMyCells

`high confidence (3-0)`

> ✅ **Implemented** in [`scripts/celltype_mapping.py`](../scripts/celltype_mapping.py) — a
> faithful, dependency-light reimplementation of the MapMyCells algorithm (marker-gene
> correlation + bootstrap confidence + optional hierarchy), validated on held-out real
> Moffitt cells: **0.77** accuracy vs **0.71** for the cosine heuristic, mean confidence
> **0.87**, hierarchical class→subtype **0.82**. Tested in `tests/test_mapping.py`. The
> genuine `cell_type_mapper` against the real ABC Atlas WMB reference is the next step
> (see `map_with_cell_type_mapper` stub + below).

Your current mapping (Zeisel 2018, ~265 types, ~500k cells) can be replaced with a
**~20× larger, ~5-years-newer** reference: the BICCN/ABC Atlas **whole-mouse-brain (WMB)
taxonomy** — `CCN20230722`: **34 classes / 338 subclasses / 1,201 supertypes / 5,322
clusters** from ~4M cells (2023).

**MapMyCells** (Allen Institute) provides three *principled* algorithms to replace the
ad-hoc shared-PCA + cosine heuristic:
- **Flat correlation** — one-step nearest cluster-centroid by correlation.
- **Hierarchical** — marker-gene mapping with 100-iteration bootstrap likelihood scoring
  down the taxonomy tree (gives confidence per level).
- **Deep generative** — conditional VAE + MLP classifier.

It exposes WMB `CCN20230722` as a selectable reference and accepts spatial input. This is
**complementary to** the cell2location / Tangram / scANVI direction the README already
flags (line 178) — MapMyCells is the *atlas-native* option; the scvi-tools methods are the
*probabilistic deconvolution* option.

> **Do NOT overstate:** the claim that MapMyCells was specifically *validated* on MERFISH
> (4.3M cells) + STARmap-PLUS was **refuted (1-2)**. Say it maps onto mouse-brain
> taxonomies and accepts spatial input — not that it's MERFISH-validated.

> Also **refuted (0-3):** that the WMB atlas was built by integrating ~4M scRNA-seq with
> ~4.3M MERFISH cells. Don't assert that specific construction recipe.

Sources: [MapMyCells / cell_type_mapper (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13060854/) ·
[cell_type_mapper (GitHub)](https://github.com/AllenInstitute/cell_type_mapper) ·
[Yao et al. 2023, *Nature* WMB atlas](https://www.nature.com/articles/s41586-023-06812-z) ·
[ABC Atlas access docs](https://alleninstitute.github.io/abc_atlas_access/)

---

## 3. Transcript-spillover correction — SPLIT

`high confidence (3-0)`

**SPLIT** (*Spatial Purification of Layered Intracellular Transcripts*, *Nature Methods*
2026) pairs snRNA-seq with deconvolution (RCTD, or in v0.2.0+ *any* cells×cell-types
weight matrix) to correct transcript spillover and sharpen cell-type specificity. It is
**segmentation-agnostic** — it works *with* whatever segmentation you choose in §1.

**Caveats:** it's an **R package** (your pipeline is Python/scanpy → needs `rpy2`/an R
bridge or a reimplementation), and it was validated on **Xenium**, not MERSCOPE — though
both are imaging ST with the same spillover problem.

Related panel lesson (`3-0`): Xenium's 5K whole-transcriptome-scale panel captures *more*
transcripts but has **reduced per-gene sensitivity and persistent diffusion** — i.e., a
bigger panel does **not** eliminate contamination. Relevant if you ever weigh expanding
your ~500-gene panel.

Sources: [SPLIT (*Nature Methods* 2026)](https://www.nature.com/articles/s41592-026-03089-8) ·
[SPLIT (GitHub)](https://github.com/bdsc-tds/SPLIT)

---

## 4. Platform / chemistry context (mostly confirms the README)

- **MERSCOPE Ultra runs MERFISH 2.0**, ~1,000-gene ceiling (18-bit/140, 21-bit/300,
  27-bit/1000). `medium (2-1)` — vendor spec, not an independent benchmark; already in
  README line 150/165.
- **Two independent 2025 *Nature Communications* benchmarks** now compare imaging
  platforms head-to-head on **human FFPE tumors** (Xenium vs MERSCOPE vs CosMx, 263 TMA
  cores / >5M cells / ~395M transcripts; and Xenium-5K vs CosMx-6K vs Stereo-seq vs Visium
  HD). `high (3-0)`. Useful for *platform justification*, but **neither uses mouse brain**.
- **CosMx 6K** detects more transcripts but shows **elevated negative-control background**
  (higher Moran's I aggregation); **Xenium 5K** keeps a lower negative-control fraction.
  `high (3-0)` — but this benchmark **did not test MERSCOPE** on these axes.

> **Do NOT claim** MERSCOPE is the least-sensitive platform / ~20× below Xenium — that was
> **refuted (1-2)**.

Sources: [MERSCOPE Ultra](https://vizgen.com/merscope-ultra/) ·
[Benchmark i (*Nat Commun*)](https://www.nature.com/articles/s41467-025-64990-y) ·
[Benchmark ii (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12534522/)

---

## 5. Refuted claims — do not repeat these

Surfaced during search but **killed** under adversarial verification:

1. MERSCOPE is the *least sensitive* imaging platform (~20× below Xenium). `1-2`
2. RNA2seg *outperforms all* of Baysor/Cellpose/ComSeg/GeneSegNet/proseg. `1-2`
3. MapMyCells was *validated on MERFISH* (4.3M cells) + STARmap-PLUS. `1-2`
4. The WMB atlas was built by integrating ~4M scRNA-seq with ~4.3M MERFISH cells. `0-3`
5. A scanpy `score_genes` marker-scoring baseline *outperforms* reference deconvolution
   (cell2location/RCTD/CARD/…) for rare cell types without a reference. `0-3` — so a
   lightweight marker-scoring shortcut is **not** a justified substitute for principled
   reference mapping.

---

## 6. Gaps this pass did *not* cover (open questions)

The verified findings clustered on segmentation, reference mapping, and contamination.
These README "Modern extensions" topics were **not** covered by confirmed claims and are
worth a dedicated follow-up:

- **Spatial domain discovery** — CellCharter, BANKSY, and successors (2025–26 state).
- **Spatially variable genes** — SPARK-X / SpatialDE vs `squidpy` Moran's I; current
  consensus.
- **Spatially-resolved cell–cell communication** — LIANA+ and alternatives.
- **Spatial foundation models** for imaging ST (some candidate sources were fetched but no
  claim cleared verification).
- **Has anyone benchmarked proseg/Cellpose-SAM/RNA2seg/segger specifically on MERSCOPE
  *mouse brain*?** Current rankings are extrapolated from Xenium/human tissue.

---

## Suggested sequencing

1. ✅ **MapMyCells-style re-mapping** — *done* (`scripts/celltype_mapping.py`): the
   algorithm, validated on real data, beating the cosine heuristic with added confidence.
   Remaining: point it at the genuine ABC Atlas WMB reference via `cell_type_mapper`.
2. 🔬 **Segmentation swap** — *demonstrated* (`scripts/segmentation_demo.py`): transcript
   -aware beats the morphology baseline downstream. Remaining: run the genuine Cellpose-SAM
   (`cellpose>=4`) / `proseg --merscope` on real MERSCOPE mosaics + transcripts.
3. **proseg `--merscope`** if you want a fully transcript-aware re-segmentation and are
   willing to move the entry point to `detected_transcripts`.
4. **SPLIT** only if spillover proves to be a real problem in your data (needs R interop).

# Atlas registration — aligning new MERFISH sections to the Allen CCFv3 (2025–2026)

> Research pass over the state of the art for **registering 2D mouse-brain MERFISH
> sections into the Allen Common Coordinate Framework v3 (CCFv3)** and assigning every
> segmented cell a brain-region label. Each claim below was adversarially verified
> (3-vote: kept only if it survives skeptical re-checking) against the cited primary
> sources; confidence and vote tallies are stated explicitly. Produced 2026-06-20.
>
> **Scope** (per the project's needs): mouse, **2D sections → 3D CCFv3**, focus on the
> *registration + per-cell region-assignment* problem. Segmentation and cell-type calls
> are assumed to already exist (they do — see [`celltype_mapping.py`](../scripts/celltype_mapping.py)).

This is the gap the README's *Modern extensions* flagged as open: *"map onto the BICCN /
Allen Brain Cell Atlas whole-mouse-brain"*. The problem you hit — "it's hard to align the
separate brain regions on a new scan" — is, precisely, **2D-section-to-3D-atlas
registration**, and the field has converged on a clear, layered solution.

---

## The recommended architecture, in one line

**DeepSlice** (automated affine anchoring + cut-angle estimation) → **STalign** LDDMM
(molecular-aware diffeomorphic 2D→3D warp to CCFv3) → **per-cell annotation lift-over**,
with **ANTs 2.5D** as the proven classical baseline (it built the reference data),
**ABBA/Fiji** as the human-in-the-loop fallback, and **moscot/PASTE2/CAST** for
cross-sample consensus and cross-validation against the Allen ABC reference.

The genuinely *novel* layer — and the one with **no off-the-shelf answer** — is
**calibrated per-cell uncertainty + QC**. That is where this repo's existing strength
(bootstrap-calibrated confidence in `celltype_mapping.py`) becomes a real contribution.

---

## 1. Image-based section → CCFv3 registration (the mature core)

### DeepSlice — automated affine first stage; solves the cut-angle problem natively
`high confidence (3-0)`

DeepSlice is a CNN (Xception backbone) that regresses **nine values** — the anchoring
vectors `Oxyz, Uxyz, Vxyz` giving the CCF coordinates of a section image's corners. An
origin + two edge vectors *geometrically defines an arbitrarily-oriented plane*, so an
**obliquely-cut section is represented natively** — the cut angle isn't a nuisance
parameter, it's part of the output. It predicts dorsoventral & mediolateral cutting
angles (highly correlated with ground truth) and an **"Angle Integration"** post-step that
averages per-section angles across a serial set and normalizes each to the shared
average (serial sections share one cut angle).

- Accuracy **equivalent to human experts** (median error **6.6 ± 1.7 voxels** vs 7.8 ± 2.2),
  at **~74 ms/section** (a 50-section set in <4 s) — a >1000× speedup.
- **Affine/linear only** — it does *not* predict non-linear deformation; that is delegated
  downstream. Outputs QuickNII/VisuAlign-compatible JSON.
- **Limitations (present-tense):** coronal-only; best on brightfield (a real caveat for
  fluorescence MERFISH — render a DAPI/transcript-density grayscale, or use ABBA which
  lifts the coronal-only restriction).

Sources: [Carey et al. 2023 *Nat Commun*](https://www.nature.com/articles/s41467-023-41645-4) ·
[QUINT/DeepSlice docs](https://quint-workflow.readthedocs.io/en/latest/DeepSlice.html) ·
[DeepSlice GitHub](https://github.com/PolarBean/DeepSlice)

### The classical two-stage chains: QUINT and brainreg
`high confidence (3-0)`

The established pattern is **affine, then deformable**, in two explicit stages:

- **QUINT**: **QuickNII** does user-guided affine anchoring of a 2D image into CCFv3, then
  **VisuAlign** applies user-guided **non-linear *in-plane*** refinement. Supports CCFv3
  (2017/2015 delineations), sections cut at **any orientation angle** (manual oblique-plane
  definition), and partial/distorted/damaged sections.
- **brainreg** (BrainGlobe): a three-step pipeline — reorientation → affine → **freeform
  B-spline** — whose **sole backend is the classical NiftyReg** (so it's a classical, not
  deep-learning, baseline). Once registered, **atlas region annotations are warped into
  sample space** — this *is* the per-cell label-transfer mechanism.

Caveats: VisuAlign's nonlinear warp is **2D in-plane** (not full 3D deformable) and
optional; brainreg targets **3D whole-brain volumes**, so it is less ideal for a *single
2D section* than DeepSlice/STalign.

Sources: [QUINT](https://quint-workflow.readthedocs.io/en/latest/QUINTintro.html) ·
[QuickNII](https://quicknii.readthedocs.io) · [VisuAlign](https://visualign.readthedocs.io) ·
[brainreg](https://brainglobe.info/documentation/brainreg/index.html) ·
[NiftyReg](https://github.com/KCL-BMEIS/niftyreg)

### ABBA — the human-in-the-loop front-end that wraps DeepSlice
`high confidence (3-0)`

**ABBA** (EPFL BIOP, Fiji/ImageJ plugin) registers 2D serial sections cut in **any
orientation** (coronal/sagittal/horizontal) to the Allen Mouse CCF, the Waxholm rat atlas,
and **all BrainGlobe atlases**. It **calls DeepSlice** for ML positioning + cut-angle +
in-plane affine, then lets you **manually refine** (BigWarp/Elastix) — a step *not*
available inside DeepSlice. This is the strongest interactive alternative to QUINT and it
lifts standalone DeepSlice's coronal-only restriction.

Sources: [Allen registration toolkit](http://portal.brain-map.org/explore/toolkit/community-tools/computational-tools/data-registration-tools) ·
[ABBA docs](https://abba-documentation.readthedocs.io) ·
[ABBA GitHub](https://github.com/BIOP/ijp-imagetoatlas) ·
[Chiaruttini et al. 2025 *Cell Reports* (ABBA+BraiAn)](https://doi.org/10.1016/j.celrep.2025.115876)

### HERBS and the general paradigm
`high confidence (3-0)`

**HERBS** does rigid + **interactive landmark-based piecewise-affine** warping (manual
anchors), handling oblique cuts by letting the user **tilt the atlas up to 30°** (manual,
not learned). More usefully, it crisply frames the whole task: *atlas-based analysis =
spatially register data to the atlas, assign an anatomical region to each
pixel/voxel/cell, then transfer region labels* — exactly the section→volume→per-cell-label
workflow we're building.

Sources: [Liu et al. 2023 *eLife*](https://elifesciences.org/articles/83496) ·
[Kleven et al. 2023 *Front Neuroinform*](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2023.1154080/full)

---

## 2. Molecular / expression-aware registration (the modern edge)

### STalign — LDDMM diffeomorphic alignment of single-cell MERFISH to the 3D CCF
`high confidence (3-0)`

**STalign** is the recommended **open-source, molecular-aware deformable core**. It builds
on **LDDMM** with **image varifolds**, composing an **affine `A` + a diffeomorphism `φ`**
(the affine→deformable path in *one* framework). It rasterizes single-cell positions into
images (driven by cell/expression signal, *not* only DAPI/Nissl) and was applied to align
**9 adult-mouse-brain MERFISH datasets to a 3D 50 µm grayscale CCF**. It explicitly handles
**both** the global 2D-plane-vs-3D-volume placement (oblique/cut-angle — best-fit plane was
found tilted **~9°**) **and** local in/out-of-plane distortion, and **lifts CCF annotations
onto each cell** (`analyze3Dalign` → per-cell `struct_id`/acronym).

Caveats: the tutorial's slice index / starting rotation are partly **user-initialized**; no
one-to-one cell correspondence; harder on thin structures.

Sources: [Clifton et al. 2023 *Nat Commun*](https://www.nature.com/articles/s41467-023-43915-7) ·
[STalign MERFISH↔3D-CCF tutorial](https://jef.works/STalign/notebooks/merfish-allen3Datlas-alignment.html)

### CAST — expression-driven GNN registration (cross-sample)
`high confidence (3-0)` core · `2-1` on the "more accurate than image-based" sub-claim

**CAST** (Tang et al. 2024 *Nat Methods*) is a **graph neural network** with three modules:
**Mark** (self-supervised graph-contrastive region identification), **Stack** (physical
alignment: **gradient-descent affine → B-spline free-form deformation**), **Projection**
(single-cell label transfer). It needs **only gene expression + spatial coordinates** (no
DAPI/histology/cell-type labels) and works across technologies incl. MERFISH.

**Important:** CAST aligns sections to **another spatial sample**, not to an external 3D
atlas out of the box — CCF use means designating a CCF-registered sample as the reference.
The "may be more advantageous and accurate than image-based registration" line is a
**hedged motivational claim (2-1)**, not a benchmark. Best used here for **replicate/
cross-sample consensus**, not native section→volume.

Sources: [Tang et al. 2024 *Nat Methods*](https://www.nature.com/articles/s41592-024-02410-7) ·
[CAST GitHub](https://github.com/wanglab-broad/CAST)

### Optimal transport — moscot, PASTE2 (consensus + cross-validation)
`high confidence (3-0)`

- **moscot** (Klein et al. 2025 *Nature*) aligns multiple coronal mouse-brain sections to a
  reference slide and builds a **consensus** view via **(fused) Gromov-Wasserstein OT**
  (demonstrated on real MERSCOPE).
- **PASTE2** does **partial** pairwise alignment — only a subset of spots/cells need match —
  directly handling **partial z-overlap and slice-specific cell types** (which the original
  PASTE cannot). Core: **partial Fused Gromov-Wasserstein**, solved by Frank-Wolfe.

In both, the "reference" is **another ST slide, not CCFv3** — so these support
**replicate consensus and cross-checking**, not direct atlas registration.

Sources: [Klein et al. 2025 *Nature* (moscot)](https://www.nature.com/articles/s41586-024-08453-2) ·
[PASTE2 (bioRxiv/Genome Res)](https://www.biorxiv.org/content/10.1101/2023.01.08.523162.full.pdf) ·
[PASTE2 GitHub](https://github.com/raphael-group/paste2)

---

## 3. Allen prior art — how the reference data was actually made

### The Allen production pipeline used ANTs 2.5D, *not* STalign
`high confidence (3-0)` on the contrast · `attributed via verifier evidence` on exact scheme

The Allen Institute's own whole-brain **MERFISH→CCFv3** registration (Yao/Zhuang et al.
2023) used a **custom ANTs-based 2.5D pipeline**: **3D global affine → per-section 2D
affine → 2D SyN diffeomorphic**. So **ANTs (and SimpleElastix/SimpleITK) is the *proven
classical baseline* that produced the reference ABC Atlas data** — STalign is a capable
open-source *alternative*, not the method the reference was registered with. The resulting
ABC data carries **`x,y,z` CCF coordinates + a `parcellation_index`** (region label) **per
cell** — exactly the section→volume→per-cell output we target.

> **Honest caveat:** the exact ANTs parameters live in the Yao/Zhuang 2023 *Nature* methods
> and the `allensdk`/`abc_atlas_access` docs. The 2.5D description is **well-attributed but
> reached through verifier evidence**, not a standalone primary fetch here — **confirm
> against the original methods before quoting parameters.**

### CCFv3 access — prefer `brainglobe-atlasapi` over legacy `allensdk`
The CCFv3 reference + annotation volume + ontology are available via **`brainglobe-atlasapi`**
(`allen_mouse_25um` / `allen_mouse_10um`), which is actively maintained and **numpy-2
friendly** — important because this repo runs **numpy 2.4**, and legacy `allensdk` pins
older numpy/pandas and risks dependency conflicts. The ABC-registered MERFISH reference and
per-cell `parcellation_index` come via `abc_atlas_access`.

Sources: [Yao et al. 2023 *Nature* (ABC Atlas)](https://www.nature.com/articles/s41586-023-06812-z) ·
[Zhang et al. 2023 *Nature* (WMB MERFISH)](https://www.nature.com/articles/s41586-023-06808-9) ·
[ABC Atlas MERFISH-CCF](https://alleninstitute.github.io/abc_atlas_access/descriptions/MERFISH-C57BL6J-638850-CCF.html) ·
[Zhuang MERFISH tutorial](https://alleninstitute.github.io/abc_atlas_access/notebooks/zhuang_merfish_tutorial.html)

---

## 4. Per-cell region assignment, uncertainty & QC

### Label transfer — verified mechanism
`high confidence (3-0)` on the mechanism

Per-cell assignment is the **standard multi-atlas label-transfer paradigm**: warp the
**CCFv3 annotation (label) volume into section space** (or map cell centroids into CCF) and
**look up each cell's region** by nearest-voxel resampling. Every tool surveyed does this —
brainreg warps annotations into sample space; STalign lifts annotations onto each cell;
QUINT/HERBS assign a region per pixel/cell; the ABC Atlas stores a `parcellation_index` per
cell.

### Calibrated uncertainty + QC — **design inference, not evidence-backed**
`low confidence — no surviving benchmarked claim`

This was an explicit research thread and **no verified claim provided a concrete, calibrated
recipe** for per-cell region-label uncertainty or quantitative registration QC. So the
following is **our engineering design**, to be validated empirically — but it is well-motivated
and reuses this repo's proven calibration machinery:

1. **Registration-ensemble bootstrap confidence (per cell).** Perturb the fitted
   registration `K` times within its plausible error (jitter the affine + cut-plane angle by
   ~DeepSlice's 6.6-voxel / few-degree error), re-look-up each cell's label, and report
   `confidence = fraction of perturbations agreeing with the modal label`. This is **the same
   bootstrap-confidence pattern already in [`map_to_reference`](../scripts/celltype_mapping.py)**
   — cells deep inside a region stay stable (high confidence); cells near a boundary flip (low
   confidence). It captures **registration error *and* boundary ambiguity in one calibrated
   score.**
2. **Distance-to-region-boundary** (CCF mm) as a cheap complementary signal.
3. **Section-level QC = cell-type composition cross-check vs the ABC reference.** For each
   assigned region, compare the section's cell-type distribution to the ABC reference's known
   per-region composition via **Jensen-Shannon divergence**; many high-JSD regions ⇒ likely
   misregistration. This makes the **existing cell-type map an orthogonal validator** — the
   same "orthogonal validation" philosophy as the repo's MERFISH-vs-bulk-RNAseq QC, and it
   directly answers the open question the research left unanswered.

---

## 5. Recommended pipeline for *this* repo

SpatialData-native, Python/PyTorch, open-source-first. Each stage is a pluggable backend so
the dependency-light synthetic path runs in offline pytest / GitHub Actions (mirroring how
`celltype_mapping.py` reimplements MapMyCells without the Allen download), while the real
engines wire in behind the same interface.

| Stage | Does | Library / model | Maturity · risk |
|---|---|---|---|
| **0 Ingest** | section cells (AnnData: `obsm['spatial']` + cell types) + CCFv3 | `spatialdata` (✅ installed), `brainglobe-atlasapi` | mature · numpy-2 OK |
| **1 Affine anchor + cut-angle** | section image → `O/U/V` plane in CCF | **DeepSlice** | mature, peer-reviewed · **coronal-only, brightfield-best** (render DAPI grayscale, or ABBA) |
| **2 Deformable warp** | affine→diffeomorphic 2D→3D, molecular-aware | **STalign** (LDDMM); **ANTs 2.5D SyN** as proven baseline | peer-reviewed · STalign user-inits slice; ANTs params not turnkey; verify `antspyx` on numpy 2 |
| **3 Label transfer** | warp cell centroids → CCF; look up `parcellation_index`/acronym + ontology path | annotation-volume resampling | textbook · — |
| **4 Calibrated UQ** | registration-ensemble bootstrap confidence + distance-to-boundary per cell | **our design** (reuses repo's bootstrap pattern) | **novel · validate empirically** |
| **5 QC cross-check** | per-region cell-type composition vs ABC ref (Jensen-Shannon) → section QC score | **our design** | **novel · validate empirically** |

**Outputs:** AnnData/SpatialData with per-cell `region`, `region_confidence`,
`dist_to_boundary`; a section-level QC report; a validation figure in the house style.

**Phased build (lowest-risk first):**
1. **Core + synthetic harness (no downloads):** geometry (`O/U/V` oblique-plane sampling),
   label transfer, calibrated UQ, QC cross-check — fully tested on a synthetic labelled CCF,
   proving (a) label-transfer correctness, (b) confidence calibration, (c) QC *detects*
   deliberate misregistration. ← **start here.**
2. **Real CCFv3** via `brainglobe-atlasapi` (`allen_mouse_25um`); per-cell lift-over on real data.
3. **DeepSlice** affine anchoring adapter (render section grayscale).
4. **STalign** LDDMM deformable backend; **ANTs 2.5D** baseline behind the same interface.
5. **moscot/PASTE2/CAST** cross-sample consensus + ABC composition cross-validation at scale.

---

## Implementation status & empirical validation (this repo)

Implemented in [`scripts/atlas_registration.py`](../scripts/atlas_registration.py) with
[`tests/test_atlas_registration.py`](../tests/test_atlas_registration.py). The
dependency-light core runs in offline pytest / GitHub Actions on a synthetic labelled atlas;
the real engines wire in behind the same interface (the `celltype_mapping.py` pattern).

**Real Allen CCFv3 loads + the core runs on it (validated).** `load_ccf_brainglobe` pulls the
real CCFv3 via `brainglobe-atlasapi` — **670 leaf regions in the volume / 841 in the ontology**
— and, crucially, **NumPy stayed at 2.4.6** (brainglobe did *not* downgrade it, confirming the
"brainglobe over legacy `allensdk`" recommendation in §3). STalign LDDMM now runs in the same
unified env (installed with `--no-deps` to bypass upstream's stale numpy pin; `np2typing` replaces
`nptyping` for numpy-2 compatibility).

**Hierarchical roll-up is required for usable per-cell labels — and the calibrated confidence
holds at every granularity.** A single section cannot resolve 670 leaf regions; `depth=` rolls
labels up the CCF ontology tree. Measured on a real oblique coronal section (allen_mouse_100um,
2.5-voxel anchoring error, 48-sample ensemble):

| ontology depth | regions (global / section) | accuracy | **high-conf accuracy** | frac high-conf | QC good→bad |
|---|---|---|---|---|---|
| 2 | 19 / 11 | 0.78 | **0.984** | 0.38 | 0.12 → — |
| 3 (recommended) | 50 / 16 | 0.76 | **0.982** | 0.31 | 0.14 → 0.96 |
| 4 | 113 / 28 | 0.71 | **0.976** | 0.23 | 0.18 |
| 6 | 421 / 81 | 0.49 | **0.993** | 0.05 | 0.34 |
| leaf | 841 / 166 | 0.33 | — (none reach ≥0.9) | 0.00 | 0.53 |

Three things are robustly true on **real CCFv3 data**: (1) label transfer is **exact under the
true plane** (1.00) at every depth — the indexing/lift-over is correct; (2) **high-confidence
calls are ~98–99% accurate regardless of granularity** — the registration-ensemble confidence
reliably flags which cells to trust, so you choose an operating point (coarse for coverage, fine
and trust only confident cells); (3) the **QC cross-check cleanly separates good vs deliberately-
bad registration** (e.g. 0.14 → 0.96 at depth 3). This is the §4 design — left open by the
literature — working on real data. Figure: [`assets/atlas_registration_ccfv3.png`](../assets/atlas_registration_ccfv3.png).

**Real STalign LDDMM deformable fit (validated).** Beyond the synthetic-error study above,
`stalign_register` wires the genuine **STalign `LDDMM_3D_to_slice`** (molecular-aware affine +
diffeomorphism) behind the same `Registration` interface, so label transfer / UQ / QC run on it
unchanged. On a section cut from a *known* CCFv3 plane (allen_mouse_100um, niter=100, ~2 min CPU),
STalign recovers the section's anterior–posterior position to **46.7 µm (< 0.5 voxel)** and matches
**0.89** of per-cell region labels at depth-3 — a *real* diffeomorphic fit, ground-truthed. Covered
by the live test `test_stalign_recovers_known_ccf_slice` and
[`scripts/stalign_demo.py`](../scripts/stalign_demo.py); figure
[`assets/atlas_registration_stalign.png`](../assets/atlas_registration_stalign.png). Upstream
STalign still lists `numpy==1.23.4` in `requirements.txt`, but the code runs on numpy 2.4 when
installed with `--no-deps` and `np2typing` (see README).

**Auto AP-anchor — fixing STalign's init-sensitivity (`coarse_ap_search`).** STalign's 2D→3D fit is
sensitive to its starting plane: from the default init a *posterior* thalamic section (AP 80)
misconverged to the wrong AP basin (**AP error ~1.44 mm, thalamus IoU 0.23**). The recommended
architecture has DeepSlice supply the affine anchor; since DeepSlice needs trained models +
brightfield section images, `coarse_ap_search` is a **training-free** stand-in — it slides the
section against every CCF coronal plane by normalized cross-correlation, picks the best-matching AP,
and `stalign_register(init="auto")` feeds it to LDDMM as the initial translation. That recovers the
same posterior section to **AP error ~148 µm (< 1.5 voxel), thalamus IoU 0.68** — a ~10× AP-error
reduction. Across three thalamic scans (AP 64 / 72 / 80) all now align (AP error ≤ 157 µm, thalamus
IoU 0.68–0.83); see [`scripts/thalamus_stalign_demo.py`](../scripts/thalamus_stalign_demo.py),
[`assets/thalamus_stalign_alignment.png`](../assets/thalamus_stalign_alignment.png), offline unit
tests in [`tests/test_atlas_init.py`](../tests/test_atlas_init.py), and the live test
`test_stalign_autoinit_recovers_posterior_scan`.

**Scale + rotation anchor, and the production DeepSlice (extensions).** `coarse_ap_search` is
generalized by **`coarse_anchor`**, which also grid-searches **in-plane scale and rotation** (NCC
over AP × scale × θ) — for real sections that differ in magnification or mounting angle from the
atlas — and builds the STalign init affine (in-plane scale + rotation, AP translation).
`stalign_register(init="auto")` now uses it; on a deliberately **1.25×-scaled** real section it
recovers **scale = 1.25** while keeping the AP within 1 voxel (unscaled → 1.0), with no regression
to the posterior recovery above. For the **production** anchor, **`deepslice_anchor(image_dir)`**
runs the real **DeepSlice** on a folder of coronal brightfield/DAPI-grayscale section images and
converts its QuickNII `O/U/V` output to `AnchoredPlane`s via **`anchoring_to_plane`** — wired and
guarded (raises with install hints until DeepSlice + section images are available, the "once images
are available" path). Both are covered by offline tests in
[`tests/test_atlas_init.py`](../tests/test_atlas_init.py).

> **Honest scope:** the calibration/QC numbers in the table use a *synthetic anchoring error* on
> real CCF geometry (ground truth known) and synthetic region-conditioned "cell types" for the QC
> signal — they validate the geometry + label-transfer + UQ + QC layers, not a registration
> engine's accuracy. The STalign result above *is* a real-fit validation. **DeepSlice** and **ANTs**
> remain stubbed with install hints.

## Refuted / do-not-claim

- **None killed** in this pass (25/25 claims confirmed). The only hedged item: CAST being
  *"more accurate than image-based registration"* is a **motivational assertion (2-1)**, not
  a head-to-head benchmark — don't quote it as established.

## Gaps & open questions (what this pass did *not* settle)

1. **Calibrated UQ/QC has no benchmarked recipe** — §4's design is inference; validate it.
2. **Learned-deformable frontier unestablished here** — VoxelMorph, TransMorph, KeyMorph,
   `uniGradICON` (foundation-model registration), and even the ANTs/SimpleElastix baselines
   produced **no surviving standalone claims**; their fit for 2D-MERFISH→3D-CCF is an open
   benchmark, not a recommendation.
3. **No quantitative head-to-head** (STalign vs ANTs vs CAST on MERFISH→CCF) survived — "best
   method" here rests on **capability fit, not benchmarked superiority**.
4. **Allen ANTs 2.5D specifics + CCFv3/ABC access details** were reached largely via verifier
   evidence — confirm against the Yao/Zhuang 2023 *Nature* methods and `allensdk`/
   `abc_atlas_access` before relying on parameters.

<sub>Research: 5 angles · 25 sources fetched · 125 claims extracted · 25 adversarially
verified (3-vote) · 25 confirmed · 0 killed · 10 synthesized findings. Nearly all findings
rest on peer-reviewed primary literature (*Nature*, *Nat Methods*, *Nat Commun*, *Genome
Research*, *eLife*, *Cell Reports*) + official tool docs.</sub>

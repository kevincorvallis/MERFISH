"""Tests for the segmentation-swap demo (segmentation_demo).

Operationalizes the verified 2025 finding (docs/methods-review-2026.md §1, "Segmentation
Matters"): the segmentation method that produces the cell-by-gene matrix measurably
changes downstream Leiden clustering. We simulate molecule-level MERFISH data with known
ground truth and show a modern **transcript-aware** segmentation (Baysor/proseg-style:
spatial position + expression likelihood) recovers boundary transcripts that a
nucleus-expansion / Voronoi baseline misassigns — yielding cleaner cell-type recovery.
"""

import numpy as np
import pytest

import segmentation_demo as seg


def test_transcript_aware_beats_voronoi_assignment():
    sim = seg.simulate_molecules(seed=0)
    vor = seg.segment_voronoi(sim)
    aware = seg.segment_transcript_aware(sim, seed=0)
    acc_vor = seg.assignment_accuracy(vor, sim)
    acc_aware = seg.assignment_accuracy(aware, sim)
    assert acc_aware > acc_vor, (
        f"transcript-aware should recover more transcripts than Voronoi: "
        f"aware={acc_aware:.3f} vor={acc_vor:.3f}")


def test_segmentation_changes_downstream_clustering():
    """The headline finding: different segmentation -> different Leiden ARI."""
    res = seg.evaluate(seed=0)
    methods = ("nucleus_only", "voronoi", "transcript_aware")
    aris = {m: res[m]["ari"] for m in methods}
    # the methods do not all yield the same clustering result
    assert len(set(round(v, 3) for v in aris.values())) > 1, (
        f"segmentation choice should change downstream ARI, got {aris}")
    # transcript-aware should not be worse than the Voronoi baseline downstream
    assert res["transcript_aware"]["ari"] >= res["voronoi"]["ari"] - 0.05


def test_simulation_is_deterministic():
    a = seg.assignment_accuracy(seg.segment_transcript_aware(seg.simulate_molecules(seed=1),
                                                              seed=0),
                                seg.simulate_molecules(seed=1))
    b = seg.assignment_accuracy(seg.segment_transcript_aware(seg.simulate_molecules(seed=1),
                                                              seed=0),
                                seg.simulate_molecules(seed=1))
    assert a == b


def test_cellpose_sam_hook_is_documented():
    """The production image-based path exists as a clearly-guarded optional hook."""
    assert hasattr(seg, "cellpose_sam_segment")
    if not seg.has_cellpose():
        with pytest.raises((ImportError, NotImplementedError)):
            seg.cellpose_sam_segment(np.zeros((8, 8), dtype="uint16"))

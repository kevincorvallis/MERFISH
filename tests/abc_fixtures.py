"""Bundled Allen ABC region x cell-class composition for offline atlas-registration QC."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "abc_region_composition.json"


def load_abc_reference() -> dict[int, dict[str, float]]:
    """Return ``{region_id: {cell_class: fraction}}`` from the bundled ABC table."""
    data = json.loads(_FIXTURE.read_text())
    return {int(r): comp for r, comp in data["regions"].items()}


def sample_cell_types(region_ids, reference: dict[int, dict[str, float]], rng: np.random.Generator,
                      p_dominant: float = 0.85) -> np.ndarray:
    """Assign each cell a cell class by sampling its true region's ABC reference mix."""
    region_ids = np.asarray(region_ids)
    present = sorted(reference)
    ct = np.empty(len(region_ids), dtype=object)
    for i, r in enumerate(region_ids):
        comp = reference.get(int(r))
        if comp is None:
            ct[i] = "Unknown"
            continue
        classes = list(comp.keys())
        probs = np.array([comp[c] for c in classes], dtype=float)
        probs /= probs.sum()
        if rng.random() < p_dominant:
            ct[i] = classes[rng.choice(len(classes), p=probs)]
        else:
            other = int(rng.choice(present))
            ocomp = reference[other]
            oc = list(ocomp.keys())
            op = np.array([ocomp[c] for c in oc], dtype=float)
            op /= op.sum()
            ct[i] = oc[rng.choice(len(oc), p=op)]
    return ct

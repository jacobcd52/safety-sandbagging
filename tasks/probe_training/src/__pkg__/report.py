"""Best-layer selection, the per-layer accuracy table and probe serialisation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .probe import LayerResult


def best_layer(accs: dict[int, float]) -> int:
    """Layer with the highest held-out accuracy. Ties go to the lowest layer index."""
    if not accs:
        raise ValueError("no layers")
    return max(accs.items())[0]


def format_table(accs: dict[int, float], best: int | None = None) -> str:
    lines = ["layer  val_acc"]
    for layer in sorted(accs):
        mark = "  <- best" if layer == best else ""
        lines.append(f"{layer:>5}  {accs[layer]:.3f}{mark}")
    return "\n".join(lines)


def save_probe(path: str | Path, result: LayerResult) -> None:
    """Save the __probe_noun__ (weights, bias, layer, normalisation stats) as an .npz."""
    np.savez(Path(path), w=result.probe.w, b=np.array(result.probe.b), layer=np.array(result.layer),
             mean=result.scaler.mean_, std=result.scaler.std_)


def load_probe(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

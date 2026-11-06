"""Best-layer selection, the per-layer accuracy table and probe serialisation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .activations import select_layer
from .probe import LayerResult, _sigmoid


def best_layer(accs: dict[int, float]) -> int:
    """Layer with the highest held-out accuracy. Ties go to the lowest layer index."""
    if not accs:
        raise ValueError("no layers")
    return max(accs, key=lambda layer: (accs[layer], -layer))


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


def apply_probe(probe: dict[str, np.ndarray], acts: np.ndarray) -> np.ndarray:
    """P(__PosLabel__) for every sample of an [n, layers, d] cache, using a probe from `load_probe`.

    Reads the probe's saved layer and applies the saved train-split normalisation
    (mean/std) before the linear map, so the scores match what the probe produced
    at training time.
    """
    X = (select_layer(acts, int(probe["layer"])) - probe["mean"]) / probe["std"]
    return _sigmoid(X @ probe["w"] + float(probe["b"]))

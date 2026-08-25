"""Reading the attention: which part of the conduction system the model looks at.

This is what the 1D-CNN classifier in Modelo3.ipynb could not give. There,
attention weighted positions in a convolutional feature map, with no anatomical
meaning. Here it weights *named nodes*, so you can ask whether a beat classified
as ventricular actually activated the ventricular nodes.

Honest warning: attention concentrating on a node does not prove the model
reasons about that node. Attention weights are a plausible explanation, not a
verified cause.
"""

from __future__ import annotations

import numpy as np
import torch

from .aami import CLASS_NAMES
from .graph import NODE_NAMES


@torch.no_grad()
def node_importance_by_class(
    model, loader, device, max_batches: int | None = None
) -> dict[str, np.ndarray]:
    """Mean importance of each node, grouped by true AAMI class."""
    model.eval()
    n_nodes = len(NODE_NAMES)
    sums = np.zeros((len(CLASS_NAMES), n_nodes), dtype=np.float64)
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = batch["x"].to(device)
        rule_vec = batch["rule_vec"].to(device)
        regime = batch["regime"].to(device)
        out = model(x, rule_vec, regime, return_attention=True)
        imp = out["node_importance"].cpu().numpy()
        ys = batch["y"].numpy()
        for cls in range(len(CLASS_NAMES)):
            mask = ys == cls
            if mask.any():
                sums[cls] += imp[mask].sum(axis=0)
                counts[cls] += int(mask.sum())

    mean = np.zeros_like(sums)
    seen = counts > 0
    mean[seen] = sums[seen] / counts[seen][:, None]
    return {"mean_importance": mean, "counts": counts}


def print_node_report(result: dict[str, np.ndarray], top_k: int = 5) -> None:
    mean, counts = result["mean_importance"], result["counts"]
    uniform = 1.0 / len(NODE_NAMES)

    print(f"\nMost-attended nodes per class (uniform attention = {uniform:.4f})")
    print("-" * 62)
    for cls_idx, cls in enumerate(CLASS_NAMES):
        if counts[cls_idx] == 0:
            print(f"{cls}: no samples in the evaluated set")
            continue
        order = np.argsort(-mean[cls_idx])[:top_k]
        tops = ", ".join(f"{NODE_NAMES[i]} ({mean[cls_idx, i]:.3f})" for i in order)
        print(f"{cls} (n={counts[cls_idx]:5d}): {tops}")
    print("-" * 62)


def contrast_against_normal(result: dict[str, np.ndarray], cls: str = "V", top_k: int = 5) -> None:
    """Attention difference between one class and normal beats.

    More informative than absolute importance: some nodes always draw high
    attention, and what matters is what *changes* when the beat is abnormal.
    """
    mean, counts = result["mean_importance"], result["counts"]
    i_n, i_c = CLASS_NAMES.index("N"), CLASS_NAMES.index(cls)
    if counts[i_n] == 0 or counts[i_c] == 0:
        print(f"Not enough samples to contrast {cls} against N")
        return
    delta = mean[i_c] - mean[i_n]
    order = np.argsort(-np.abs(delta))[:top_k]
    print(f"\nAttention shift for class {cls} relative to N:")
    for i in order:
        print(f"  {NODE_NAMES[i]:<14} {delta[i]:+.4f}")

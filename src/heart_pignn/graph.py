"""Cardiac conduction graph: SA node -> AV -> His-Purkinje -> myocardium.

Taken directly from Modelo2.ipynb. This is the physics-informed part of the
model: the topology is not learned, it is imposed from anatomy, and conduction
delays ride along as edge attributes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

NODE_NAMES: list[str] = [
    "SA", "AV", "HIS", "LBB", "RBB",
    "LP1", "LP2", "LP3", "RP1", "RP2",
    "LA_ant", "LA_post", "RA_ant", "RA_post",
    "LV_septal", "LV_anterior", "LV_lateral", "LV_posterior", "LV_apex", "LV_base",
    "RV_septal", "RV_freewall", "RV_apex", "RV_base",
]

# 0 sinus node, 1 AV node, 2 His bundle / branches, 3 Purkinje, 4 atrium, 5 ventricle
TYPE_MAP: dict[str, int] = {
    "SA": 0, "AV": 1, "HIS": 2, "LBB": 2, "RBB": 2,
    "LP1": 3, "LP2": 3, "LP3": 3, "RP1": 3, "RP2": 3,
    "LA_ant": 4, "LA_post": 4, "RA_ant": 4, "RA_post": 4,
    "LV_septal": 5, "LV_anterior": 5, "LV_lateral": 5,
    "LV_posterior": 5, "LV_apex": 5, "LV_base": 5,
    "RV_septal": 5, "RV_freewall": 5, "RV_apex": 5, "RV_base": 5,
}

# 0 specialized conduction, 1 left atrium, 2 right atrium, 3 LV, 4 RV, 5 Purkinje
CHAMBER_MAP: dict[str, int] = {
    "SA": 0, "AV": 0, "HIS": 0, "LBB": 5, "RBB": 5,
    "LP1": 5, "LP2": 5, "LP3": 5, "RP1": 5, "RP2": 5,
    "LA_ant": 1, "LA_post": 1, "RA_ant": 2, "RA_post": 2,
    "LV_septal": 3, "LV_anterior": 3, "LV_lateral": 3,
    "LV_posterior": 3, "LV_apex": 3, "LV_base": 3,
    "RV_septal": 4, "RV_freewall": 4, "RV_apex": 4, "RV_base": 4,
}

COORDS: dict[str, tuple[float, float, float]] = {
    "SA": (-0.20, 0.70, 0.20), "AV": (-0.10, 0.30, 0.00), "HIS": (0.00, 0.10, 0.00),
    "LBB": (-0.20, -0.10, -0.10), "RBB": (0.20, -0.10, -0.10),
    "LP1": (-0.35, -0.35, -0.10), "LP2": (-0.45, -0.45, -0.20), "LP3": (-0.25, -0.55, -0.25),
    "RP1": (0.35, -0.35, -0.10), "RP2": (0.25, -0.55, -0.20),
    "LA_ant": (-0.15, 0.55, 0.10), "LA_post": (-0.10, 0.45, -0.10),
    "RA_ant": (0.15, 0.55, 0.10), "RA_post": (0.10, 0.45, -0.10),
    "LV_septal": (-0.10, -0.30, 0.00), "LV_anterior": (-0.20, -0.35, 0.15),
    "LV_lateral": (-0.45, -0.35, 0.00), "LV_posterior": (-0.25, -0.40, -0.20),
    "LV_apex": (-0.20, -0.65, -0.15), "LV_base": (-0.15, -0.10, 0.00),
    "RV_septal": (0.08, -0.28, 0.00), "RV_freewall": (0.32, -0.35, 0.00),
    "RV_apex": (0.20, -0.58, -0.12), "RV_base": (0.12, -0.12, 0.00),
}


@dataclass
class HeartGraphSpec:
    node_names: list[str]
    node_types: list[int]
    chambers: list[int]
    coords: torch.Tensor      # [N, 3]
    edge_index: torch.Tensor  # [2, E]
    edge_attr: torch.Tensor   # [E, 3] -> (delay_s, weight, coupling_type)

    @property
    def n_nodes(self) -> int:
        return len(self.node_names)

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


def build_heart_graph(device: str | torch.device = "cpu") -> HeartGraphSpec:
    """Build the default 24-node conduction graph."""
    name_to_idx = {n: i for i, n in enumerate(NODE_NAMES)}

    def e(a: str, b: str, delay: float, weight: float, kind: int):
        return (name_to_idx[a], name_to_idx[b], [delay, weight, float(kind)])

    edges = [
        # Specialized pathway: SA -> atria -> AV -> His -> branches -> Purkinje
        e("SA", "RA_ant", 0.020, 1.0, 0), e("SA", "LA_ant", 0.030, 0.9, 0),
        e("SA", "AV", 0.050, 1.0, 0), e("AV", "HIS", 0.080, 1.0, 0),
        e("HIS", "LBB", 0.015, 1.0, 0), e("HIS", "RBB", 0.015, 1.0, 0),
        e("LBB", "LP1", 0.010, 1.0, 0), e("LBB", "LP2", 0.012, 0.9, 0),
        e("LBB", "LP3", 0.014, 0.9, 0),
        e("RBB", "RP1", 0.010, 1.0, 0), e("RBB", "RP2", 0.013, 0.9, 0),
    ]

    lv_nodes = ["LV_septal", "LV_anterior", "LV_lateral", "LV_posterior", "LV_apex", "LV_base"]
    rv_nodes = ["RV_septal", "RV_freewall", "RV_apex", "RV_base"]
    atria_l = ["LA_ant", "LA_post"]
    atria_r = ["RA_ant", "RA_post"]

    for n in lv_nodes:
        edges += [e("LP1", n, 0.010, 0.8, 0), e("LP2", n, 0.012, 0.7, 0)]
    for n in rv_nodes:
        edges += [e("RP1", n, 0.010, 0.8, 0), e("RP2", n, 0.012, 0.7, 0)]
    for n in atria_l:
        edges += [e("LA_ant", n, 0.020, 0.7, 1), e(n, "LA_ant", 0.020, 0.7, 1)]
    for n in atria_r:
        edges += [e("RA_ant", n, 0.020, 0.7, 1), e(n, "RA_ant", 0.020, 0.7, 1)]
    for a, b in zip(lv_nodes, lv_nodes[1:], strict=False):
        edges += [e(a, b, 0.025, 0.7, 1), e(b, a, 0.025, 0.7, 1)]
    for a, b in zip(rv_nodes, rv_nodes[1:], strict=False):
        edges += [e(a, b, 0.040, 0.5, 2), e(b, a, 0.040, 0.5, 2)]

    src = torch.tensor([x[0] for x in edges], dtype=torch.long, device=device)
    dst = torch.tensor([x[1] for x in edges], dtype=torch.long, device=device)
    edge_attr = torch.tensor([x[2] for x in edges], dtype=torch.float32, device=device)

    return HeartGraphSpec(
        node_names=list(NODE_NAMES),
        node_types=[TYPE_MAP[n] for n in NODE_NAMES],
        chambers=[CHAMBER_MAP[n] for n in NODE_NAMES],
        coords=torch.tensor([COORDS[n] for n in NODE_NAMES], dtype=torch.float32, device=device),
        edge_index=torch.stack([src, dst], dim=0),
        edge_attr=edge_attr,
    )

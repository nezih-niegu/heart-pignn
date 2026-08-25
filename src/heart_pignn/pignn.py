"""PIGNN encoder: beat waveform -> node states over the conduction graph.

Same engine as Modelo2.ipynb (GraphGRUCell + anatomical graph), with two changes
that make it viable for beat-level classification:

1. **Vectorized message passing.** Modelo2 aggregated with a Python loop over the
   batch (`for b in range(B): out[b].index_add_(...)`). Since the graph is
   identical for every element of the batch, a single `index_add_` along the node
   dimension covers the whole batch at once. At batch size 128 that is the
   difference between "trains" and "never finishes".

2. **Compressed time grid.** Modelo2 ran one graph step per sample (500 steps).
   Here the convolutional encoder reduces the 360-sample window to `graph_steps`
   steps (32 by default) before entering the graph. Beat morphology is preserved
   in the channels; what gets compressed is the axis along which the activation
   wavefront propagates.

The model also exposes `vm` (a membrane-potential-like variable) and `tension`,
neither of which is directly supervised -- only regularized. That is the
physics-informed piece that survives from the reconstruction model.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import HeartGraphSpec


def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    activation: type[nn.Module] = nn.SiLU,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), activation()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class GraphGRUCell(nn.Module):
    """One message-passing step followed by a per-node GRU update."""

    def __init__(self, hidden_dim: int, edge_dim: int, msg_dim: int, dropout: float = 0.0):
        super().__init__()
        self.msg_net = build_mlp(2 * hidden_dim + edge_dim, [msg_dim], msg_dim, dropout=dropout)
        self.gru = nn.GRUCell(msg_dim + hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.msg_dim = msg_dim

    def forward(
        self,
        h: torch.Tensor,            # [B, N, H]
        local_drive: torch.Tensor,  # [B, N, H]
        edge_index: torch.Tensor,   # [2, E]
        edge_attr: torch.Tensor,    # [E, edge_dim]
    ) -> torch.Tensor:
        src_idx, dst_idx = edge_index
        b, n, hdim = h.shape

        h_src = h.index_select(1, src_idx)          # [B, E, H]
        h_dst = h.index_select(1, dst_idx)          # [B, E, H]
        ea = edge_attr.unsqueeze(0).expand(b, -1, -1)
        msgs = self.msg_net(torch.cat([h_src, h_dst, ea], dim=-1))  # [B, E, M]

        # The graph is identical across the batch, so one index_add_ along the
        # node dimension aggregates every element at once.
        agg = msgs.new_zeros(b, n, self.msg_dim)
        agg.index_add_(1, dst_idx, msgs)

        inp = torch.cat([agg, local_drive], dim=-1)
        h_new = self.gru(inp.reshape(b * n, -1), h.reshape(b * n, hdim)).reshape(b, n, hdim)
        return self.norm(h_new)


class SignalEncoder(nn.Module):
    """Beat window [B, 1, T] -> observation sequence [B, S, H]."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 64, graph_steps: int = 32):
        super().__init__()
        self.graph_steps = graph_steps
        half = max(hidden_dim // 2, 8)
        self.conv1 = nn.Conv1d(in_channels, half, kernel_size=15, stride=2, padding=7)
        self.norm1 = nn.BatchNorm1d(half)
        self.conv2 = nn.Conv1d(half, hidden_dim, kernel_size=9, stride=2, padding=4)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, stride=2, padding=3)
        self.norm3 = nn.BatchNorm1d(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.silu(self.norm1(self.conv1(x)))
        z = F.silu(self.norm2(self.conv2(z)))
        z = F.silu(self.norm3(self.conv3(z)))
        z = F.adaptive_avg_pool1d(z, self.graph_steps)  # [B, H, S]
        return self.proj(z.transpose(1, 2))             # [B, S, H]


class PIGNNEncoder(nn.Module):
    """Propagate the observation over the conduction graph and return node states."""

    def __init__(
        self,
        graph: HeartGraphSpec,
        hidden_dim: int = 64,
        node_emb_dim: int = 32,
        msg_dim: int = 64,
        n_layers: int = 2,
        graph_steps: int = 32,
        in_channels: int = 1,
        type_vocab: int = 6,
        chamber_vocab: int = 6,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_nodes = graph.n_nodes
        self.hidden_dim = hidden_dim
        self.graph_steps = graph_steps

        self.signal_encoder = SignalEncoder(in_channels, hidden_dim, graph_steps)

        self.type_emb = nn.Embedding(type_vocab, node_emb_dim)
        self.chamber_emb = nn.Embedding(chamber_vocab, node_emb_dim)
        self.coord_proj = nn.Linear(3, node_emb_dim)
        self.node_init = build_mlp(3 * node_emb_dim, [hidden_dim], hidden_dim, dropout=dropout)

        self.obs_to_nodes = build_mlp(2 * hidden_dim, [hidden_dim], hidden_dim, dropout=dropout)
        self.obs_gate = build_mlp(2 * hidden_dim, [hidden_dim], hidden_dim, dropout=dropout)

        self.cells = nn.ModuleList(
            [
                GraphGRUCell(hidden_dim, edge_dim=graph.edge_attr.shape[1], msg_dim=msg_dim, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        self.ion_head = build_mlp(hidden_dim, [hidden_dim // 2], 1)
        self.tension_head = build_mlp(hidden_dim, [hidden_dim // 2], 1)

        self.register_buffer("node_types", torch.tensor(graph.node_types, dtype=torch.long))
        self.register_buffer("chambers", torch.tensor(graph.chambers, dtype=torch.long))
        self.register_buffer("coords", graph.coords.clone())
        self.register_buffer("edge_index", graph.edge_index.clone())
        self.register_buffer("edge_attr", graph.edge_attr.clone())

    def initial_node_state(self, batch_size: int) -> torch.Tensor:
        h0 = self.node_init(
            torch.cat(
                [
                    self.type_emb(self.node_types),
                    self.chamber_emb(self.chambers),
                    self.coord_proj(self.coords),
                ],
                dim=-1,
            )
        )
        return h0.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def _drive(self, obs_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        obs_node = obs_t.unsqueeze(1).expand(-1, self.n_nodes, -1)
        fused = torch.cat([obs_node, h], dim=-1)
        gate = torch.sigmoid(self.obs_gate(fused))
        return gate * self.obs_to_nodes(fused)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """x: [B, C, T] -> dict with node_states [B, S, N, H], vm and tension [B, S, N]."""
        obs = self.signal_encoder(x)  # [B, S, H]
        b = obs.shape[0]
        h = self.initial_node_state(b)

        states, vms, tensions = [], [], []
        for t in range(self.graph_steps):
            drive = self._drive(obs[:, t, :], h)
            for cell in self.cells:
                h = cell(h, drive, self.edge_index, self.edge_attr)
            states.append(h)
            vms.append(self.ion_head(h).squeeze(-1))
            tensions.append(self.tension_head(h).squeeze(-1))

        return {
            "node_states": torch.stack(states, dim=1),  # [B, S, N, H]
            "vm": torch.stack(vms, dim=1),              # [B, S, N]
            "tension": torch.stack(tensions, dim=1),    # [B, S, N]
        }

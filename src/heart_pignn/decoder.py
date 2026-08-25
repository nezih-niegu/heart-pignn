"""Regression head: from node states back to a waveform.

This is Modelo2's ECGProjectionHead adapted to this scale. Each node projects an
electrical contribution, contributions are aggregated using anatomical
coordinates, and a temporal refinement corrects the result.

The important difference from the classifier: this head does NOT read the
attention latent `z`, it reads the full `node_states` sequence. Attention
collapses time into a single vector, and you cannot rebuild a waveform from a
vector. Hanging regression off the sequence and classification off the vector is
exactly what splits the work -- the graph has to preserve fine temporal detail
for the first and summarizable structure for the second.

What gets predicted depends on `DataConfig.regression_target`:

- `reconstruct` (default): the model's own clean input window, i.e. a denoising
  autoencoder. Measured on MIT-BIH this reaches PRD ~26% and Pearson ~0.96.
- `cross_lead`: channel 1 (V1/V5) predicted from channel 0. Physically the more
  interesting task, but measured on MIT-BIH it *collapses* -- PRD ~101%, Pearson
  ~0.02. Predicting V1 morphology from a single MLII beat with no record identity
  is close to ill-posed, and MSE's optimum for an unpredictable target is the
  conditional mean, which is a flat line. Kept available for experiments, not
  recommended as a default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pignn import build_mlp


class DipoleSignalDecoder(nn.Module):
    """node_states [B, S, N, H] -> signal [B, T, L]."""

    def __init__(
        self,
        hidden_dim: int,
        n_leads: int = 1,
        out_len: int = 360,
        source_dim: int = 3,
        refine_hidden: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.out_len = out_len
        self.n_leads = n_leads
        refine_hidden = refine_hidden or max(hidden_dim // 2, 16)

        # Coordinates are concatenated: a node's contribution to the electrode
        # depends on where it sits, not only on its state.
        self.node_to_lead = build_mlp(hidden_dim + 3, [hidden_dim], n_leads, dropout=dropout)
        self.source_head = build_mlp(hidden_dim + 3, [hidden_dim // 2], source_dim)

        self.refine = nn.GRU(
            input_size=n_leads + source_dim,
            hidden_size=refine_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.out = nn.Linear(2 * refine_hidden, n_leads)

    def forward(self, node_states: torch.Tensor, coords: torch.Tensor) -> dict[str, torch.Tensor]:
        b, s, n, _ = node_states.shape
        coord_exp = coords.view(1, 1, n, 3).expand(b, s, -1, -1)
        feat = torch.cat([node_states, coord_exp], dim=-1)

        raw = self.node_to_lead(feat).mean(dim=2)      # [B, S, L]
        source = self.source_head(feat).mean(dim=2)    # [B, S, 3]

        # The graph grid (S steps) is stretched to sampling resolution.
        raw_up = F.interpolate(
            raw.transpose(1, 2), size=self.out_len, mode="linear", align_corners=False
        ).transpose(1, 2)
        source_up = F.interpolate(
            source.transpose(1, 2), size=self.out_len, mode="linear", align_corners=False
        ).transpose(1, 2)

        z, _ = self.refine(torch.cat([raw_up, source_up], dim=-1))
        # Residual connection over the raw projection, as in Modelo2: the
        # refinement corrects, it does not reinvent.
        return {"signal": self.out(z) + raw_up, "source": source_up, "signal_raw": raw_up}

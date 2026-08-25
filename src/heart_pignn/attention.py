"""The bridge: the attention module whose latent space joins the PIGNN to main.py.

Modelo3.ipynb already had temporal attention over the CNN output, but only as a
pooling device. Here attention is two-stage -- over nodes, then over time -- and
its output `z` is where the two systems meet:

- upward, `z` feeds the 5-class AAMI head;
- sideways, `z` feeds an auxiliary head that must reproduce the verdict of the
  main.py rule tree;
- downward, the soft rule vector modulates `z` through FiLM before classification.

On top of that, `BatchPrototypeMemory` keeps a moving average of `z` per rhythm
regime, updated from the weights of the current batch. Each beat's similarity to
those prototypes enters the classifier as an extra feature: the model compares a
beat not only against fixed weights, but against the summary training itself has
been building for each regime.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CardiacAttentionBridge(nn.Module):
    """Node -> time attention. Returns the latent and the weights that produced it.

    The weights are not a by-product: `node_importance` says which region of the
    heart dominated each beat's decision, which is what lets you check whether the
    model actually looks at the ventricle when it calls a beat "V".
    """

    def __init__(self, hidden_dim: int, attn_dim: int | None = None):
        super().__init__()
        attn_dim = attn_dim or max(hidden_dim // 2, 16)
        self.node_score = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1)
        )
        self.time_score = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """node_states: [B, S, N, H]."""
        alpha = torch.softmax(self.node_score(node_states).squeeze(-1), dim=-1)  # [B, S, N]
        pooled_nodes = torch.einsum("bsn,bsnh->bsh", alpha, node_states)         # [B, S, H]

        beta = torch.softmax(self.time_score(pooled_nodes).squeeze(-1), dim=-1)  # [B, S]
        z = torch.einsum("bs,bsh->bh", beta, pooled_nodes)                       # [B, H]

        node_importance = torch.einsum("bs,bsn->bn", beta, alpha)                # [B, N]

        return {
            "latent": self.norm(z),
            "node_attention": alpha,
            "time_attention": beta,
            "node_importance": node_importance,
        }


class RuleFiLM(nn.Module):
    """Modulate the latent with the soft rule vector from main.py.

    FiLM rather than concatenation: concatenating lets the network ignore the
    rules by growing the norm of the rest of the vector. With FiLM the rules scale
    and shift every latent dimension, so rhythm context changes *how morphology is
    read* instead of competing with it.
    """

    def __init__(self, hidden_dim: int, rule_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(rule_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        # Near-identity init: FiLM barely touches the latent at first and the
        # rules phase in gradually. Weights are small but not zero -- at exactly
        # zero the last layer blocks gradient to the earlier ones and the rule
        # encoder takes epochs to start moving.
        nn.init.normal_(self.encoder[-1].weight, std=0.01)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, z: torch.Tensor, rule_vec: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.encoder(rule_vec).chunk(2, dim=-1)
        return self.norm(z * (1.0 + gamma) + beta)


class BatchPrototypeMemory(nn.Module):
    """Latent prototypes per rhythm regime, kept as a moving average over batches.

    Each training batch averages the latents of every regime (normal,
    bradycardia, tachycardia, irregular) using weights derived from that batch,
    then blends them into the accumulated prototype. At evaluation time the
    prototypes are frozen. The output is each beat's cosine similarity to each
    prototype.
    """

    def __init__(self, hidden_dim: int, n_regimes: int, momentum: float = 0.95):
        super().__init__()
        self.momentum = momentum
        self.n_regimes = n_regimes
        self.register_buffer("prototypes", torch.zeros(n_regimes, hidden_dim))
        self.register_buffer("initialized", torch.zeros(n_regimes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, z: torch.Tensor, regimes: torch.Tensor) -> None:
        z = z.detach().float()
        onehot = F.one_hot(regimes.long(), self.n_regimes).float()  # [B, R]
        counts = onehot.sum(dim=0)                                  # [R]
        sums = onehot.t() @ z                                       # [R, H]

        present = counts > 0
        if not bool(present.any()):
            return

        batch_mean = torch.zeros_like(self.prototypes)
        batch_mean[present] = sums[present] / counts[present].unsqueeze(-1)

        fresh = present & (~self.initialized)
        stale = present & self.initialized

        if bool(fresh.any()):
            self.prototypes[fresh] = batch_mean[fresh]
            self.initialized[fresh] = True
        if bool(stale.any()):
            m = self.momentum
            self.prototypes[stale] = m * self.prototypes[stale] + (1.0 - m) * batch_mean[stale]

    def similarity(self, z: torch.Tensor) -> torch.Tensor:
        protos = self.prototypes.to(z.dtype)
        mask = self.initialized.to(z.dtype).unsqueeze(0)  # [1, R]
        sim = F.cosine_similarity(z.unsqueeze(1), protos.unsqueeze(0), dim=-1)
        return sim * mask

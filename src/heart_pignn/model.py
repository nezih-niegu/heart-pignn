"""Full model: PIGNN -> two outputs, signal regression and pathology classification.

    beat window [B,1,T]
            |
    SignalEncoder (conv1d x3, stride 2, pooled to S steps)
            |
    PIGNNEncoder: GraphGRUCell x L over the conduction graph
            |  node_states [B,S,N,H] + vm + tension
            |
            +---------------------------+
            |                           |
            v                           v
    CardiacAttentionBridge      DipoleSignalDecoder
    (collapses time)            (preserves time)
            |                           |
            +--> z  SHARED LATENT       +--> REGRESSION OUTPUT
            |     |                          signal [B,T,L]
            |     +--> auxiliary head: main.py's verdict
            |     +--> similarity to batch regime prototypes
            |     +--> RuleFiLM(z, main.py's soft tree)
            |                |
            |                v
            |     CLASSIFICATION OUTPUT
            |          AAMI (N,S,V,F,Q)

The two heads share the encoder and pull it in different directions: regression
demands fine temporal detail, classification demands a summarizable
representation. That tension is the point of the design, not a side effect --
`recon_w` sets the balance and `recon_w=0` gives the ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .aami import CLASS_NAMES
from .attention import BatchPrototypeMemory, CardiacAttentionBridge, RuleFiLM
from .decoder import DipoleSignalDecoder
from .graph import HeartGraphSpec, build_heart_graph
from .heuristics import N_REGIMES, N_RULE_FEATURES
from .pignn import PIGNNEncoder, build_mlp


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    node_emb_dim: int = 32
    msg_dim: int = 64
    n_layers: int = 2
    graph_steps: int = 32
    dropout: float = 0.1
    rule_dim: int = N_RULE_FEATURES
    n_rule_labels: int = 3
    n_classes: int = len(CLASS_NAMES)
    use_prototypes: bool = True
    prototype_momentum: float = 0.95
    # --- regression output ---
    use_regression: bool = True
    n_signal_outputs: int = 1
    signal_len: int = 360
    decoder_hidden: int | None = None


class PIGNNBeatModel(nn.Module):
    """Two heads over a shared PIGNN encoder."""

    def __init__(self, config: ModelConfig | None = None, graph: HeartGraphSpec | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.graph = graph or build_heart_graph()

        self.encoder = PIGNNEncoder(
            graph=self.graph,
            hidden_dim=cfg.hidden_dim,
            node_emb_dim=cfg.node_emb_dim,
            msg_dim=cfg.msg_dim,
            n_layers=cfg.n_layers,
            graph_steps=cfg.graph_steps,
            dropout=cfg.dropout,
        )

        # --- classification branch ---
        self.bridge = CardiacAttentionBridge(cfg.hidden_dim)
        self.film = RuleFiLM(cfg.hidden_dim, cfg.rule_dim, dropout=cfg.dropout)
        self.prototypes = (
            BatchPrototypeMemory(cfg.hidden_dim, N_REGIMES, cfg.prototype_momentum)
            if cfg.use_prototypes
            else None
        )
        head_in = cfg.hidden_dim + (N_REGIMES if cfg.use_prototypes else 0)
        self.classifier = nn.Sequential(
            nn.Linear(head_in, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.n_classes),
        )
        # Fed the raw latent, before FiLM: reading the modulated latent would let
        # it copy the rules straight from its own input instead of learning to
        # reconstruct them.
        self.rule_head = build_mlp(cfg.hidden_dim, [cfg.hidden_dim // 2], cfg.n_rule_labels)

        # --- regression branch ---
        self.decoder = (
            DipoleSignalDecoder(
                hidden_dim=cfg.hidden_dim,
                n_leads=cfg.n_signal_outputs,
                out_len=cfg.signal_len,
                refine_hidden=cfg.decoder_hidden,
                dropout=cfg.dropout,
            )
            if cfg.use_regression and cfg.n_signal_outputs > 0
            else None
        )

    def forward(
        self,
        x: torch.Tensor,                      # [B, 1, T]
        rule_vec: torch.Tensor,               # [B, rule_dim]
        regimes: torch.Tensor | None = None,  # [B]
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        enc = self.encoder(x)
        node_states = enc["node_states"]

        # --- classification ---
        attn = self.bridge(node_states)
        z = attn["latent"]
        rule_logits = self.rule_head(z)

        if self.prototypes is not None:
            if self.training and regimes is not None:
                self.prototypes.update(z, regimes)
            proto_sim = self.prototypes.similarity(z)
            feats = torch.cat([self.film(z, rule_vec), proto_sim], dim=-1)
        else:
            proto_sim = None
            feats = self.film(z, rule_vec)

        out = {
            "logits": self.classifier(feats),
            "rule_logits": rule_logits,
            "latent": z,
            "vm": enc["vm"],
            "tension": enc["tension"],
        }
        if proto_sim is not None:
            out["prototype_similarity"] = proto_sim

        # --- regression ---
        if self.decoder is not None:
            dec = self.decoder(node_states, self.encoder.coords)
            out["signal"] = dec["signal"]
            out["source"] = dec["source"]

        if return_attention:
            out["node_attention"] = attn["node_attention"]
            out["time_attention"] = attn["time_attention"]
            out["node_importance"] = attn["node_importance"]
        return out


# Former name, from when the model only classified. Kept so existing checkpoints
# and notebooks that imported it keep working.
PIGNNBeatClassifier = PIGNNBeatModel

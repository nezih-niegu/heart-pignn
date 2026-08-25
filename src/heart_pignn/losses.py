"""Composite loss: signal regression + classification + rules + physics.

    L = recon_w * L_signal + CE(capped weights) + rule_w * BCE_rules + phys_w * L_physics

`L_signal` reproduces Modelo2's combination: MSE for exact shape, `1 - Pearson`
for morphology, and temporal smoothness against artefacts. MSE alone rewards
predicting the mean when the signal is hard; the Pearson term is what forces the
wave to have the right shape even if the scale is off.

Every signal term is masked: beats whose record has no target lead contribute
nothing, instead of feeding in zeros the model would learn to predict.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(per_sample: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average over samples with a valid target only."""
    mask = mask.to(per_sample.dtype)
    return (per_sample * mask).sum() / mask.sum().clamp(min=1.0)


class BeatLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        rule_w: float = 0.3,
        phys_w: float = 0.05,
        recon_w: float = 0.5,
        pearson_w: float = 0.3,
        smooth_w: float = 0.05,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.rule_w = rule_w
        self.phys_w = phys_w
        self.recon_w = recon_w
        self.pearson_w = pearson_w
        self.smooth_w = smooth_w
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.bce = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------- regression

    @staticmethod
    def pearson_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """1 - Pearson correlation, per sample and per lead."""
        pc = pred - pred.mean(dim=1, keepdim=True)
        tc = target - target.mean(dim=1, keepdim=True)
        corr = (pc * tc).sum(dim=1) / (pc.norm(dim=1) * tc.norm(dim=1) + 1e-8)  # [B, L]
        return _masked_mean((1.0 - corr).mean(dim=-1), mask)

    def signal_loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        l_mse = _masked_mean(((pred - target) ** 2).mean(dim=(1, 2)), mask)
        l_pearson = self.pearson_loss(pred, target, mask)
        dy = pred[:, 1:, :] - pred[:, :-1, :]
        l_smooth = _masked_mean((dy**2).mean(dim=(1, 2)), mask)
        total = l_mse + self.pearson_w * l_pearson + self.smooth_w * l_smooth
        return {"signal": total, "signal_mse": l_mse, "signal_pearson": l_pearson}

    # -------------------------------------------------------------- physics

    @staticmethod
    def physiological_regularization(pred: dict[str, torch.Tensor]) -> torch.Tensor:
        vm, tension = pred["vm"], pred["tension"]
        reg = 0.1 * (vm**2).mean()                            # bounded potential
        reg = reg + 0.1 * F.relu(tension.abs() - 5.0).mean()  # bounded tension
        dvm = vm[:, 1:, :] - vm[:, :-1, :]
        reg = reg + 0.2 * (dvm**2).mean()                     # continuous activation front
        if "source" in pred:
            dsrc = pred["source"][:, 1:, :] - pred["source"][:, :-1, :]
            reg = reg + 0.2 * (dsrc**2).mean()                # dipole without jumps
        return reg

    # ---------------------------------------------------------------- total

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        y: torch.Tensor,
        rule_target: torch.Tensor | None = None,
        y_signal: torch.Tensor | None = None,
        signal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        l_ce = self.ce(pred["logits"], y)
        total = l_ce
        out = {"ce": l_ce}

        if "signal" in pred and y_signal is not None and self.recon_w > 0:
            if signal_mask is None:
                signal_mask = y_signal.new_ones(y_signal.shape[0])
            parts = self.signal_loss(pred["signal"], y_signal, signal_mask)
            total = total + self.recon_w * parts["signal"]
            out.update(parts)

        if rule_target is not None and self.rule_w > 0:
            l_rule = self.bce(pred["rule_logits"], rule_target)
            total = total + self.rule_w * l_rule
            out["rule_bce"] = l_rule

        if self.phys_w > 0 and "vm" in pred:
            l_phys = self.physiological_regularization(pred)
            total = total + self.phys_w * l_phys
            out["phys"] = l_phys

        out["loss"] = total
        return out

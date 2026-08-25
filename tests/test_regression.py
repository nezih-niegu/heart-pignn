"""Tests for the regression head and its interaction with the classification head."""

import numpy as np
import pytest
import torch

from heart_pignn.data import DataConfig, MITBIHBeatDataset
from heart_pignn.demo_data import generate_dataset
from heart_pignn.heuristics import N_REGIMES, N_RULE_FEATURES
from heart_pignn.losses import BeatLoss
from heart_pignn.model import ModelConfig, PIGNNBeatModel
from heart_pignn.train import TrainConfig, run_test, run_training

SMALL = ModelConfig(
    hidden_dim=16, msg_dim=16, node_emb_dim=8, n_layers=1, graph_steps=4, signal_len=360
)


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("mitdb_reg")
    generate_dataset(path, n_records=8, n_beats=120, seed=3)
    return path


def make_batch(b=6, t=360):
    return (
        torch.randn(b, 1, t),
        torch.rand(b, N_RULE_FEATURES),
        torch.randint(0, N_REGIMES, (b,)),
    )


def test_model_emits_both_outputs():
    model = PIGNNBeatModel(SMALL)
    x, rule, regime = make_batch()
    out = model(x, rule, regime)
    assert out["logits"].shape == (6, 5)          # classification
    assert out["signal"].shape == (6, 360, 1)     # regression, at sampling resolution
    assert torch.isfinite(out["signal"]).all()


def test_regression_can_be_disabled():
    model = PIGNNBeatModel(ModelConfig(**{**SMALL.__dict__, "use_regression": False}))
    out = model(*make_batch())
    assert "signal" not in out
    assert out["logits"].shape == (6, 5)


def test_masked_signal_loss_ignores_beats_without_target():
    crit = BeatLoss(recon_w=1.0, rule_w=0.0, phys_w=0.0)
    pred = {"logits": torch.randn(4, 5), "rule_logits": torch.randn(4, 3),
            "signal": torch.zeros(4, 360, 1)}
    target = torch.ones(4, 360, 1)
    y = torch.randint(0, 5, (4,))

    all_valid = crit(pred, y, None, target, torch.ones(4))["signal_mse"]
    half_valid = crit(pred, y, None, target, torch.tensor([1.0, 1.0, 0.0, 0.0]))["signal_mse"]
    none_valid = crit(pred, y, None, target, torch.zeros(4))["signal_mse"]

    assert pytest.approx(float(all_valid), abs=1e-6) == float(half_valid)  # same mean error
    assert float(none_valid) == pytest.approx(0.0, abs=1e-6)


def test_both_heads_receive_gradient():
    model = PIGNNBeatModel(SMALL)
    x, rule, regime = make_batch()
    out = model(x, rule, regime)
    losses = BeatLoss(rule_w=0.3, phys_w=0.05, recon_w=0.5)(
        out, torch.randint(0, 5, (6,)), torch.rand(6, 3).round(),
        torch.randn(6, 360, 1), torch.ones(6),
    )
    losses["loss"].backward()

    dec_grad = model.decoder.node_to_lead[0].weight.grad
    cls_grad = model.classifier[0].weight.grad
    enc_grad = model.encoder.cells[0].msg_net[0].weight.grad
    assert dec_grad is not None and dec_grad.abs().sum() > 0
    assert cls_grad is not None and cls_grad.abs().sum() > 0
    assert enc_grad is not None and enc_grad.abs().sum() > 0, "the encoder is shared"


def test_cross_lead_target_is_not_the_input(demo_dir):
    # cross_lead is no longer the default (it collapses on MIT-BIH), so ask for it.
    cfg = DataConfig(regression_target="cross_lead")
    ds = MITBIHBeatDataset(demo_dir, split="train", config=cfg, verbose=False)
    item = ds[0]
    assert item["y_signal"].shape == (360, 1)
    assert float(item["signal_mask"]) == 1.0
    x = item["x"].squeeze(0)
    y = item["y_signal"].squeeze(-1)
    assert not torch.allclose(x, y), "cross_lead must predict another lead, not copy"


def test_reconstruct_target_matches_clean_window(demo_dir):
    cfg = DataConfig(regression_target="reconstruct", augment=False)
    ds = MITBIHBeatDataset(demo_dir, split="train", config=cfg, verbose=False)
    item = ds[0]
    assert torch.allclose(item["x"].squeeze(0), item["y_signal"].squeeze(-1))


def test_no_regression_mode_masks_everything(demo_dir):
    cfg = DataConfig(regression_target="none")
    ds = MITBIHBeatDataset(demo_dir, split="train", config=cfg, verbose=False)
    assert cfg.n_signal_outputs == 0
    assert float(ds[0]["signal_mask"]) == 0.0


def test_default_regression_target_is_reconstruct():
    """cross_lead collapsed on real MIT-BIH; reconstruct is the working default."""
    assert DataConfig().regression_target == "reconstruct"


def test_end_to_end_reports_signal_metrics(demo_dir, tmp_path):
    cfg = TrainConfig(
        data_root=str(demo_dir),
        output_dir=str(tmp_path / "ckpt"),
        epochs=2,
        batch_size=32,
        samples_per_epoch=128,
        recon_w=0.5,
        data=DataConfig(),
        model=ModelConfig(hidden_dim=16, msg_dim=16, node_emb_dim=8, n_layers=1, graph_steps=4),
    )
    result = run_training(cfg)
    assert all(np.isfinite(h["val_prd"]) for h in result["history"])

    run = run_test(cfg)
    assert run["signal_prd"] > 0
    assert -1.0 <= run["signal_corr"] <= 1.0


def test_combined_selection_penalizes_bad_reconstruction():
    from heart_pignn.train import _selection_score

    good = {"f1": 0.6, "signal": {"prd": 20.0}}
    bad = {"f1": 0.6, "signal": {"prd": 90.0}}
    assert _selection_score(good, "combined") > _selection_score(bad, "combined")
    assert _selection_score(good, "f1") == _selection_score(bad, "f1")

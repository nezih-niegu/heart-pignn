"""End-to-end test over synthetic records: data -> training -> test."""

import numpy as np
import pytest
import torch

from heart_pignn.data import DataConfig, MITBIHBeatDataset, split_records
from heart_pignn.demo_data import generate_dataset
from heart_pignn.heuristics import N_RULE_FEATURES
from heart_pignn.model import ModelConfig
from heart_pignn.train import TrainConfig, run_test, run_training


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("mitdb")
    generate_dataset(path, n_records=8, n_beats=120, seed=1)
    return path


def test_splits_do_not_share_records(demo_dir):
    cfg = DataConfig()
    splits = {s: set(split_records(demo_dir, cfg, s)) for s in ("train", "val", "test")}
    assert splits["train"] and splits["test"]
    assert not splits["train"] & splits["val"]
    assert not splits["train"] & splits["test"]
    assert not splits["val"] & splits["test"]


def test_dataset_item_shapes(demo_dir):
    ds = MITBIHBeatDataset(demo_dir, split="train", config=DataConfig(), verbose=False)
    item = ds[0]
    assert item["x"].shape == (1, 360)
    assert item["rule_vec"].shape == (N_RULE_FEATURES,)
    assert item["y"].dtype == torch.int64
    assert 0 <= int(item["regime"]) < 4
    assert torch.isfinite(item["x"]).all()
    assert torch.isfinite(item["rule_vec"]).all()


def test_rule_features_vary_with_heart_rate(demo_dir):
    ds = MITBIHBeatDataset(demo_dir, split="train", config=DataConfig(), verbose=False)
    hr = ds.rule_features[:, 11]  # hr_norm
    assert np.ptp(hr) > 0.1, "synthetic records should span several heart rates"


def test_training_and_test_run(demo_dir, tmp_path):
    cfg = TrainConfig(
        data_root=str(demo_dir),
        output_dir=str(tmp_path / "ckpt"),
        epochs=2,
        batch_size=32,
        samples_per_epoch=128,
        patience=5,
        data=DataConfig(),
        model=ModelConfig(hidden_dim=16, msg_dim=16, node_emb_dim=8, n_layers=1, graph_steps=4),
    )
    result = run_training(cfg)
    assert len(result["history"]) == 2
    assert (tmp_path / "ckpt" / "best_model.pt").exists()

    run = run_test(cfg)
    assert 0.0 <= run["test_f1_macro"] <= 1.0
    assert (tmp_path / "ckpt" / "test_runs.json").exists()


def test_training_resumes_from_checkpoint(demo_dir, tmp_path, capsys):
    cfg = TrainConfig(
        data_root=str(demo_dir),
        output_dir=str(tmp_path / "resume"),
        epochs=1,
        batch_size=32,
        samples_per_epoch=64,
        data=DataConfig(),
        model=ModelConfig(hidden_dim=16, msg_dim=16, node_emb_dim=8, n_layers=1, graph_steps=4),
    )
    run_training(cfg)
    capsys.readouterr()
    run_training(cfg)
    assert "Resuming at epoch" in capsys.readouterr().out

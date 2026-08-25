"""Tests for the augmentation transforms and their interaction with the dataset."""

import numpy as np
import pytest
import torch

from heart_pignn.augment import (
    AugmentConfig,
    augment_window,
    baseline_wander,
    corrupt_input,
    sample_shift,
    time_warp,
)
from heart_pignn.data import DataConfig, MITBIHBeatDataset
from heart_pignn.demo_data import generate_dataset


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("mitdb_aug")
    generate_dataset(path, n_records=8, n_beats=150, seed=7)
    return path


def test_shift_is_bounded_and_centred():
    cfg = AugmentConfig(max_shift_sec=0.06)
    rng = np.random.default_rng(0)
    shifts = [sample_shift(cfg, 360, rng) for _ in range(500)]
    assert max(abs(s) for s in shifts) <= int(0.06 * 360)
    assert abs(np.mean(shifts)) < 3, "shift should be roughly zero-mean"
    assert len(set(shifts)) > 10, "shift should actually vary"


def test_shift_is_zero_when_disabled():
    rng = np.random.default_rng(0)
    cfg = AugmentConfig(enabled=False)
    assert all(sample_shift(cfg, 360, rng) == 0 for _ in range(20))


def test_time_warp_preserves_length_and_finiteness():
    rng = np.random.default_rng(1)
    window = np.random.randn(360, 2).astype(np.float32)
    for _ in range(20):
        out = time_warp(window, 0.06, rng)
        assert out.shape == window.shape
        assert np.isfinite(out).all()


def test_time_warp_does_not_zero_pad_edges():
    """Compressing must extend the edges, not leave a flatline the model can spot."""
    rng = np.random.default_rng(2)
    window = np.ones((360, 1), dtype=np.float32) * 3.0
    for _ in range(20):
        out = time_warp(window, 0.08, rng)
        assert out.min() > 1.0, "edges were zero-filled"


def test_augment_applies_same_geometry_to_all_channels():
    """Input and regression target must stay physically consistent."""
    rng = np.random.default_rng(3)
    base = np.random.randn(360, 1).astype(np.float32)
    window = np.concatenate([base, base * 2.0], axis=1)
    out = augment_window(window, AugmentConfig(), 360, rng)
    assert np.allclose(out[:, 1], out[:, 0] * 2.0, atol=1e-4)


def test_corrupt_input_changes_signal_but_stays_finite():
    rng = np.random.default_rng(4)
    x = np.zeros(360, dtype=np.float32)
    out = corrupt_input(x, AugmentConfig(), 360, rng)
    assert np.isfinite(out).all()
    assert out.std() > 0, "noise and wander should have been added"


def test_baseline_wander_is_low_frequency():
    rng = np.random.default_rng(5)
    w = baseline_wander(3600, AugmentConfig(baseline_wander_std=1.0), 360, rng)
    spectrum = np.abs(np.fft.rfft(w))
    freqs = np.fft.rfftfreq(len(w), 1 / 360)
    assert freqs[int(spectrum.argmax())] < 2.0, "wander should sit below 2 Hz"


def test_augmentation_makes_repeated_reads_differ(demo_dir):
    cfg = DataConfig(augment=True, augmentation=AugmentConfig())
    ds = MITBIHBeatDataset(demo_dir, split="train", config=cfg, verbose=False)
    a, b = ds[0]["x"], ds[0]["x"]
    assert not torch.allclose(a, b), "the same beat should differ across epochs"


def test_augmentation_off_is_deterministic(demo_dir):
    ds = MITBIHBeatDataset(demo_dir, split="train", config=DataConfig(augment=False), verbose=False)
    assert torch.allclose(ds[0]["x"], ds[0]["x"])


def test_reconstruct_target_stays_clean_under_augmentation(demo_dir):
    """Noise goes on the input only -- the target must not carry it."""
    cfg = DataConfig(augment=True, regression_target="reconstruct")
    ds = MITBIHBeatDataset(demo_dir, split="train", config=cfg, verbose=False)
    item = ds[0]
    x, y = item["x"].squeeze(0), item["y_signal"].squeeze(-1)
    assert not torch.allclose(x, y), "input should be noisier than the target"
    assert float((x - y).std()) > 0


def test_minority_classes_get_stronger_augmentation(demo_dir):
    cfg = DataConfig(augment=True, augmentation=AugmentConfig(minority_boost=2.5))
    ds = MITBIHBeatDataset(demo_dir, split="train", config=cfg, verbose=False)
    from heart_pignn.aami import CLASS_TO_IDX

    normal = ds.aug_boost[ds.labels == CLASS_TO_IDX["N"]]
    ventricular = ds.aug_boost[ds.labels == CLASS_TO_IDX["V"]]
    assert np.allclose(normal, 1.0)
    if len(ventricular):
        assert np.allclose(ventricular, 2.5)

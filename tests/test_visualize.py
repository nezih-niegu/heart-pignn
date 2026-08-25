"""Tests for the validation visualizer."""

import pytest

from heart_pignn.data import DataConfig, MITBIHBeatDataset
from heart_pignn.demo_data import generate_dataset
from heart_pignn.model import ModelConfig, PIGNNBeatModel
from heart_pignn.utils import pick_device
from heart_pignn.visualize import collect_predictions, render_summary, visualize_record

SMALL = ModelConfig(hidden_dim=16, msg_dim=16, node_emb_dim=8, n_layers=1, graph_steps=4)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    path = tmp_path_factory.mktemp("mitdb_vis")
    generate_dataset(path, n_records=8, n_beats=150, seed=11)
    ds = MITBIHBeatDataset(path, split="test", config=DataConfig(), verbose=False)
    return ds, PIGNNBeatModel(SMALL).eval(), pick_device("cpu")


def test_beats_in_record_is_time_ordered(setup):
    ds, _, _ = setup
    rec = ds.sample_records[0]
    idx = ds.beats_in_record(rec)
    positions = ds.sample_positions[idx]
    assert len(idx) > 0
    assert (positions[:-1] <= positions[1:]).all()


def test_collect_predictions_shapes(setup):
    ds, model, dev = setup
    rec = ds.sample_records[0]
    strip, beats, fs = collect_predictions(model, ds, dev, rec, n_beats=8)
    assert fs == ds.cfg.fs_target
    assert len(beats) == 8
    assert strip.ndim == 1 and len(strip) > ds.cfg.window_len
    for b in beats:
        assert 0 <= b.pred_class < 5
        assert 0.0 <= b.confidence <= 1.0
        assert b.recon.shape == b.target.shape
        assert 0 <= b.position < len(strip)


def test_beat_positions_land_inside_the_strip(setup):
    """Positions are offsets into the strip, not into the record."""
    ds, model, dev = setup
    rec = ds.sample_records[0]
    strip, beats, _ = collect_predictions(model, ds, dev, rec, n_beats=10)
    assert all(0 < b.position < len(strip) for b in beats)
    assert beats[0].position < beats[-1].position


def test_summary_png_is_written(setup, tmp_path):
    ds, model, dev = setup
    rec = ds.sample_records[0]
    strip, beats, fs = collect_predictions(model, ds, dev, rec, n_beats=6)
    out = render_summary(strip, beats, fs, tmp_path / "s.png", rec)
    assert out.exists() and out.stat().st_size > 5000


def test_visualize_record_writes_gif_and_png(setup, tmp_path):
    ds, model, dev = setup
    outputs = visualize_record(
        model, ds, dev, record=None, n_beats=5, out_dir=tmp_path, fps=8, make_gif=True
    )
    assert outputs["summary"].exists()
    assert outputs["realtime"].exists()
    assert outputs["realtime"].suffix == ".gif"
    assert outputs["realtime"].stat().st_size > 10000


def test_unknown_record_raises(setup):
    ds, model, dev = setup
    with pytest.raises(ValueError, match="no beats"):
        collect_predictions(model, ds, dev, "does-not-exist", n_beats=4)

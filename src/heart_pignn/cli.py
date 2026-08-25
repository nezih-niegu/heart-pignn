"""Command-line entry point."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="AAMI beat classifier: PIGNN + main.py rules joined through the attention latent.",
)

DEFAULT_DATA = "mit-bih-arrhythmia-database-1.0.0"
DEFAULT_CKPT = "checkpoints_pignn"


def _config(**kw):
    """Build a TrainConfig from CLI kwargs. Commands pass locals(), extras are ignored."""
    from .augment import AugmentConfig
    from .data import DataConfig
    from .model import ModelConfig
    from .train import TrainConfig

    aug_on = kw.get("augment", True)
    data = DataConfig(
        seed=kw["seed"],
        split_seed=kw.get("split_seed", 42),
        augment=aug_on,
        regression_target=kw.get("regression_target", "reconstruct"),
        augmentation=AugmentConfig(
            enabled=aug_on,
            max_shift_sec=kw.get("max_shift_ms", 60.0) / 1000.0,
            minority_boost=kw.get("minority_boost", 1.8),
        ),
    )

    fields = dict(
        data_root=kw["data_root"],
        output_dir=kw["output_dir"],
        epochs=kw.get("epochs", 1),
        batch_size=kw.get("batch_size", 128),
        lr=kw.get("lr", 1e-3),
        weight_decay=kw.get("weight_decay", 1e-4),
        rule_w=kw.get("rule_w", 0.3),
        phys_w=kw.get("phys_w", 0.05),
        recon_w=kw.get("recon_w", 0.5),
        imbalance_strategy=kw.get("imbalance_strategy", "sampler"),
        sampler_alpha=kw.get("sampler_alpha", 0.5),
        selection_metric=kw.get("selection_metric", "f1"),
        samples_per_epoch=kw.get("samples_per_epoch") or None,
        device=kw.get("device", "auto"),
        amp=kw.get("amp", True),
        seed=kw["seed"],
        data=data,
        model=ModelConfig(
            hidden_dim=kw.get("hidden_dim", 64),
            graph_steps=kw.get("graph_steps", 32),
            dropout=kw.get("dropout", 0.1),
        ),
    )
    # -1 means "let TrainConfig auto-detect" (0 on Windows, a few elsewhere).
    workers = kw.get("num_workers", -1)
    if workers is not None and workers >= 0:
        fields["num_workers"] = workers
    return TrainConfig(**fields)


@app.command()
def train(
    data_root: str = typer.Option(DEFAULT_DATA, help="Folder holding the MIT-BIH records"),
    output_dir: str = typer.Option(DEFAULT_CKPT, help="Where to store checkpoint and history"),
    epochs: int = typer.Option(30, help="Epochs for this session (added on top of the checkpoint)"),
    batch_size: int = typer.Option(128),
    lr: float = typer.Option(1e-3),
    weight_decay: float = typer.Option(1e-4, help="AdamW weight decay; raise to fight overfitting"),
    dropout: float = typer.Option(0.1, help="Dropout through the whole model; raise to fight overfitting"),
    hidden_dim: int = typer.Option(64),
    graph_steps: int = typer.Option(32, help="Graph steps per beat"),
    rule_w: float = typer.Option(0.3, help="Weight of the auxiliary rule head; 0 disables it"),
    phys_w: float = typer.Option(0.05, help="Weight of the physiological regularizer"),
    recon_w: float = typer.Option(0.5, help="Weight of the signal regression head; 0 disables it"),
    regression_target: str = typer.Option(
        "reconstruct", help="Regression target: reconstruct | cross_lead | none"
    ),
    augment: bool = typer.Option(True, help="Enable waveform augmentation (shift, warp, noise)"),
    max_shift_ms: float = typer.Option(60.0, help="Max R-peak crop jitter in milliseconds"),
    minority_boost: float = typer.Option(1.8, help="Augmentation multiplier for rare classes"),
    imbalance_strategy: str = typer.Option(
        "sampler", help="sampler | weights | both | none. 'both' double-corrects."
    ),
    sampler_alpha: float = typer.Option(0.5, help="1.0 fully balances, 0.5 is square-root balancing"),
    selection_metric: str = typer.Option("f1", help="f1 | combined (combined penalizes PRD)"),
    samples_per_epoch: int = typer.Option(20000, help="Beats sampled per epoch; 0 uses all"),
    num_workers: int = typer.Option(-1, help="DataLoader workers; -1 auto-detects (0 on Windows)"),
    split_seed: int = typer.Option(42, help="Record-split seed. KEEP FIXED across --seed repeats, or you benchmark different data."),
    device: str = typer.Option("auto", help="auto | cuda | cuda:0 | cpu"),
    amp: bool = typer.Option(True, help="Mixed precision on CUDA"),
    seed: int = typer.Option(42),
    test: bool = typer.Option(True, help="Evaluate on test when finished"),
    visualize: bool = typer.Option(True, help="Render the validation ECG monitor when finished"),
) -> None:
    """Train the model. Resumes only if a checkpoint already exists."""
    from .train import run_test, run_training

    cfg = _config(**locals())
    run_training(cfg)
    if test:
        run_test(cfg)
    if visualize:
        _run_visualize(cfg, record=None, n_beats=25, out_dir="figures", fps=20, gif=True)


@app.command()
def evaluate(
    data_root: str = typer.Option(DEFAULT_DATA),
    output_dir: str = typer.Option(DEFAULT_CKPT),
    batch_size: int = typer.Option(128),
    hidden_dim: int = typer.Option(64),
    graph_steps: int = typer.Option(32),
    num_workers: int = typer.Option(-1),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
) -> None:
    """Evaluate the best checkpoint on test and diff it against the previous run."""
    from .train import run_test

    run_test(_config(**locals(), augment=False))


@app.command()
def visualize(
    data_root: str = typer.Option(DEFAULT_DATA),
    output_dir: str = typer.Option(DEFAULT_CKPT),
    record: str = typer.Option("", help="Record to replay; empty picks the first test record"),
    n_beats: int = typer.Option(25, help="Consecutive beats to replay"),
    out_dir: str = typer.Option("figures", help="Where to write the GIF and PNG"),
    fps: int = typer.Option(20),
    gif: bool = typer.Option(True, help="Render the animation; False writes only the summary PNG"),
    split: str = typer.Option("val", help="val | test"),
    focus_class: str = typer.Option("", help="Seek a stretch rich in this AAMI class: N|S|V|F|Q"),
    hidden_dim: int = typer.Option(64),
    graph_steps: int = typer.Option(32),
    batch_size: int = typer.Option(128),
    num_workers: int = typer.Option(-1),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
) -> None:
    """Render the real-time ECG monitor and the final summary image."""
    cfg = _config(**locals(), augment=False)
    _run_visualize(cfg, record or None, n_beats, out_dir, fps, gif, split, focus_class)


def _run_visualize(cfg, record, n_beats, out_dir, fps, gif, split: str = "val",
                   focus_class: str = "") -> None:
    import torch

    from .data import DataConfig, MITBIHBeatDataset
    from .train import build_model
    from .utils import pick_device
    from .visualize import visualize_record

    ckpt_path = Path(cfg.output_dir) / "best_model.pt"
    if not ckpt_path.exists():
        raise typer.BadParameter(f"No checkpoint at {ckpt_path}")

    dev = pick_device(cfg.device)
    data_cfg = DataConfig(**{**cfg.data.__dict__, "augment": False})
    dataset = MITBIHBeatDataset(cfg.data_root, split=split, config=data_cfg)
    model = build_model(cfg, dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=False)["model_state"])

    from .aami import CLASS_TO_IDX
    fc = CLASS_TO_IDX[focus_class] if focus_class else None
    print(f"\nRendering the {split} ECG monitor...")
    visualize_record(model, dataset, dev, record, n_beats, out_dir, fps, gif, focus_class=fc)


@app.command()
def explain(
    data_root: str = typer.Option(DEFAULT_DATA),
    output_dir: str = typer.Option(DEFAULT_CKPT),
    batch_size: int = typer.Option(128),
    hidden_dim: int = typer.Option(64),
    graph_steps: int = typer.Option(32),
    max_batches: int = typer.Option(20),
    num_workers: int = typer.Option(-1),
    device: str = typer.Option("auto"),
    seed: int = typer.Option(42),
) -> None:
    """Report which conduction nodes dominate attention for each class."""
    import torch

    from .explain import contrast_against_normal, node_importance_by_class, print_node_report
    from .train import build_datasets, build_loaders, build_model
    from .utils import pick_device

    cfg = _config(**locals(), augment=False)
    dev = pick_device(cfg.device)
    loaders = build_loaders(cfg, build_datasets(cfg), dev)
    model = build_model(cfg, dev)

    ckpt_path = Path(cfg.output_dir) / "best_model.pt"
    if not ckpt_path.exists():
        raise typer.BadParameter(f"No checkpoint at {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=False)["model_state"])

    result = node_importance_by_class(model, loaders["test"], dev, max_batches)
    print_node_report(result)
    contrast_against_normal(result, "V")


@app.command("demo-data")
def demo_data(
    out_dir: str = typer.Option("demo-mitdb", help="Destination folder"),
    n_records: int = typer.Option(12),
    n_beats: int = typer.Option(220, help="Beats per record"),
    seed: int = typer.Option(0),
) -> None:
    """Create synthetic WFDB records to exercise the pipeline without downloading anything."""
    from .demo_data import generate_dataset

    generate_dataset(out_dir, n_records=n_records, seed=seed, n_beats=n_beats)


@app.command("rule-baseline")
def rule_baseline(
    data_root: str = typer.Option(DEFAULT_DATA),
    seed: int = typer.Option(42),
) -> None:
    """Measure how informative main.py's tree is on its own, before training anything."""
    from .data import DataConfig, MITBIHBeatDataset
    from .train import summarize_rule_baseline

    ds = MITBIHBeatDataset(data_root, split="test", config=DataConfig(seed=seed))
    summarize_rule_baseline(ds)


def main() -> None:
    """Entry point declared in [project.scripts]."""
    app()


if __name__ == "__main__":
    main()

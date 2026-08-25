"""Training pipeline, with the discipline of Modelo3.ipynb plus GPU and progress bars.

Three things carried over from Modelo3: capped class weights, resumable
checkpoints, and a historical test log that diffs each run against the previous
one.

One thing deliberately changed: **imbalance is corrected once, not twice.**
Modelo3 applied a balanced sampler *and* capped class weights simultaneously,
which double-corrects. On MIT-BIH the symptom is unmistakable -- class N ends up
with precision 0.95 and recall 0.55, i.e. the model over-predicts every minority
class. `imbalance_strategy` picks one mechanism; the default is the sampler
alone, with square-root weighting (`sampler_alpha=0.5`) rather than full
balancing, because a 66:1 ratio fully balanced means the same 900 S beats get
drawn thousands of times per epoch and memorized.
"""

from __future__ import annotations

import datetime
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from .aami import CLASS_NAMES
from .augment import AugmentConfig
from .data import DataConfig, MITBIHBeatDataset
from .graph import build_heart_graph
from .losses import BeatLoss
from .model import ModelConfig, PIGNNBeatModel
from .utils import (
    count_parameters,
    cuda_available_but_unused,
    describe_device,
    pick_device,
    read_json,
    set_seed,
    write_json,
)


def default_workers() -> int:
    """Windows and DataLoader multiprocessing do not get along; elsewhere use a few."""
    import os
    import platform

    if platform.system() == "Windows":
        return 0
    return min(4, max(0, (os.cpu_count() or 1) - 1))


@dataclass
class TrainConfig:
    data_root: str = "mit-bih-arrhythmia-database-1.0.0"
    output_dir: str = "checkpoints_pignn"
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    # "sampler" | "weights" | "both" | "none". "both" double-corrects; see module docstring.
    imbalance_strategy: str = "sampler"
    sampler_alpha: float = 0.5      # 1.0 = fully balanced, 0.5 = square-root balancing
    class_weight_cap: float = 6.0
    rule_w: float = 0.3
    phys_w: float = 0.05
    recon_w: float = 0.5
    label_smoothing: float = 0.05
    samples_per_epoch: int | None = 20000
    num_workers: int = field(default_factory=default_workers)
    seed: int = 42
    device: str = "auto"
    amp: bool = True                # mixed precision, CUDA only
    grad_clip: float = 1.0
    # "f1" | "combined". "combined" penalizes reconstruction PRD when picking a checkpoint.
    selection_metric: str = "f1"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


# ------------------------------------------------------------------- weighting


def compute_class_weights(dataset: MITBIHBeatDataset, cap: float = 6.0) -> torch.Tensor:
    """Inverse frequency with a cap.

    Without a cap, F can weigh two orders of magnitude more than N and the model
    learns to over-predict the rare class: exactly the failure documented in the
    first version of Modelo3.ipynb.
    """
    counts = Counter(int(i) for i in dataset.labels)
    total = sum(counts.values())
    raw = torch.zeros(len(CLASS_NAMES))
    for idx in range(len(CLASS_NAMES)):
        raw[idx] = total / (len(CLASS_NAMES) * max(counts.get(idx, 1), 1))

    capped = torch.clamp(raw, max=raw.min() * cap)
    capped = capped / capped.mean()

    print("Uncapped weights ->", {CLASS_NAMES[i]: round(float(w), 2) for i, w in enumerate(raw)})
    print("Capped weights   ->", {CLASS_NAMES[i]: round(float(w), 2) for i, w in enumerate(capped)})
    return capped


def make_balanced_sampler(
    dataset: MITBIHBeatDataset, samples_per_epoch: int | None = None, alpha: float = 0.5
) -> WeightedRandomSampler:
    """Sampling weights proportional to 1/count**alpha.

    alpha=1 balances the classes exactly, which sounds right and usually is not:
    at a 66:1 ratio it draws the rarest beats thousands of times per epoch and the
    model memorizes those specific waveforms. alpha=0.5 (square-root) lifts the
    rare classes substantially while keeping some of the real prior.
    """
    counts = Counter(int(i) for i in dataset.labels)
    weights = [1.0 / (counts[int(i)] ** alpha) for i in dataset.labels]
    return WeightedRandomSampler(
        weights=weights, num_samples=samples_per_epoch or len(dataset), replacement=True
    )


# ----------------------------------------------------------------- signal stats


def _new_signal_stats() -> dict:
    return {"sse": 0.0, "sst": 0.0, "n": 0, "corr_sum": 0.0, "corr_n": 0}


@torch.no_grad()
def _accumulate_signal(stats: dict, pred, target, mask) -> None:
    """Accumulate partial sums instead of holding every waveform in memory."""
    keep = mask.bool()
    if not bool(keep.any()):
        return
    p, t = pred[keep].float(), target[keep].float()
    stats["sse"] += float(((p - t) ** 2).sum())
    stats["sst"] += float((t**2).sum())
    stats["n"] += int(p.numel())
    pc = p - p.mean(dim=1, keepdim=True)
    tc = t - t.mean(dim=1, keepdim=True)
    corr = (pc * tc).sum(dim=1) / (pc.norm(dim=1) * tc.norm(dim=1) + 1e-8)
    stats["corr_sum"] += float(corr.sum())
    stats["corr_n"] += int(corr.numel())


def _finalize_signal(stats: dict) -> dict:
    """RMSE, PRD and Pearson -- the same metrics Modelo2 reported."""
    if stats["n"] == 0:
        return {"rmse": 0.0, "prd": 0.0, "corr": 0.0, "n_samples": 0}
    return {
        "rmse": float(np.sqrt(stats["sse"] / stats["n"])),
        "prd": float(100.0 * np.sqrt(stats["sse"] / (stats["sst"] + 1e-8))),
        "corr": float(stats["corr_sum"] / max(stats["corr_n"], 1)),
        "n_samples": stats["corr_n"],
    }


# ----------------------------------------------------------------------- loops


def _to_device(batch: dict, device: torch.device) -> dict:
    non_blocking = device.type == "cuda"
    return {k: v.to(device, non_blocking=non_blocking) for k, v in batch.items()}


def train_one_epoch(
    model, loader, optimizer, criterion, device, grad_clip=1.0, scaler=None, epoch: int = 0
) -> dict:
    model.train()
    total, n = 0.0, 0
    preds, labels = [], []
    use_amp = scaler is not None and device.type == "cuda"

    bar = tqdm(loader, desc=f"epoch {epoch:03d} train", unit="batch", leave=False)
    for batch in bar:
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(batch["x"], batch["rule_vec"], batch["regime"])
            losses = criterion(
                out, batch["y"], batch["rule_target"], batch.get("y_signal"),
                batch.get("signal_mask"),
            )

        if use_amp:
            scaler.scale(losses["loss"]).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses["loss"].backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        bs = batch["y"].shape[0]
        total += float(losses["loss"].detach()) * bs
        n += bs
        preds.extend(out["logits"].argmax(dim=1).detach().cpu().tolist())
        labels.extend(batch["y"].detach().cpu().tolist())
        bar.set_postfix(loss=f"{total / max(n, 1):.4f}")

    return {
        "loss": total / max(n, 1),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc: str = "eval") -> dict:
    model.eval()
    total, n = 0.0, 0
    preds, labels = [], []
    rule_preds, rule_targets = [], []
    signal_stats = _new_signal_stats()

    for batch in tqdm(loader, desc=desc, unit="batch", leave=False):
        batch = _to_device(batch, device)
        out = model(batch["x"], batch["rule_vec"], batch["regime"])
        losses = criterion(
            out, batch["y"], batch["rule_target"], batch.get("y_signal"), batch.get("signal_mask")
        )
        if "signal" in out and "y_signal" in batch:
            _accumulate_signal(signal_stats, out["signal"], batch["y_signal"], batch["signal_mask"])

        bs = batch["y"].shape[0]
        total += float(losses["loss"]) * bs
        n += bs
        preds.extend(out["logits"].argmax(dim=1).cpu().tolist())
        labels.extend(batch["y"].cpu().tolist())
        rule_preds.append((torch.sigmoid(out["rule_logits"]) > 0.5).float().cpu())
        rule_targets.append(batch["rule_target"].cpu())

    rule_acc = (
        float((torch.cat(rule_preds) == torch.cat(rule_targets)).float().mean())
        if rule_preds
        else 0.0
    )

    return {
        "loss": total / max(n, 1),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "rule_acc": rule_acc,
        "signal": _finalize_signal(signal_stats),
        "preds": preds,
        "labels": labels,
    }


# -------------------------------------------------------------------- builders


def build_datasets(cfg: TrainConfig) -> dict[str, MITBIHBeatDataset]:
    out = {}
    for split in ("train", "val", "test"):
        data_cfg = DataConfig(**{**asdict(cfg.data), "augment": cfg.data.augment and split == "train"})
        data_cfg.augmentation = AugmentConfig(**asdict(cfg.data.augmentation))
        out[split] = MITBIHBeatDataset(cfg.data_root, split=split, config=data_cfg)
    return out


def build_loaders(cfg: TrainConfig, datasets: dict, device: torch.device | None = None) -> dict:
    pin = device is not None and device.type == "cuda"
    common = {
        "num_workers": cfg.num_workers,
        "pin_memory": pin,
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.imbalance_strategy in ("sampler", "both"):
        train_loader = DataLoader(
            datasets["train"],
            batch_size=cfg.batch_size,
            sampler=make_balanced_sampler(datasets["train"], cfg.samples_per_epoch, cfg.sampler_alpha),
            **common,
        )
    else:
        train_loader = DataLoader(
            datasets["train"], batch_size=cfg.batch_size, shuffle=True, **common
        )
    return {
        "train": train_loader,
        "val": DataLoader(datasets["val"], batch_size=cfg.batch_size, shuffle=False, **common),
        "test": DataLoader(datasets["test"], batch_size=cfg.batch_size, shuffle=False, **common),
    }


def build_model(cfg: TrainConfig, device: torch.device) -> PIGNNBeatModel:
    model_cfg = ModelConfig(
        **{
            **asdict(cfg.model),
            "n_rule_labels": cfg.data.n_rule_labels,
            "n_signal_outputs": cfg.data.n_signal_outputs,
            "signal_len": cfg.data.window_len,
            "use_regression": cfg.data.n_signal_outputs > 0,
        }
    )
    model = PIGNNBeatModel(model_cfg, graph=build_heart_graph()).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    return model


def _selection_score(metrics: dict, mode: str) -> float:
    """A single number for picking a checkpoint when there are two objectives."""
    if mode == "combined":
        prd = min(metrics["signal"]["prd"] / 100.0, 1.0)
        return metrics["f1"] - 0.5 * prd
    return metrics["f1"]


# -------------------------------------------------------------------- training


def run_training(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    print(f"Device: {describe_device(device)}")
    if cuda_available_but_unused(device):
        print("  note: a CUDA GPU is visible but device='cpu' was requested.")
    if device.type != "cuda" and not torch.cuda.is_available():
        print("  note: no CUDA GPU detected. For GPU, install a CUDA build of torch:")
        print("        uv pip install torch --index-url https://download.pytorch.org/whl/cu124")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"
    history_path = out_dir / "history.json"

    datasets = build_datasets(cfg)
    loaders = build_loaders(cfg, datasets, device)
    model = build_model(cfg, device)

    if cfg.imbalance_strategy in ("weights", "both"):
        weights = compute_class_weights(datasets["train"], cfg.class_weight_cap).to(device)
    else:
        weights = None
        print(
            f"Imbalance handled by sampler only (alpha={cfg.sampler_alpha}); "
            "class weights off to avoid double-correcting."
        )

    criterion = BeatLoss(
        weights, cfg.rule_w, cfg.phys_w, cfg.recon_w, label_smoothing=cfg.label_smoothing
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None
    if use_amp:
        print("Mixed precision (AMP) enabled.")

    start_epoch, best_score = 1, -1e9
    history = read_json(history_path, default=[]) or []

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        if scaler is not None and ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("val_score", ckpt.get("val_f1", -1e9)))
        print(f"Checkpoint found. Resuming at epoch {start_epoch} (best score={best_score:.4f})")
    else:
        print("No previous checkpoint. Training from scratch.")

    patience_counter = 0
    end_epoch = start_epoch + cfg.epochs - 1

    for epoch in range(start_epoch, end_epoch + 1):
        tr = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, cfg.grad_clip, scaler, epoch
        )
        va = evaluate(model, loaders["val"], criterion, device, desc=f"epoch {epoch:03d} val")
        scheduler.step(va["loss"])
        score = _selection_score(va, cfg.selection_metric)
        sig = va["signal"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_f1": tr["f1"],
                "val_loss": va["loss"],
                "val_f1": va["f1"],
                "val_score": score,
                "val_rule_acc": va["rule_acc"],
                "val_prd": sig["prd"],
                "val_corr": sig["corr"],
                "val_rmse": sig["rmse"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        write_json(history_path, history)

        gap = tr["f1"] - va["f1"]
        print(
            f"Epoch {epoch:03d} | train loss={tr['loss']:.4f} F1={tr['f1']:.4f} "
            f"| val loss={va['loss']:.4f} F1={va['f1']:.4f} "
            f"PRD={sig['prd']:.1f}% r={sig['corr']:.3f} rules={va['rule_acc']:.3f} "
            f"| gap={gap:+.3f}"
        )
        if gap > 0.35:
            print("  warning: large train/val F1 gap -- the model is memorizing. "
                  "Consider stronger augmentation or a lower sampler_alpha.")

        if score > best_score:
            best_score = score
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict() if scaler is not None else None,
                    "epoch": epoch,
                    "val_loss": va["loss"],
                    "val_f1": va["f1"],
                    "val_score": score,
                    "val_signal": sig,
                    "model_config": asdict(model.config),
                    "data_config": asdict(cfg.data),
                },
                ckpt_path,
            )
            print(f"  -> saved new best model ({cfg.selection_metric}={best_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {cfg.patience} epochs)")
                break

    return {"best_score": best_score, "best_val_f1": best_score, "history": history}


# ------------------------------------------------------------------------ test


def run_test(cfg: TrainConfig, loaders: dict | None = None, model=None) -> dict:
    """Evaluate on test and diff against the previously recorded run."""
    device = pick_device(cfg.device)
    out_dir = Path(cfg.output_dir)
    ckpt_path = out_dir / "best_model.pt"
    log_path = out_dir / "test_runs.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first.")

    if loaders is None:
        loaders = build_loaders(cfg, build_datasets(cfg), device)
    if model is None:
        model = build_model(cfg, device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    criterion = BeatLoss(None, cfg.rule_w, cfg.phys_w, cfg.recon_w)
    res = evaluate(model, loaders["test"], criterion, device, desc="test")

    report = classification_report(
        res["labels"], res["preds"], target_names=CLASS_NAMES, zero_division=0, output_dict=True
    )
    per_class_f1 = {c: round(report[c]["f1-score"], 4) for c in CLASS_NAMES}
    support = {c: int(report[c]["support"]) for c in CLASS_NAMES}

    print("\n=== Test evaluation (current run) ===")
    print(f"Checkpoint epoch: {ckpt['epoch']}")
    print(f"Test loss: {res['loss']:.4f}")
    print(f"Test F1 (macro): {res['f1']:.4f}")
    print("\nClassification report:")
    print(classification_report(res["labels"], res["preds"], target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(res["labels"], res["preds"]))

    thin = [c for c, s in support.items() if 0 < s < 30]
    if thin:
        print(
            f"\nnote: classes {thin} have under 30 test beats. Their F1 is dominated by "
            "sampling noise, and so is the macro average that includes them."
        )

    sig = res["signal"]
    if sig["n_samples"] > 0:
        print("\n=== Signal regression (test) ===")
        print(f"RMSE: {sig['rmse']:.4f}")
        print(f"PRD:  {sig['prd']:.2f}%")
        print(f"Pearson: {sig['corr']:.4f}")
        print(f"Beats with a valid target: {sig['n_samples']}")
        if sig["prd"] > 90:
            print("  warning: PRD near 100% means the decoder is emitting a near-constant. "
                  "With regression_target='cross_lead' this is the known collapse; see decoder.py.")

    previous = read_json(log_path, default=[]) or []
    current = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "epoch": int(ckpt["epoch"]),
        "test_loss": round(res["loss"], 4),
        "test_f1_macro": round(res["f1"], 4),
        "per_class_f1": per_class_f1,
        "support": support,
        "rule_head_acc": round(res["rule_acc"], 4),
        "signal_prd": round(sig["prd"], 4),
        "signal_corr": round(sig["corr"], 4),
        "signal_rmse": round(sig["rmse"], 4),
    }

    print("\n" + "=" * 60)
    if previous:
        last = previous[-1]
        d_f1 = current["test_f1_macro"] - last["test_f1_macro"]
        print("COMPARISON WITH THE PREVIOUS RUN")
        print("=" * 60)
        print(f"{'Metric':<15}{'Previous':>12}{'Current':>12}{'Change':>12}")
        print(f"{'Test loss':<15}{last['test_loss']:>12.4f}{current['test_loss']:>12.4f}"
              f"{current['test_loss'] - last['test_loss']:>+12.4f}")
        print(f"{'Test F1 macro':<15}{last['test_f1_macro']:>12.4f}{current['test_f1_macro']:>12.4f}{d_f1:>+12.4f}")
        for c in CLASS_NAMES:
            prev_f1 = last["per_class_f1"].get(c, 0.0)
            curr_f1 = current["per_class_f1"].get(c, 0.0)
            print(f"  F1 class {c:<3}: {prev_f1:.4f} -> {curr_f1:.4f}  ({curr_f1 - prev_f1:+.4f})")
        if "signal_prd" in last:
            d_prd = current["signal_prd"] - last["signal_prd"]
            print(f"{'Signal PRD':<15}{last['signal_prd']:>12.2f}{current['signal_prd']:>12.2f}{d_prd:>+12.2f}")
        verdict = "IMPROVED" if d_f1 > 0 else ("REGRESSED" if d_f1 < 0 else "unchanged")
        print(f"\nThe model {verdict} relative to the previous run.")
    else:
        print("No previous runs recorded. This is the first.")
    print("=" * 60)

    previous.append(current)
    write_json(log_path, previous)
    return current


def summarize_rule_baseline(dataset: MITBIHBeatDataset) -> dict[str, float]:
    """Cheap reference: how informative main.py's tree is on its own."""
    agreement = dataset.rule_agreement()
    print("\nAgreement between main.py's tree and non-normal beats:")
    for name, value in agreement.items():
        print(f"  {name:<14}: {value:.3f}")
    majority = float(np.mean(np.asarray(dataset.labels) == 0))
    print(f"  majority baseline (predict 'N' always): {majority:.3f} accuracy")
    return agreement

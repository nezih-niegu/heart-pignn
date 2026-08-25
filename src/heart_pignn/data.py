"""MIT-BIH beat-level dataset, with the rhythm context the rule tree consumes.

Each sample carries five things:

- `x`            beat window centred on the R peak, [1, T]
- `y_signal`     regression target, [T, 1] (see `regression_target`)
- `signal_mask`  1 if that beat has a valid regression target, 0 otherwise
- `rule_vec`     main.py's soft tree evaluated on the RR context, [N_RULE_FEATURES]
- `regime`       rhythm regime index (normal / brady / tachy / irregular)
- `y`            cardiologist-annotated AAMI class (the classification target)

Splits are by record, never by beat: two windows from the same patient share
morphology, and mixing them across train and test inflates metrics while nothing
in the code looks wrong.

On beat timing
--------------
R positions come from the .atr file, as in Modelo3.ipynb and in essentially all
MIT-BIH classification literature. That assumes R detection is solved. What is
carefully avoided is using annotation *symbols* to build features: the symbols
are the label. `rr_source="detected"` swaps annotated positions for `find_peaks`
detection if you want to measure the pipeline without that assumption.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import wfdb
from scipy.signal import find_peaks, resample
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .aami import AAMI_MAP, CLASS_NAMES, CLASS_TO_IDX, MINORITY_CLASSES
from .augment import AugmentConfig, augment_window, corrupt_input, sample_shift
from .heuristics import (
    N_RULE_FEATURES,
    RuleThresholds,
    SoftTemperatures,
    regime_index,
    rhythm_stats_from_rr,
    rule_labels,
    soft_rule_features,
)


@dataclass
class DataConfig:
    fs_target: int = 360
    window_sec: float = 1.0
    pre_r_sec: float = 0.4
    rr_context: int = 10           # RR intervals on each side of the beat
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42
    split_seed: int = 42  # controls the record split ONLY; keep fixed across training seeds
    channel: int = 0
    rr_source: str = "annotations"  # "annotations" | "detected"
    use_local_rr: bool = True
    include_hypoxemia: bool = False  # MIT-BIH carries no real SpO2
    max_samples: int | None = None   # cap on beats per split
    augment: bool = False
    augmentation: AugmentConfig = field(default_factory=AugmentConfig)
    # Regression head target:
    #   "reconstruct" -> rebuild the model's own clean window (denoising AE)
    #   "cross_lead"  -> predict `target_channel` from `channel`. Collapses on
    #                    MIT-BIH (PRD ~101%, r ~0.02); see decoder.py.
    #   "none"        -> no regression head
    regression_target: str = "reconstruct"
    target_channel: int = 1

    @property
    def window_len(self) -> int:
        return int(self.window_sec * self.fs_target)

    @property
    def pre_r(self) -> int:
        return int(self.pre_r_sec * self.fs_target)

    @property
    def n_rule_labels(self) -> int:
        return 4 if self.include_hypoxemia else 3

    @property
    def n_signal_outputs(self) -> int:
        return 0 if self.regression_target == "none" else 1


def split_records(mit_path: Path, cfg: DataConfig, split: str) -> list[str]:
    """Deterministically assign records to train/val/test."""
    import random as _random

    records = sorted({f.stem for f in Path(mit_path).glob("*.dat")})
    if not records:
        raise FileNotFoundError(
            f"No .dat records found in {mit_path}. Download MIT-BIH from PhysioNet "
            "or generate synthetic data with 'heart-pignn demo-data'."
        )
    rng = _random.Random(cfg.split_seed)
    rng.shuffle(records)
    n = len(records)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    if split == "train":
        return records[:n_train]
    if split == "val":
        return records[n_train : n_train + n_val]
    if split == "test":
        return records[n_train + n_val :]
    raise ValueError(f"unknown split: {split}")


class MITBIHBeatDataset(Dataset):
    def __init__(
        self,
        mit_path: str | Path,
        split: str = "train",
        config: DataConfig | None = None,
        thresholds: RuleThresholds | None = None,
        temperatures: SoftTemperatures | None = None,
        verbose: bool = True,
    ):
        self.cfg = config or DataConfig()
        self.split = split
        self.mit_path = Path(mit_path)
        self.thresholds = thresholds or RuleThresholds()
        self.temperatures = temperatures or SoftTemperatures()
        self.augment = self.cfg.augment and split == "train"
        self.aug_cfg = self.cfg.augmentation
        self._rng = np.random.default_rng(self.cfg.seed)

        self.records = split_records(self.mit_path, self.cfg, split)

        self._signal_cache: dict[str, np.ndarray] = {}
        self.sample_records: list[str] = []
        self.sample_positions: list[int] = []
        self.labels: list[int] = []
        rule_rows: list[np.ndarray] = []
        regimes: list[int] = []
        rule_label_rows: list[np.ndarray] = []

        iterator = tqdm(
            self.records, desc=f"indexing {split}", unit="rec", disable=not verbose, leave=False
        )
        for rec_name in iterator:
            parsed = self._index_record(rec_name)
            if parsed is None:
                continue
            positions, ys, rules, regs, rlabels = parsed
            self.sample_records.extend([rec_name] * len(positions))
            self.sample_positions.extend(positions)
            self.labels.extend(ys)
            rule_rows.extend(rules)
            regimes.extend(regs)
            rule_label_rows.extend(rlabels)

        if not self.labels:
            raise RuntimeError(f"split '{split}' came out empty. Check the MIT-BIH path.")

        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.rule_features = np.stack(rule_rows).astype(np.float32)
        self.regimes = np.asarray(regimes, dtype=np.int64)
        self.rule_targets = np.stack(rule_label_rows).astype(np.float32)
        self.sample_positions = np.asarray(self.sample_positions, dtype=np.int64)

        if self.cfg.max_samples is not None and len(self.labels) > self.cfg.max_samples:
            self._subsample(self.cfg.max_samples)

        # Per-beat augmentation strength: rare classes get pushed harder, because
        # they are the ones the balanced sampler repeats most within an epoch.
        minority_idx = {CLASS_TO_IDX[c] for c in MINORITY_CLASSES}
        self.aug_boost = np.where(
            np.isin(self.labels, list(minority_idx)), self.aug_cfg.minority_boost, 1.0
        ).astype(np.float32)

        if verbose:
            dist = Counter(CLASS_NAMES[i] for i in self.labels)
            print(f"[{split:5s}] {len(self.records)} records -> {len(self.labels)} beats")
            print(f"          distribution: {dict(dist)}")
            if self.augment:
                print(
                    f"          augmentation on: +/-{self.aug_cfg.max_shift_sec * 1000:.0f} ms shift, "
                    f"+/-{self.aug_cfg.time_warp_pct * 100:.0f}% warp, "
                    f"{self.aug_cfg.minority_boost:.1f}x on {list(MINORITY_CLASSES)}"
                )

    # ---------------------------------------------------------------- indexing

    def _r_peak_positions(self, rec_name: str) -> tuple[np.ndarray, np.ndarray, int]:
        """Return (positions, symbols, fs) at the record's original sampling rate."""
        rec_path = str(self.mit_path / rec_name)
        ann = wfdb.rdann(rec_path, "atr")
        header = wfdb.rdheader(rec_path)
        fs = int(header.fs)

        if self.cfg.rr_source == "detected":
            signal, _ = self._load_signal(rec_name, return_fs=True)
            peaks, _ = find_peaks(
                signal[:, self.cfg.channel], distance=int(0.25 * self.cfg.fs_target), height=0.5
            )
            ann_res = (ann.sample * self.cfg.fs_target / fs).astype(np.int64)
            if len(peaks) == 0:
                return ann.sample.astype(np.int64), np.asarray(ann.symbol), fs
            idx = np.clip(np.searchsorted(peaks, ann_res), 0, len(peaks) - 1)
            return peaks[idx], np.asarray(ann.symbol), self.cfg.fs_target

        return ann.sample.astype(np.int64), np.asarray(ann.symbol), fs

    def _index_record(self, rec_name: str):
        try:
            positions, symbols, fs = self._r_peak_positions(rec_name)
        except Exception as exc:  # corrupt or incomplete record
            print(f"  warning: could not read {rec_name} ({exc}); skipping")
            return None

        if len(positions) < 5:
            return None

        rr_ms = np.diff(positions) / fs * 1000.0
        scale = self.cfg.fs_target / fs

        out_pos, out_y, out_rules, out_reg, out_rlabels = [], [], [], [], []
        w = self.cfg.rr_context

        for i, symbol in enumerate(symbols):
            aami = AAMI_MAP.get(str(symbol))
            if aami is None:
                continue  # rhythm marker or non-clinical symbol

            lo = max(0, i - w)
            hi = min(len(rr_ms), i + w)
            context = rr_ms[lo:hi]
            if context.size < 3:
                continue

            stats = rhythm_stats_from_rr(
                context,
                rr_prev=float(rr_ms[i - 1]) if i >= 1 else 0.0,
                rr_next=float(rr_ms[i]) if i < len(rr_ms) else 0.0,
                spo2_mean=None,  # see the leakage note in heuristics.py
                thresholds=self.thresholds,
            )

            out_pos.append(int(round(positions[i] * scale)))
            out_y.append(CLASS_TO_IDX[aami])
            out_rules.append(soft_rule_features(stats, self.temperatures, self.cfg.use_local_rr))
            out_reg.append(regime_index(stats))
            out_rlabels.append(rule_labels(stats, self.cfg.include_hypoxemia))

        if not out_pos:
            return None
        return out_pos, out_y, out_rules, out_reg, out_rlabels

    def _subsample(self, max_samples: int) -> None:
        rng = np.random.default_rng(self.cfg.seed)
        keep = rng.choice(len(self.labels), size=max_samples, replace=False)
        keep.sort()
        self.labels = self.labels[keep]
        self.rule_features = self.rule_features[keep]
        self.regimes = self.regimes[keep]
        self.rule_targets = self.rule_targets[keep]
        self.sample_positions = self.sample_positions[keep]
        self.sample_records = [self.sample_records[i] for i in keep]

    # ------------------------------------------------------------------ signal

    def _load_signal(self, rec_name: str, return_fs: bool = False):
        """Return the full record as [T, C], resampled and per-channel normalized.

        All channels are cached, not just the input one: the regression head needs
        the target channel, and re-reading the .dat per beat would be absurdly slow.
        """
        if rec_name not in self._signal_cache:
            record = wfdb.rdrecord(str(self.mit_path / rec_name))
            sig = np.atleast_2d(record.p_signal.astype(np.float32))
            if sig.ndim == 1:
                sig = sig[:, None]
            if record.fs != self.cfg.fs_target:
                sig = resample(sig, int(sig.shape[0] * self.cfg.fs_target / record.fs), axis=0)
            sig = ((sig - sig.mean(axis=0)) / (sig.std(axis=0) + 1e-8)).astype(np.float32)
            self._signal_cache[rec_name] = sig
        sig = self._signal_cache[rec_name]
        return (sig, self.cfg.fs_target) if return_fs else (sig, None)

    def _window(self, rec_name: str, center: int) -> np.ndarray:
        """Window [win_len, C] centred on the beat, edge-padded at record boundaries."""
        signal, _ = self._load_signal(rec_name)
        start = center - self.cfg.pre_r
        end = start + self.cfg.window_len
        n_ch = signal.shape[1]
        if start < 0 or end > len(signal):
            window = np.zeros((self.cfg.window_len, n_ch), dtype=np.float32)
            src_start, src_end = max(0, start), min(len(signal), end)
            dst_start = src_start - start
            window[dst_start : dst_start + (src_end - src_start)] = signal[src_start:src_end]
            return window
        return signal[start:end].copy()

    # --------------------------------------------------------------------- API

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        rec_name = self.sample_records[idx]
        center = int(self.sample_positions[idx])
        boost = float(self.aug_boost[idx]) if self.augment else 1.0

        # The shift is applied by moving the crop centre *into the record*, so no
        # zero padding is introduced -- unlike np.roll, which would wrap the tail
        # of the beat around to the front.
        if self.augment:
            center += sample_shift(self.aug_cfg, self.cfg.fs_target, self._rng, boost)

        window = self._window(rec_name, center)  # [T, C]
        if self.augment:
            window = augment_window(window, self.aug_cfg, self.cfg.fs_target, self._rng, boost)

        x_np = window[:, self.cfg.channel]
        mode = self.cfg.regression_target

        if mode == "cross_lead" and window.shape[1] > self.cfg.target_channel:
            y_np = window[:, self.cfg.target_channel].copy()
            has_target = 1.0
        elif mode == "reconstruct":
            y_np = x_np.copy()  # clean target: only the input gets corrupted
            has_target = 1.0
        else:
            y_np = np.zeros_like(x_np)
            has_target = 0.0

        if self.augment:
            x_np = corrupt_input(x_np, self.aug_cfg, self.cfg.fs_target, self._rng, boost)

        x = torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.float32)).unsqueeze(0)
        y_signal = torch.from_numpy(np.ascontiguousarray(y_np, dtype=np.float32)).unsqueeze(-1)
        return {
            "x": x,
            "y_signal": y_signal,
            "signal_mask": torch.tensor(has_target, dtype=torch.float32),
            "rule_vec": torch.from_numpy(self.rule_features[idx]),
            "regime": torch.tensor(self.regimes[idx], dtype=torch.long),
            "rule_target": torch.from_numpy(self.rule_targets[idx]),
            "y": torch.tensor(self.labels[idx], dtype=torch.long),
        }

    # --------------------------------------------------------------- utilities

    def class_counts(self) -> Counter:
        return Counter(CLASS_NAMES[i] for i in self.labels)

    def beats_in_record(self, rec_name: str) -> np.ndarray:
        """Indices of this record's beats, ordered in time. Used by the visualizer."""
        idx = np.array([i for i, r in enumerate(self.sample_records) if r == rec_name])
        return idx[np.argsort(self.sample_positions[idx])] if len(idx) else idx

    def rule_agreement(self) -> dict[str, float]:
        """How often main.py's verdict lines up with a non-normal AAMI label.

        Useful as a reference: if the tree only fires on the majority class, the
        latent is doing essentially all the work.
        """
        out = {}
        names = ["bradycardia", "tachycardia", "arrhythmia"][: self.rule_targets.shape[1]]
        for k, name in enumerate(names):
            flagged = self.rule_targets[:, k] > 0.5
            if flagged.sum() == 0:
                out[name] = 0.0
                continue
            abnormal = self.labels != CLASS_TO_IDX["N"]
            out[name] = float((flagged & abnormal).sum() / max(flagged.sum(), 1))
        return out


def assert_rule_dim(dataset: MITBIHBeatDataset) -> None:
    assert dataset.rule_features.shape[1] == N_RULE_FEATURES, (
        f"rule vector has {dataset.rule_features.shape[1]} dims, expected {N_RULE_FEATURES}"
    )

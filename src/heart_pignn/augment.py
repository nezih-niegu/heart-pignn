"""Waveform augmentation, built around time shifting.

The imbalance in MIT-BIH is brutal: in a typical split there are ~65k N beats
and ~800 F beats. A balanced sampler fixes the *ratio* but not the *variety* --
it draws the same 800 F beats forty times per epoch, and the model memorizes
them. That is visible as train F1 0.98 against val F1 0.55.

Augmentation attacks the variety problem instead, and shifting is the most
useful transform here: the R peak sits at a fixed offset in every window, so a
model can learn "the spike is always at sample 144" and lean on that instead of
on morphology. Jittering the crop centre by a few tens of milliseconds removes
that crutch and forces the network to actually look at the wave shape.

Two rules the transforms follow:

- Geometric transforms (shift, time warp, amplitude) apply to the *whole*
  multi-channel window, before it is split into input and regression target, so
  input and target stay physically consistent.
- Corrupting transforms (noise, baseline wander) apply to the input only. Asking
  the model to predict noise nobody can predict just teaches it to output the
  mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentConfig:
    """Augmentation intensity. All ranges are sampled uniformly."""

    enabled: bool = True
    # Crop-centre jitter. Applied by shifting the window into the record, not by
    # padding, so no zero-filled edges are ever introduced.
    max_shift_sec: float = 0.06        # +/- 60 ms
    # Time warp: mild local stretch/compression of the beat.
    time_warp_pct: float = 0.06        # +/- 6 %
    amplitude_range: tuple[float, float] = (0.80, 1.25)
    noise_std: float = 0.015
    baseline_wander_std: float = 0.08
    baseline_wander_hz: float = 0.6
    # Rare classes get the transforms applied more aggressively, since they are
    # the ones the sampler repeats most.
    minority_boost: float = 1.8

    def shift_samples(self, fs: int) -> int:
        return int(self.max_shift_sec * fs)


def sample_shift(cfg: AugmentConfig, fs: int, rng: np.random.Generator, boost: float = 1.0) -> int:
    """Draw a crop-centre offset in samples. Returned to the dataset, not applied here."""
    if not cfg.enabled or cfg.max_shift_sec <= 0:
        return 0
    span = int(cfg.shift_samples(fs) * boost)
    return int(rng.integers(-span, span + 1)) if span > 0 else 0


def time_warp(window: np.ndarray, pct: float, rng: np.random.Generator) -> np.ndarray:
    """Resample the window by a small factor, then crop or pad back to length.

    Simulates the beat-to-beat variation in QRS width that a fixed-length window
    otherwise hides.
    """
    if pct <= 0:
        return window
    factor = 1.0 + float(rng.uniform(-pct, pct))
    n = window.shape[0]
    warped_len = max(8, int(round(n * factor)))
    src = np.linspace(0.0, n - 1.0, warped_len)
    warped = np.empty((warped_len, window.shape[1]), dtype=np.float32)
    for c in range(window.shape[1]):
        warped[:, c] = np.interp(src, np.arange(n), window[:, c])

    if warped_len == n:
        return warped
    if warped_len > n:
        start = (warped_len - n) // 2
        return warped[start : start + n]
    out = np.zeros_like(window)
    start = (n - warped_len) // 2
    out[start : start + warped_len] = warped
    # Extend the edges instead of leaving zeros, which would look like flatline.
    out[:start] = warped[0]
    out[start + warped_len :] = warped[-1]
    return out


def baseline_wander(n: int, cfg: AugmentConfig, fs: int, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency drift, the artefact that respiration puts on every real ECG."""
    if cfg.baseline_wander_std <= 0:
        return np.zeros(n, dtype=np.float32)
    t = np.arange(n, dtype=np.float32) / fs
    phase = float(rng.uniform(0, 2 * np.pi))
    freq = float(rng.uniform(0.15, cfg.baseline_wander_hz))
    amp = abs(float(rng.normal(0.0, cfg.baseline_wander_std)))
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def augment_window(
    window: np.ndarray,
    cfg: AugmentConfig,
    fs: int,
    rng: np.random.Generator,
    boost: float = 1.0,
) -> np.ndarray:
    """Geometric transforms applied to the full [T, C] window (input and target together)."""
    if not cfg.enabled:
        return window
    out = time_warp(window, cfg.time_warp_pct * boost, rng)
    lo, hi = cfg.amplitude_range
    span = (hi - lo) * boost / 2.0
    scale = float(rng.uniform(max(1.0 - span, 0.1), 1.0 + span))
    return (out * scale).astype(np.float32)


def corrupt_input(
    x: np.ndarray, cfg: AugmentConfig, fs: int, rng: np.random.Generator, boost: float = 1.0
) -> np.ndarray:
    """Noise and drift, applied to the model input only -- never to the target."""
    if not cfg.enabled:
        return x
    out = x + rng.normal(0.0, cfg.noise_std * boost, size=x.shape).astype(np.float32)
    return (out + baseline_wander(len(x), cfg, fs, rng)).astype(np.float32)

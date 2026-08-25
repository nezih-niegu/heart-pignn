"""Generate synthetic WFDB records, so the pipeline can be tested without data.

Not useful for metrics: the morphologies are toy Gaussians and there is no
realistic noise. Useful for the other thing that matters -- verifying the project
runs end to end before downloading 100 MB from PhysioNet, and letting the tests
run without data that cannot be versioned.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb

# Symbols chosen to cover all five AAMI classes
BEAT_SYMBOLS = ["N", "A", "V", "F", "/"]
SYMBOL_PROBS = [0.78, 0.07, 0.10, 0.03, 0.02]


def _second_lead(wave: np.ndarray) -> np.ndarray:
    """Toy second lead, so the regression head has a target.

    Not a real derivation: it is the first lead with inverted QRS polarity and a
    slightly shifted T wave -- just enough that predicting channel 1 from channel 0
    is not copying.
    """
    shifted = np.roll(wave, 6)
    return -0.65 * wave + 0.55 * shifted


def _beat_waveform(symbol: str, fs: int, rng: np.random.Generator) -> np.ndarray:
    """Toy beat: P, QRS and T as Gaussians, shaped by class."""
    length = int(0.8 * fs)
    t = np.linspace(0.0, 0.8, length)

    def g(center, width, amp):
        return amp * np.exp(-0.5 * ((t - center) / width) ** 2)

    if symbol == "V":  # wide QRS, no P, opposite T
        wave = g(0.40, 0.055, 2.2) + g(0.56, 0.09, -0.9)
    elif symbol == "A":  # early, premature P
        wave = g(0.24, 0.020, 0.35) + g(0.40, 0.016, 1.7) + g(0.55, 0.06, 0.35)
    elif symbol == "F":  # blend of normal and ventricular
        wave = g(0.26, 0.022, 0.15) + g(0.40, 0.035, 1.9) + g(0.55, 0.07, -0.2)
    elif symbol == "/":  # pacemaker spike
        wave = g(0.385, 0.004, 1.2) + g(0.41, 0.030, 1.4) + g(0.55, 0.07, -0.3)
    else:  # normal
        wave = g(0.26, 0.022, 0.25) + g(0.40, 0.014, 1.8) + g(0.55, 0.065, 0.4)

    return wave + rng.normal(0.0, 0.02, size=length)


def generate_record(
    out_dir: Path,
    name: str,
    fs: int = 360,
    n_beats: int = 220,
    hr_bpm: float = 75.0,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    rr_mean = 60.0 / hr_bpm

    symbols: list[str] = []
    positions: list[int] = []
    signal = np.zeros((int(fs * (n_beats + 2) * rr_mean * 1.3), 2), dtype=np.float64)

    cursor = int(0.5 * fs)
    for _ in range(n_beats):
        symbol = str(rng.choice(BEAT_SYMBOLS, p=SYMBOL_PROBS))
        rr = rr_mean * (1.0 + rng.normal(0.0, 0.06))
        if symbol in ("A", "V"):
            rr *= 0.72  # ectopic beats arrive early
        cursor += int(rr * fs)

        wave = _beat_waveform(symbol, fs, rng)
        r_offset = int(0.40 * fs)
        start = cursor - r_offset
        if start < 0 or start + len(wave) >= len(signal):
            break
        signal[start : start + len(wave), 0] += wave
        signal[start : start + len(wave), 1] += _second_lead(wave)
        positions.append(cursor)
        symbols.append(symbol)

    signal += rng.normal(0.0, 0.03, size=signal.shape)
    out_dir.mkdir(parents=True, exist_ok=True)

    wfdb.wrsamp(
        record_name=name,
        fs=fs,
        units=["mV", "mV"],
        sig_name=["MLII", "V1"],
        p_signal=signal,
        fmt=["16", "16"],
        adc_gain=[200.0, 200.0],
        baseline=[0, 0],
        write_dir=str(out_dir),
    )
    wfdb.wrann(
        record_name=name,
        extension="atr",
        sample=np.asarray(positions, dtype=np.int64),
        symbol=symbols,
        fs=fs,
        write_dir=str(out_dir),
    )


def generate_dataset(out_dir: str | Path, n_records: int = 12, seed: int = 0, **kwargs) -> Path:
    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    for i in range(n_records):
        generate_record(
            out_dir,
            name=f"{900 + i}",
            hr_bpm=float(rng.uniform(48, 115)),  # spans brady, normal and tachy
            seed=int(rng.integers(0, 10_000)),
            **kwargs,
        )
    print(f"Generated {n_records} synthetic records in {out_dir}")
    return out_dir

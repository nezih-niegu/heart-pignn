"""The heuristic decision tree from main.py, rewritten so it can attach to a network.

main.py decides with hard cuts: HR < 60 -> bradycardia, CV > 15% -> arrhythmia,
and so on. That works as a clinical report, but a hard cut has zero gradient
everywhere, so it cannot be fused with a learned latent space.

Here the same tree appears in two forms:

- `hard_rules()` reproduces main.py's original output (problems, severities,
  details) so the textual report stays identical.
- `soft_rule_features()` returns the same tree as graded memberships, replacing
  every `x > threshold` cut with `sigmoid((x - threshold) / tau)`. That continuous
  vector is what enters the network and modulates the attention latent.

Important note on leakage
-------------------------
main.py simulated SpO2 from the record's fraction of abnormal beats
(`abnormal_ratio`, computed from the .atr symbols). Feeding that to a beat
classifier would be direct label leakage: the "saturation" already contains the
answer. Hypoxemia is therefore disabled by default and only activates if a real
SpO2 channel is connected. MIT-BIH does not have one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RULE_LABEL_NAMES: list[str] = ["bradycardia", "tachycardia", "arrhythmia", "hypoxemia"]

RULE_FEATURE_NAMES: list[str] = [
    # Graded tree memberships (soft equivalents of main.py's if-statements)
    "m_bradycardia",
    "m_bradycardia_severe",
    "m_tachycardia",
    "m_tachycardia_severe",
    "m_cv_high",
    "m_sudden_changes",
    "m_sdnn_low",
    "m_ectopic",
    "arrhythmia_score",
    "m_hypoxemia",
    "spo2_available",
    # Continuous rhythm statistics
    "hr_norm",
    "rr_mean_norm",
    "cv_norm",
    "sdnn_norm",
    "ectopic_ratio",
    # Local RR context around the beat
    "rr_prev_ratio",
    "rr_next_ratio",
    "rr_asymmetry",
    "rr_local_norm",
]

N_RULE_FEATURES: int = len(RULE_FEATURE_NAMES)

REGIME_NAMES: list[str] = ["normal", "bradycardia", "tachycardia", "irregular"]
N_REGIMES: int = len(REGIME_NAMES)


@dataclass(frozen=True)
class RuleThresholds:
    """main.py's thresholds. Changing one changes both the hard and the soft tree."""

    brady_bpm: float = 60.0
    brady_moderate_bpm: float = 50.0
    brady_severe_bpm: float = 40.0
    tachy_bpm: float = 100.0
    tachy_moderate_bpm: float = 120.0
    tachy_severe_bpm: float = 150.0
    cv_pct: float = 15.0
    sudden_change_bpm: float = 20.0
    sudden_change_count: float = 3.0
    sdnn_ms: float = 50.0
    ectopic_ratio: float = 0.10
    ectopic_low: float = 0.8
    ectopic_high: float = 1.2
    arrhythmia_moderate_score: int = 3
    arrhythmia_severe_score: int = 4
    spo2_pct: float = 95.0
    spo2_moderate_pct: float = 90.0
    spo2_severe_pct: float = 85.0


@dataclass(frozen=True)
class SoftTemperatures:
    """Transition width of each cut. Smaller = closer to a hard threshold."""

    hr: float = 6.0
    cv: float = 3.0
    count: float = 1.0
    sdnn: float = 12.0
    ratio: float = 0.04
    spo2: float = 1.5


@dataclass
class RhythmStats:
    """Rhythm statistics derived from beat timing (RR) alone."""

    hr_mean: float = 0.0
    rr_mean: float = 0.0
    cv: float = 0.0
    sudden_changes: float = 0.0
    sdnn: float = 0.0
    ectopic_ratio: float = 0.0
    rr_prev: float = 0.0
    rr_next: float = 0.0
    rr_local_mean: float = 0.0
    n_intervals: int = 0
    spo2_mean: float | None = None
    thresholds: RuleThresholds = field(default_factory=RuleThresholds)

    @property
    def valid(self) -> bool:
        return self.n_intervals >= 3 and self.rr_mean > 0


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))


def rhythm_stats_from_rr(
    rr_ms: np.ndarray,
    rr_prev: float = 0.0,
    rr_next: float = 0.0,
    spo2_mean: float | None = None,
    thresholds: RuleThresholds | None = None,
) -> RhythmStats:
    """Compute the statistics the tree consumes, from RR intervals in ms.

    `rr_ms` is the context window around the beat (say the 10 intervals before and
    after). `rr_prev` and `rr_next` are the intervals immediately before and after
    that beat: a premature beat gives itself away right there, with a short
    preceding RR followed by a compensatory pause.
    """
    thr = thresholds or RuleThresholds()
    rr_ms = np.asarray(rr_ms, dtype=np.float64)
    rr_ms = rr_ms[np.isfinite(rr_ms) & (rr_ms > 0)]

    if rr_ms.size < 3:
        return RhythmStats(spo2_mean=spo2_mean, thresholds=thr)

    hr = 60000.0 / rr_ms
    rr_mean = float(rr_ms.mean())
    hr_mean = float(hr.mean())
    cv = float(hr.std() / (hr.mean() + 1e-8) * 100.0)
    sudden = float(np.sum(np.abs(np.diff(hr)) > thr.sudden_change_bpm))
    sdnn = float(rr_ms.std())
    ectopic = np.sum(
        (rr_ms < thr.ectopic_low * rr_mean) | (rr_ms > thr.ectopic_high * rr_mean)
    )
    ectopic_ratio = float(ectopic / rr_ms.size)

    return RhythmStats(
        hr_mean=hr_mean,
        rr_mean=rr_mean,
        cv=cv,
        sudden_changes=sudden,
        sdnn=sdnn,
        ectopic_ratio=ectopic_ratio,
        rr_prev=float(rr_prev),
        rr_next=float(rr_next),
        rr_local_mean=rr_mean,
        n_intervals=int(rr_ms.size),
        spo2_mean=spo2_mean,
        thresholds=thr,
    )


def hard_rules(stats: RhythmStats) -> dict:
    """Reproduce the output of main.py's `CardiacProblemClassifier.analyze_record`."""
    thr = stats.thresholds
    problems: list[str] = []
    severities: list[str] = []
    details: list[str] = []

    if not stats.valid:
        return {
            "problems": ["normal"],
            "severities": ["n/a"],
            "details": ["Too few beats to estimate rhythm"],
            "primary_problem": "normal",
            "max_severity": "n/a",
        }

    hr = stats.hr_mean
    if hr < thr.brady_bpm:
        problems.append("bradycardia")
        if hr < thr.brady_severe_bpm:
            severities.append("severe")
        elif hr < thr.brady_moderate_bpm:
            severities.append("moderate")
        else:
            severities.append("mild")
        details.append(f"HR={hr:.1f} bpm (low)")
    elif hr > thr.tachy_bpm:
        problems.append("tachycardia")
        if hr > thr.tachy_severe_bpm:
            severities.append("severe")
        elif hr > thr.tachy_moderate_bpm:
            severities.append("moderate")
        else:
            severities.append("mild")
        details.append(f"HR={hr:.1f} bpm (high)")

    score = 0
    arr_details: list[str] = []
    if stats.cv > thr.cv_pct:
        score += 1
        arr_details.append(f"CV={stats.cv:.1f}%")
    if stats.sudden_changes > thr.sudden_change_count:
        score += 1
        arr_details.append(f"{stats.sudden_changes:.0f} sudden changes")
    if stats.sdnn < thr.sdnn_ms:
        score += 1
        arr_details.append(f"SDNN={stats.sdnn:.1f}ms")
    if stats.ectopic_ratio > thr.ectopic_ratio:
        score += 1
        arr_details.append(f"{stats.ectopic_ratio * 100:.1f}% ectopic")

    if score >= thr.arrhythmia_moderate_score:
        problems.append("arrhythmia")
        severities.append("severe" if score >= thr.arrhythmia_severe_score else "moderate")
        details.append("Irregular rhythm (" + ", ".join(arr_details[:2]) + ")")
    elif score >= 1:
        problems.append("arrhythmia")
        severities.append("mild")
        details.append("Mild irregularity (" + (arr_details[0] if arr_details else "variability") + ")")

    if stats.spo2_mean is not None and stats.spo2_mean < thr.spo2_pct:
        problems.append("hypoxemia")
        if stats.spo2_mean < thr.spo2_severe_pct:
            severities.append("severe")
        elif stats.spo2_mean < thr.spo2_moderate_pct:
            severities.append("moderate")
        else:
            severities.append("mild")
        details.append(f"SpO2={stats.spo2_mean:.1f}%")

    if not problems:
        problems.append("normal")
        severities.append("n/a")
        details.append(f"HR={hr:.1f} bpm")

    order = {"n/a": 0, "mild": 1, "moderate": 2, "severe": 3}
    max_sev = ["n/a", "mild", "moderate", "severe"][max(order.get(s, 0) for s in severities)]

    return {
        "problems": problems,
        "severities": severities,
        "details": details,
        "primary_problem": problems[0],
        "max_severity": max_sev,
    }


def rule_labels(stats: RhythmStats, include_hypoxemia: bool = False) -> np.ndarray:
    """Multi-hot labels from the hard tree: [brady, tachy, arrhythmia] (+ optional hypoxemia).

    These labels supervise the model's auxiliary head. They are not ground-truth
    diagnosis: they are main.py's opinion, and that is precisely the point.
    Forcing the latent to reproduce them is what connects the two systems.
    """
    n = 4 if include_hypoxemia else 3
    out = np.zeros(n, dtype=np.float32)
    decision = hard_rules(stats)
    for name in decision["problems"]:
        if name in RULE_LABEL_NAMES[:n]:
            out[RULE_LABEL_NAMES.index(name)] = 1.0
    return out


def regime_index(stats: RhythmStats) -> int:
    """Rhythm regime index (0..3). Selects which batch prototype the model uses."""
    if not stats.valid:
        return 0
    thr = stats.thresholds
    if stats.hr_mean < thr.brady_bpm:
        return 1
    if stats.hr_mean > thr.tachy_bpm:
        return 2
    labels = rule_labels(stats)
    if labels[2] > 0.5:
        return 3
    return 0


def soft_rule_features(
    stats: RhythmStats,
    temperatures: SoftTemperatures | None = None,
    use_local_rr: bool = True,
) -> np.ndarray:
    """Continuous `N_RULE_FEATURES`-dim vector: main.py's tree made differentiable."""
    tau = temperatures or SoftTemperatures()
    thr = stats.thresholds

    if not stats.valid:
        return np.zeros(N_RULE_FEATURES, dtype=np.float32)

    hr = stats.hr_mean
    m_brady = _sigmoid((thr.brady_bpm - hr) / tau.hr)
    m_brady_sev = _sigmoid((thr.brady_severe_bpm - hr) / tau.hr)
    m_tachy = _sigmoid((hr - thr.tachy_bpm) / tau.hr)
    m_tachy_sev = _sigmoid((hr - thr.tachy_severe_bpm) / tau.hr)
    m_cv = _sigmoid((stats.cv - thr.cv_pct) / tau.cv)
    m_sudden = _sigmoid((stats.sudden_changes - thr.sudden_change_count) / tau.count)
    m_sdnn = _sigmoid((thr.sdnn_ms - stats.sdnn) / tau.sdnn)
    m_ect = _sigmoid((stats.ectopic_ratio - thr.ectopic_ratio) / tau.ratio)
    score = (m_cv + m_sudden + m_sdnn + m_ect) / 4.0

    if stats.spo2_mean is None:
        m_hypox, spo2_flag = 0.0, 0.0
    else:
        m_hypox = _sigmoid((thr.spo2_pct - stats.spo2_mean) / tau.spo2)
        spo2_flag = 1.0

    rr_ref = stats.rr_local_mean if stats.rr_local_mean > 0 else stats.rr_mean
    if use_local_rr and rr_ref > 0:
        rr_prev_ratio = stats.rr_prev / rr_ref if stats.rr_prev > 0 else 1.0
        rr_next_ratio = stats.rr_next / rr_ref if stats.rr_next > 0 else 1.0
        rr_asym = rr_prev_ratio - rr_next_ratio
        rr_local = rr_ref / 1000.0
    else:
        rr_prev_ratio = rr_next_ratio = rr_asym = rr_local = 0.0

    feats = [
        m_brady,
        m_brady_sev,
        m_tachy,
        m_tachy_sev,
        m_cv,
        m_sudden,
        m_sdnn,
        m_ect,
        score,
        m_hypox,
        spo2_flag,
        hr / 100.0,
        stats.rr_mean / 1000.0,
        stats.cv / 100.0,
        stats.sdnn / 100.0,
        stats.ectopic_ratio,
        rr_prev_ratio,
        rr_next_ratio,
        rr_asym,
        rr_local,
    ]
    return np.asarray(feats, dtype=np.float32)


class CardiacProblemClassifier:
    """Drop-in wrapper matching main.py's class of the same name.

    It exists so the old code keeps running unchanged while the logic lives in a
    single place.
    """

    def __init__(self, thresholds: RuleThresholds | None = None):
        self.thresholds = thresholds or RuleThresholds()

    def analyze_record(
        self,
        hr_mean: float,
        hr_series: np.ndarray,
        rr_intervals: np.ndarray,
        spo2_mean: float | None = None,
    ) -> dict:
        stats = rhythm_stats_from_rr(
            np.asarray(rr_intervals, dtype=np.float64),
            spo2_mean=spo2_mean,
            thresholds=self.thresholds,
        )
        # Honour the mean HR the caller passes in, which may come from a longer
        # window than the RR intervals provided.
        stats.hr_mean = float(hr_mean)
        return hard_rules(stats)

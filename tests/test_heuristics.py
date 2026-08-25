import numpy as np

from heart_pignn.heuristics import (
    N_RULE_FEATURES,
    RULE_FEATURE_NAMES,
    hard_rules,
    regime_index,
    rhythm_stats_from_rr,
    rule_labels,
    soft_rule_features,
)


def rr_for_bpm(bpm: float, n: int = 20, jitter: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = 60000.0 / bpm
    return base * (1.0 + rng.normal(0.0, jitter, size=n))


def test_bradycardia_matches_main_py_rule():
    stats = rhythm_stats_from_rr(rr_for_bpm(45))
    decision = hard_rules(stats)
    assert "bradycardia" in decision["problems"]
    assert regime_index(stats) == 1


def test_tachycardia_matches_main_py_rule():
    stats = rhythm_stats_from_rr(rr_for_bpm(130))
    assert "tachycardia" in hard_rules(stats)["problems"]
    assert regime_index(stats) == 2


def test_soft_tree_tracks_hard_tree_across_threshold():
    idx = RULE_FEATURE_NAMES.index("m_bradycardia")
    memberships = [
        soft_rule_features(rhythm_stats_from_rr(rr_for_bpm(bpm)))[idx]
        for bpm in (38, 50, 60, 72, 90)
    ]
    assert memberships[0] > 0.95      # frank bradycardia
    assert memberships[1] > 0.80      # moderate bradycardia
    assert abs(memberships[2] - 0.5) < 0.05  # the hard threshold lands at 0.5
    assert memberships[-1] < 0.05     # normal rhythm
    assert all(a > b for a, b in zip(memberships, memberships[1:], strict=False)), "must be monotonic"


def test_soft_features_are_finite_and_correct_length():
    for bpm in (35, 60, 100, 180):
        feats = soft_rule_features(rhythm_stats_from_rr(rr_for_bpm(bpm, jitter=0.15)))
        assert feats.shape == (N_RULE_FEATURES,)
        assert np.isfinite(feats).all()


def test_degenerate_input_returns_zero_vector():
    stats = rhythm_stats_from_rr(np.array([800.0]))
    assert not stats.valid
    assert np.allclose(soft_rule_features(stats), 0.0)
    assert hard_rules(stats)["primary_problem"] == "normal"


def test_hypoxemia_disabled_by_default():
    stats = rhythm_stats_from_rr(rr_for_bpm(75))
    assert rule_labels(stats).shape == (3,)
    assert rule_labels(stats, include_hypoxemia=True).shape == (4,)


def test_local_rr_ratio_flags_premature_beat():
    # Premature beat: short preceding RR followed by a compensatory pause
    feats = soft_rule_features(
        rhythm_stats_from_rr(rr_for_bpm(75), rr_prev=450.0, rr_next=1100.0)
    )
    prev = feats[RULE_FEATURE_NAMES.index("rr_prev_ratio")]
    nxt = feats[RULE_FEATURE_NAMES.index("rr_next_ratio")]
    assert prev < 0.7 < nxt

# tests/test_rescaling.py
"""
Lightweight tests for src/rescaling.py: the Q-selection boundary rule, the
fixed-mode regression guard (must reproduce pre-existing behavior exactly),
and an end-to-end check that a real demographic model's graph is rescaled
consistently through resolve_scaling_factor().

Run with:
    conda run -n snakemake-env python -m pytest tests/test_rescaling.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rescaling import (
    DEFAULT_Q_VALUES,
    DEFAULT_THRESHOLDS,
    min_population_size,
    resolve_scaling_factor,
    select_scaling_factor,
)
from src.demes_models import split_migration_growth_both_model


# ---------------------------------------------------------------------------
# select_scaling_factor: boundary behavior of the default rule
#   Q=50  if N_min < 30,000
#   Q=100 if 30,000 <= N_min < 60,000
#   Q=200 if N_min >= 60,000
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_min, expected_q",
    [
        (100.0, 50.0),          # far below first threshold
        (29_999.0, 50.0),       # just below first threshold
        (30_000.0, 100.0),      # exactly at first threshold -> next tier
        (30_001.0, 100.0),      # just above first threshold
        (59_999.0, 100.0),      # just below second threshold
        (60_000.0, 200.0),      # exactly at second threshold -> next tier
        (60_001.0, 200.0),      # just above second threshold
        (5_000_000.0, 200.0),   # far above last threshold
    ],
)
def test_default_rule_boundaries(n_min, expected_q):
    assert select_scaling_factor(n_min) == expected_q
    # explicit defaults must match the implicit ones
    assert select_scaling_factor(n_min, DEFAULT_THRESHOLDS, DEFAULT_Q_VALUES) == expected_q


def test_select_scaling_factor_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        select_scaling_factor(50_000.0, thresholds=[30_000.0], q_values=[50.0, 100.0, 200.0])


def test_select_scaling_factor_custom_rule_is_honored():
    # proves the rule is genuinely configurable, not hardcoded
    thresholds = [1_000.0]
    q_values = [10.0, 20.0]
    assert select_scaling_factor(500.0, thresholds, q_values) == 10.0
    assert select_scaling_factor(1_000.0, thresholds, q_values) == 20.0
    assert select_scaling_factor(50_000.0, thresholds, q_values) == 20.0


# ---------------------------------------------------------------------------
# min_population_size: exercised against a real demes.Graph
# ---------------------------------------------------------------------------

SMALL_N_PARAMS = dict(
    N_ANC=15_000, N_CO0=15_000, N_CO1=20_000, N_FR0=12_000, N_FR1=18_000,
    T=8_000, m_CO_FR=5e-5, m_FR_CO=5e-5,
)
LARGE_N_PARAMS = dict(
    N_ANC=180_000, N_CO0=180_000, N_CO1=240_000, N_FR0=144_000, N_FR1=216_000,
    T=8_000, m_CO_FR=5e-5, m_FR_CO=5e-5,
)


def test_min_population_size_matches_smallest_epoch_size():
    g = split_migration_growth_both_model(SMALL_N_PARAMS)
    # N_FR0 (12,000) is the smallest of the five population sizes above
    assert min_population_size(g) == pytest.approx(12_000.0)


def test_min_population_size_raises_on_graph_with_no_epochs():
    class _EmptyDeme:
        epochs: list = []

    class _EmptyGraph:
        demes = [_EmptyDeme()]

    with pytest.raises(ValueError):
        min_population_size(_EmptyGraph())


# ---------------------------------------------------------------------------
# resolve_scaling_factor: fixed mode (regression guard) vs conditional mode
# ---------------------------------------------------------------------------

def test_fixed_mode_reproduces_legacy_behavior():
    g = split_migration_growth_both_model(SMALL_N_PARAMS)
    sel_cfg = {"slim_scaling": 10.0}  # rescaling_mode absent -> defaults to "fixed"
    resolved = resolve_scaling_factor(g, sel_cfg)
    assert resolved["slim_scaling"] == 10.0
    assert resolved["rescaling_mode"] == "fixed"
    # N_min is still computed and reported even though it doesn't drive Q here
    assert resolved["n_min"] == pytest.approx(12_000.0)


def test_fixed_mode_falls_back_to_default_q_when_unset():
    g = split_migration_growth_both_model(SMALL_N_PARAMS)
    resolved = resolve_scaling_factor(g, {})
    assert resolved["slim_scaling"] == 10.0  # matches sel.get("slim_scaling", 10.0) default


def test_conditional_mode_picks_q_from_n_min_small_draw():
    g = split_migration_growth_both_model(SMALL_N_PARAMS)  # N_min = 12,000
    sel_cfg = {"rescaling_mode": "conditional"}
    resolved = resolve_scaling_factor(g, sel_cfg)
    assert resolved["n_min"] == pytest.approx(12_000.0)
    assert resolved["slim_scaling"] == 50.0
    assert resolved["rescaling_mode"] == "conditional"


def test_conditional_mode_picks_q_from_n_min_large_draw():
    g = split_migration_growth_both_model(LARGE_N_PARAMS)  # N_min = 144,000 (N_FR0)
    sel_cfg = {"rescaling_mode": "conditional"}
    resolved = resolve_scaling_factor(g, sel_cfg)
    assert resolved["n_min"] == pytest.approx(144_000.0)
    assert resolved["slim_scaling"] == 200.0


def test_conditional_mode_honors_custom_rescaling_rule():
    g = split_migration_growth_both_model(LARGE_N_PARAMS)  # N_min = 144,000
    sel_cfg = {
        "rescaling_mode": "conditional",
        "rescaling_rule": {"thresholds": [100_000.0], "q_values": [75.0, 150.0]},
    }
    resolved = resolve_scaling_factor(g, sel_cfg)
    assert resolved["slim_scaling"] == 150.0


def test_unknown_rescaling_mode_raises():
    g = split_migration_growth_both_model(SMALL_N_PARAMS)
    with pytest.raises(ValueError):
        resolve_scaling_factor(g, {"rescaling_mode": "bogus"})


def test_q_is_deterministic_and_consistent_across_repeated_resolution():
    """
    Simulates what happens across a base sim + its window replicates: the
    same (model_type, sampled_params) draw gets its graph rebuilt and Q
    re-resolved independently several times. Since resolve_scaling_factor is
    a pure function of the graph, every one of those calls must agree --
    this is what "Q selected once per draw, fixed throughout" reduces to.
    """
    sel_cfg = {"rescaling_mode": "conditional"}
    qs = set()
    for _ in range(5):
        g = split_migration_growth_both_model(LARGE_N_PARAMS)  # rebuilt fresh each time
        qs.add(resolve_scaling_factor(g, sel_cfg)["slim_scaling"])
    assert qs == {200.0}

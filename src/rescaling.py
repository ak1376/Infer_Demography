# src/rescaling.py
"""
Conditional SLiM rescaling: choose the scaling factor Q from the smallest
unscaled population size (N_min) across every demographic epoch, instead of
one fixed Q for every parameter draw.

Rationale: rescaling bias in forward-time simulation is governed by how far
Ne = N/Q has been pushed down for the smallest population in the model, not
by Q alone -- a fixed Q that is safe for a large-N draw can be far too
aggressive for a small-N draw from the same prior. See resolve_scaling_factor()
for the single entry point src/simulation.py uses; it defaults to "fixed"
mode, which reproduces today's behavior (Q read verbatim from
selection.slim_scaling) exactly.
"""
from __future__ import annotations

from typing import Any, Dict, List

import demes

# Default rule: Q=50 below 30k, Q=100 in [30k, 60k), Q=200 at/above 60k.
DEFAULT_THRESHOLDS: List[float] = [30_000.0, 60_000.0]
DEFAULT_Q_VALUES: List[float] = [50.0, 100.0, 200.0]


def min_population_size(g: demes.Graph) -> float:
    """
    Smallest start_size/end_size across every epoch of every deme in g.

    This is the unscaled Ne floor of the demographic model: whichever
    epoch/deme it comes from is the one most exposed to rescaling artifacts,
    since Ne = N/Q shrinks that floor the most.
    """
    sizes = [
        size
        for deme in g.demes
        for epoch in deme.epochs
        for size in (epoch.start_size, epoch.end_size)
    ]
    if not sizes:
        raise ValueError("Graph has no epochs to compute N_min from.")
    return float(min(sizes))


def select_scaling_factor(
    n_min: float,
    thresholds: List[float] = DEFAULT_THRESHOLDS,
    q_values: List[float] = DEFAULT_Q_VALUES,
) -> float:
    """
    Step-function rule: q_values[i] applies for
    thresholds[i-1] <= n_min < thresholds[i], with q_values[0] below the
    first threshold and q_values[-1] at/above the last one.

    len(q_values) must be len(thresholds) + 1.
    """
    if len(q_values) != len(thresholds) + 1:
        raise ValueError(
            "q_values must have exactly one more entry than thresholds "
            f"(got {len(q_values)} q_values, {len(thresholds)} thresholds)"
        )
    for threshold, q in zip(thresholds, q_values):
        if n_min < threshold:
            return float(q)
    return float(q_values[-1])


def resolve_scaling_factor(g: demes.Graph, sel_cfg: Dict[str, Any]) -> Dict[str, float]:
    """
    Single entry point used by src/simulation.py. Decides Q per
    sel_cfg["rescaling_mode"]:
      - "fixed" (default, unchanged behavior): Q = sel_cfg["slim_scaling"]
        (falls back to 10.0, exactly like the pre-existing direct read did).
      - "conditional": Q is picked from N_min via select_scaling_factor(),
        using sel_cfg["rescaling_rule"] = {"thresholds": [...], "q_values": [...]}
        if present, else DEFAULT_THRESHOLDS/DEFAULT_Q_VALUES.

    N_min is always computed and returned (cheap, and worth recording even
    in fixed mode), but only determines Q in "conditional" mode.

    Returns {"slim_scaling": Q, "n_min": N_min, "rescaling_mode": mode}.
    """
    mode = sel_cfg.get("rescaling_mode", "fixed")
    n_min = min_population_size(g)

    if mode == "fixed":
        q = float(sel_cfg.get("slim_scaling", 10.0))
    elif mode == "conditional":
        rule = sel_cfg.get("rescaling_rule") or {}
        thresholds = [float(x) for x in rule.get("thresholds", DEFAULT_THRESHOLDS)]
        q_values = [float(x) for x in rule.get("q_values", DEFAULT_Q_VALUES)]
        q = select_scaling_factor(n_min, thresholds, q_values)
    else:
        raise ValueError(
            f"Unknown selection.rescaling_mode: {mode!r} (expected 'fixed' or 'conditional')"
        )

    return dict(slim_scaling=q, n_min=n_min, rescaling_mode=mode)

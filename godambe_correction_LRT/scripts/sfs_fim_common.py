#!/usr/bin/env python3
# godambe_correction_LRT/scripts/sfs_fim_common.py
"""Shared helpers for the per-start / collect-and-fim SFS identifiability jobs."""

from __future__ import annotations

import numpy as np

ILL_COND_THRESHOLD = 1e8
RAIL_FRAC = 0.01


def make_safe_model_func(model_func):
    """Clip absolute params to a physically valid domain before building the
    demes graph, so finite-difference probes near a railed prior bound don't
    crash (migration must be in [0, 1]; sizes/times must be > 0)."""

    def safe_model_func(p_abs):
        p_abs = dict(p_abs)
        for k, v in p_abs.items():
            if k.startswith("m_"):
                p_abs[k] = float(np.clip(v, 1e-12, 1 - 1e-9))
            elif k.startswith("N_") or k == "T":
                p_abs[k] = max(float(v), 1e-12)
        return model_func(p_abs)

    return safe_model_func

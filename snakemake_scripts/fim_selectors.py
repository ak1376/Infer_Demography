#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional


def pick_best_params_from_blob(tool_blob: dict) -> Optional[dict]:
    """
    Accepts one sub-blob like data['dadi'] or data['moments'] or data['momentsLD'].
    Returns the param dict for the *highest likelihood* entry, or None.
    Supports:
      - {'best_params': [dict, ...], 'best_ll': [float, ...]}  # multi-restart, top-K
      - {'best_params': dict, 'best_ll': float}                # single-shot run
    """
    if not tool_blob:
        return None

    bplist = tool_blob.get("best_params")

    # multi-restart format (list of dicts + list of lls)
    if isinstance(bplist, list) and bplist:
        blls = tool_blob.get("best_ll")
        if isinstance(blls, list) and len(blls) == len(bplist):
            i = int(np.nanargmax(np.asarray(blls, dtype=float)))
            return dict(bplist[i])
        return dict(bplist[0])  # fallback if no LLs provided

    # single-shot format (one dict, one scalar ll)
    if isinstance(bplist, dict):
        return dict(bplist)

    return None


def best_theta_for_engine(
    all_inf: dict, engine: str, param_order: List[str]
) -> Optional[List[float]]:
    """
    engine: 'dadi' or 'moments' (or 'momentsLD')
    Returns theta (list of floats) in the given param_order, or None.
    """
    key = engine
    blob = all_inf.get(key)
    pmap = pick_best_params_from_blob(blob)
    if not pmap:
        return None
    return [float(pmap.get(name, np.nan)) for name in param_order]

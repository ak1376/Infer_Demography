#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional


def pick_best_params_from_blob(tool_blob: dict) -> Optional[dict]:
    """
    Accepts one sub-blob like data['dadi'] or data['moments'] or data['momentsLD'].
    Returns the param dict for the *highest likelihood* entry, or None.
    Supports:
      - {'best_params': [dict, ...], 'best_ll': [float, ...]}  # dadi/moments
      - {'best_params': dict, 'best_lls': float}               # momentsLD
    """
    if not tool_blob:
        return None

    # momentsLD format (single dict + scalar ll)
    if isinstance(tool_blob.get("best_params"), dict) and "best_lls" in tool_blob:
        return dict(tool_blob["best_params"])

    # dadi/moments format (lists of dicts + list of lls)
    bplist = tool_blob.get("best_params")
    blls = tool_blob.get("best_ll")
    if isinstance(bplist, list) and bplist:
        if isinstance(blls, list) and len(blls) == len(bplist):
            i = int(np.nanargmax(np.asarray(blls, dtype=float)))
            return dict(bplist[i])
        return dict(bplist[0])  # fallback if no LLs provided

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

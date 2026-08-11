"""Shared FIM / SFS-residual payload assembly for combine_results(_real).

Both combine_results and combine_results_real attach the same two optional
payloads to all_inferences.pkl: flattened upper-triangular FIMs per engine,
and SFS-residual vectors (raw or Gram-Schmidt-reduced) per engine. This
module factors that assembly out so it isn't duplicated between the
simulated- and real-data rules.
"""
import json
import os
import re

import numpy as np


def build_fim_payload(fim_paths):
    payload = {}
    for fim_path in fim_paths:
        eng = re.sub(r".*?/fim/([^.]+)\.fim\.npy$", r"\1", fim_path)
        F = np.load(fim_path)
        iu = np.triu_indices(F.shape[0])
        payload[eng] = {
            "shape": [int(F.shape[0]), int(F.shape[1])],
            "tri_flat": F[iu].astype(float).tolist(),
            "indices": "upper_including_diagonal",
            "order": "row-major",
        }
    return payload


def build_residual_payload(resid_vec_paths):
    payload = {}
    for vec_path in resid_vec_paths:
        m = re.search(r"/sfs_residuals/([^/]+)/([^/]+)\.npy$", vec_path)
        if not m:
            continue
        eng = m.group(1)
        stem = m.group(2)  # residuals_flat OR residuals_gs_coeffs
        base = os.path.dirname(vec_path)

        vec = np.load(vec_path)

        # full residual array shape (optional)
        arr_path = os.path.join(base, "residuals.npy")
        arr = np.load(arr_path) if os.path.exists(arr_path) else None

        entry = {
            "vector": vec.astype(float).tolist(),
            "vector_dim": int(vec.size),
            "vector_type": (
                "gram_schmidt_coeffs" if stem == "residuals_gs_coeffs" else "raw_flat_residuals"
            ),
            "full_residual_shape": (list(arr.shape) if arr is not None else None),
            "order": "row-major",
        }

        # If GS: attach GS metadata/basis shapes when available
        if stem == "residuals_gs_coeffs":
            meta_path = os.path.join(base, "meta.json")
            basis_path = os.path.join(base, "residuals_gs_basis.npy")
            if os.path.exists(basis_path):
                Q = np.load(basis_path)
                entry["gs_basis_shape"] = [int(Q.shape[0]), int(Q.shape[1])]
            if os.path.exists(meta_path):
                try:
                    mj = json.loads(open(meta_path, "r").read())
                    entry["gram_schmidt_k"] = mj.get("gram_schmidt_k", None)
                    entry["gram_schmidt_k_effective"] = mj.get("gram_schmidt_k_effective", None)
                    entry["gram_schmidt_basis"] = mj.get("gram_schmidt_basis", None)
                    entry["gram_schmidt_eps"] = mj.get("gram_schmidt_eps", None)
                except Exception:
                    pass

        payload[eng] = entry
    return payload

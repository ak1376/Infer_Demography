#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib, json, pickle
from pathlib import Path
import numpy as np

from fim_selectors import best_theta_for_engine

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sfs-pkl", required=True, type=Path)
    ap.add_argument("--config",  required=True, type=Path)
    ap.add_argument("--model",   required=True, type=str, help='pkg.module:function_name')
    ap.add_argument("--all-inferences", required=True, type=Path)
    ap.add_argument("--engine", choices=["dadi","moments"], required=True)
    ap.add_argument("--param-names", required=True, type=str,
                    help="Comma-separated parameter names (canonical order, typically cfg['priors'].keys())")
    ap.add_argument("--out-prefix", required=True, type=Path)
    ap.add_argument("--pts", type=str, default="auto")   # dadi: "n1,n2,n3" or "auto"
    ap.add_argument("--rel-step", type=float, default=1e-4)
    args = ap.parse_args()

    # Load
    sfs = pickle.loads(args.sfs_pkl.read_bytes())
    cfg = json.loads(args.config.read_text())
    priors = cfg["priors"]
    # Param order is explicitly provided (keeps you flexible)
    param_order = [x.strip() for x in args.param-names.split(",") if x.strip()]

    all_inf = pickle.loads(args.all_inferences.read_bytes())
    theta = best_theta_for_engine(all_inf, engine=args.engine, param_order=param_order)
    if theta is None or any([not np.isfinite(v) for v in theta]):
        raise SystemExit(f"No valid {args.engine} parameter vector found (or contains NaNs).")

    # Build CLI to fisher_info_sfs.py
    fisher = Path(__file__).with_name("fisher_info_sfs.py")
    params_str = ",".join(str(float(v)) for v in theta)
    pname_str  = ",".join(param_order)

    # Model is passed straight through
    pts_arg = []
    if args.engine == "dadi" and args.pts and args.pts.lower() != "auto":
        pts_arg = ["--pts", args.pts]  # else 'auto' inside fisher_info_sfs

    import subprocess, sys
    cmd = [
        sys.executable, str(fisher),
        "--sfs-pkl", str(args.sfs_pkl),
        "--config",  str(args.config),
        "--model",   args.model,
        "--param-names", pname_str,
        "--params",      params_str,
        "--engine", args.engine,
        "--rel-step", str(args.rel_step),
        "--out-prefix", str(args.out_prefix),
    ] + pts_arg
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()

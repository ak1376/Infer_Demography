#!/usr/bin/env python3
# godambe_correction_LRT/snakemake_scripts/diagonalize_varcov.py

"""
Zero the off-diagonal elements of every per-r-bin (and H) covariance block in
a means.varcovs.pkl, keeping only the per-statistic variances.

This is a deliberate simplification, not a numerical-stability patch: the
off-diagonal terms in these blocks are largely REAL structural correlation
(e.g. DD_0_0/DD_0_1/DD_1_1 come from the same underlying pairs), not just
estimation noise, so dropping them treats correlated statistics as
independent evidence. That will bias the fit toward directions where several
correlated statistics agree, and will make any resulting standard errors
too small (the composite likelihood believes it has more independent
information than it actually does). Chosen anyway as a standing choice for
this analysis; the un-modified means.varcovs.pkl is kept alongside so the
comparison is always available.

The exactly-zero-variance "normalizing" statistic per block (pi2_0_0_0_0 for
LD stats, H_0_0 for heterozygosity -- forced to a constant by moments.LD's
normalization convention) is left untouched either way; it gets stripped out
by moments.LD.Inference.remove_normalized_data() inside the fitting code
before any inversion happens, regardless of what's in that row/column here.

Usage:
  python godambe_correction_LRT/snakemake_scripts/diagonalize_varcov.py \
      --mv godambe_correction_LRT/real_arms/Chr3L/momentsld/means.varcovs.pkl \
      --out godambe_correction_LRT/real_arms/Chr3L/momentsld/means.varcovs.diagonly.pkl
"""

import argparse
import copy
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with open(args.mv, "rb") as f:
        mv = pickle.load(f)

    mv_diag = copy.deepcopy(mv)
    mv_diag["varcovs"] = [np.diag(np.diag(np.asarray(vc))) for vc in mv["varcovs"]]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(mv_diag, f)
    print(f"wrote {args.out} ({len(mv_diag['varcovs'])} diagonal-only blocks)")


if __name__ == "__main__":
    main()

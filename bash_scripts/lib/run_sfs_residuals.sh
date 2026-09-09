#!/usr/bin/env bash
# Shared body for the Snakefile's sfs_residuals / sfs_residuals_real rules.
# The two rules differ only in which paths they pass in, so this script
# takes those as positional args and does the run + output-verification /
# Gram-Schmidt-sentinel logic that used to be duplicated in both rules.
#
# Usage: run_sfs_residuals.sh <engine> <cfg> <model_py> <obs_sfs> <inf_dir> \
#            <out_dir> <use_gs> <resid_script> <basedir> [n_bins]
set -euo pipefail

engine="$1"
cfg="$2"
model_py="$3"
obs_sfs="$4"
inf_dir="$5"
out_dir="$6"
use_gs="$7"
resid_script="$8"
basedir="$9"
n_bins="${10:-}"

mkdir -p "$out_dir"

N_BINS_ARG=""
if [ -n "$n_bins" ]; then
    N_BINS_ARG="--n-bins $n_bins"
fi

PYTHONPATH="$basedir" \
python "$resid_script" \
  --mode "$engine" \
  --config "$cfg" \
  --model-py "$model_py" \
  --observed-sfs "$obs_sfs" \
  --inference-dir "$inf_dir" \
  --outdir "$out_dir" \
  $N_BINS_ARG

# Base outputs must exist
test -f "$out_dir/residuals.npy"
test -f "$out_dir/residuals_flat.npy"
test -f "$out_dir/meta.json"
test -f "$out_dir/residuals_histogram.png"

# GS outputs: if enabled, require real artifacts; else create sentinels
if [ "$use_gs" = "True" ]; then
    test -f "$out_dir/residuals_gs_coeffs.npy"
    test -f "$out_dir/residuals_gs_basis.npy"
else
    touch "$out_dir/.gs_disabled" "$out_dir/.gs_basis_disabled"
fi

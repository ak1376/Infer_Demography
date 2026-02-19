#!/usr/bin/env python3
"""
Split a large VCF into overlapping windows for LD analysis.
Generates:
  - window_<idx>.vcf.gz
  - samples.txt
  - flat_map.txt
for each window.
"""

import argparse
import subprocess
from pathlib import Path


def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)


def get_vcf_bounds(vcf_path):
    """Get the start and end positions from the VCF using bcftools."""
    # Get first position
    cmd_first = f"bcftools query -f '%POS\n' '{vcf_path}' | head -n 1"
    first_pos = int(subprocess.check_output(cmd_first, shell=True).strip())

    # Get last position
    cmd_last = f"bcftools query -f '%POS\n' '{vcf_path}' | tail -n 1"
    last_pos = int(subprocess.check_output(cmd_last, shell=True).strip())

    return first_pos, last_pos


def get_chrom_name(vcf_path):
    """Get the chromosome name."""
    cmd = f"bcftools query -f '%CHROM\n' '{vcf_path}' | head -n 1"
    return subprocess.check_output(cmd, shell=True).strip().decode()


def main():
    p = argparse.ArgumentParser(description="Split VCF into overlapping windows")
    p.add_argument(
        "--input-vcf", required=True, type=Path, help="Input VCF file (bgzipped)"
    )
    p.add_argument(
        "--popfile", required=True, type=Path, help="Population file (sampleID popID)"
    )
    p.add_argument(
        "--out-dir", required=True, type=Path, help="Output directory for windows"
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=1000000,
        help="Window size in bp (default 1MB)",
    )
    p.add_argument(
        "--num-windows", type=int, default=100, help="Number of windows to generate"
    )
    p.add_argument(
        "--recomb-rate", type=float, default=1e-8, help="Recombination rate per bp"
    )
    p.add_argument(
        "--window-index",
        type=int,
        default=None,
        help="If provided, only generate this window (0-based index)",
    )

    args = p.parse_args()

    if not args.input_vcf.exists():
        raise FileNotFoundError(f"Input VCF not found: {args.input_vcf}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure VCF is indexed
    if (
        not Path(str(args.input_vcf) + ".tbi").exists()
        and not Path(str(args.input_vcf) + ".csi").exists()
    ):
        print("Indexing VCF...")
        run_command(f"bcftools index -t '{args.input_vcf}'")

    # Get VCF info
    print("Getting VCF bounds...")
    start_pos, end_pos = get_vcf_bounds(args.input_vcf)
    chrom = get_chrom_name(args.input_vcf)
    print(
        f"VCF covers {chrom}:{start_pos}-{end_pos} (Length: {end_pos - start_pos + 1} bp)"
    )

    # Calculate window positions
    total_span = end_pos - start_pos
    if total_span < args.window_size:
        print(
            f"Warning: VCF span ({total_span}) is smaller than window size ({args.window_size}). Creating single window."
        )
        step_size = 0
        actual_num_windows = 1
    else:
        if args.num_windows > 1:
            step_size = (total_span - args.window_size) / (args.num_windows - 1)
        else:
            step_size = 0
        actual_num_windows = args.num_windows

    # Decide which windows to generate
    if args.window_index is not None:
        if args.window_index < 0 or args.window_index >= actual_num_windows:
            raise ValueError(
                f"--window-index {args.window_index} is out of range [0, {actual_num_windows - 1}]"
            )
        window_indices = [args.window_index]
    else:
        window_indices = range(actual_num_windows)

    # Prepare samples.txt (common for all windows)
    samples_out = args.out_dir / "samples.txt"
    if not samples_out.exists():
        with open(args.popfile) as fin, open(samples_out, "w") as fout:
            fout.write("sample\tpop\n")
            for line in fin:
                parts = line.strip().split()
                if len(parts) >= 2:
                    fout.write(f"{parts[0]}\t{parts[1]}\n")
        print(f"Created {samples_out}")
    else:
        print(f"{samples_out} already exists, skipping creation")

    # Generate the requested window(s)
    for i in window_indices:
        w_start = int(start_pos + i * step_size)
        w_end = w_start + args.window_size - 1
        if w_end > end_pos:
            w_end = end_pos

        print(f"Generating window {i}: {chrom}:{w_start}-{w_end}")

        win_vcf = args.out_dir / f"window_{i}.vcf.gz"
        cmd = f"bcftools view -r {chrom}:{w_start}-{w_end} -O z -o '{win_vcf}' '{args.input_vcf}'"
        run_command(cmd)
        run_command(f"bcftools index -t '{win_vcf}'")

    # Generate the single flat_map.txt covering the whole VCF range
    map_out = args.out_dir / "flat_map.txt"
    if not map_out.exists():
        with open(map_out, "w") as f:
            f.write("pos\tMap(cM)\n")
            # Start point
            f.write(f"{start_pos}\t0\n")
            # End point
            total_cm = (end_pos - start_pos) * args.recomb_rate * 100
            f.write(f"{end_pos}\t{total_cm}\n")
        print(f"Created global map: {map_out}")
    else:
        print(f"{map_out} already exists, skipping creation")


if __name__ == "__main__":
    main()

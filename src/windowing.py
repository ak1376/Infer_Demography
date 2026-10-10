"""
src/windowing.py

Chop an existing .trees or .vcf(.gz) file into windows for LD analysis.

Windows are laid out the same way as the existing
snakemake_scripts/split_vcf_windows.py convention: `num_windows`
equally-spaced windows of `window_size` bp across the input's total span,

    step = (total_span - window_size) / (num_windows - 1)      [num_windows > 1]

so windows overlap whenever num_windows * window_size exceeds the span.

  .vcf(.gz) input -> window_<i>.vcf.gz, sliced from the input with
                      `bcftools view -r` (positions stay absolute, i.e.
                      relative to the *input* VCF's own coordinates).

  .trees input    -> window_<i>.trees, sliced at the tree-sequence level
                      with `ts.keep_intervals([[start, end]]).ltrim().rtrim()`
                      (coordinates re-base to start at 0, matching
                      src.simulation.simulate_one_window_replicate). The
                      window's VCF is written directly from that trimmed
                      tree sequence, so window_<i>.trees and
                      window_<i>.vcf.gz always agree exactly.

Both paths also write one samples.txt + flat_map.txt per output directory
(shared across all windows written into it), in the same format consumed
by src.LD_stats.compute_ld_window.
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import tskit

PathLike = Union[str, Path]


def _run(cmd: str) -> None:
    subprocess.check_call(cmd, shell=True)


def compute_window_bounds(
    start: float, end: float, window_size: int, num_windows: int
) -> List[Tuple[int, int]]:
    """
    Equally-spaced (possibly-overlapping) [start, end] window bounds.

    If the span is smaller than window_size, falls back to a single window
    covering the full span (matching the previous
    snakemake_scripts/split_vcf_windows.py behavior) instead of raising.
    """
    total_span = end - start
    if total_span < window_size:
        return [(int(start), int(end))]
    step = (total_span - window_size) / (num_windows - 1) if num_windows > 1 else 0.0
    bounds = []
    for i in range(num_windows):
        w_start = int(start + i * step)
        w_end = min(w_start + window_size - 1, int(end))
        bounds.append((w_start, w_end))
    return bounds


def _resolve_indices(
    bounds: List[Tuple[int, int]], num_windows: int, window_index: Optional[int]
) -> List[int]:
    actual_num_windows = len(bounds)
    if actual_num_windows < num_windows:
        print(
            f"[windowing] span too small for {num_windows} windows; "
            f"using a single full-span window instead."
        )
    if window_index is not None:
        if window_index < 0 or window_index >= actual_num_windows:
            raise ValueError(
                f"--window-index {window_index} is out of range "
                f"[0, {actual_num_windows - 1}]"
            )
        return [window_index]
    return list(range(actual_num_windows))


def _write_flat_map(out_dir: Path, start: float, end: float, recomb_rate: float) -> Path:
    map_out = out_dir / "flat_map.txt"
    total_cm = (end - start) * recomb_rate * 100.0
    map_out.write_text(f"pos\tMap(cM)\n{start:.0f}\t0\n{end:.0f}\t{total_cm}\n")
    return map_out


def _write_samples_from_popfile(popfile: Path, out_dir: Path) -> Path:
    samples_out = out_dir / "samples.txt"
    with open(popfile) as fin, open(samples_out, "w") as fout:
        fout.write("sample\tpop\n")
        for line in fin:
            parts = line.strip().split()
            if len(parts) >= 2:
                fout.write(f"{parts[0]}\t{parts[1]}\n")
    return samples_out


def _write_samples_from_ts(ts: tskit.TreeSequence, out_dir: Path) -> Path:
    """
    samples.txt matching ts.write_vcf()'s default VCF-column layout:
      - if sample nodes are grouped into individuals in the data model,
        one row per such individual (increasing individual-ID order) —
        write_vcf combines each individual's sample nodes into one
        multiploid column regardless of ploidy.
      - otherwise (no individuals table), one row per sample node
        (write_vcf's ploidy=1 default).
    Either way rows are tsk_0, tsk_1, ... in the same order as the VCF
    columns write_vcf produces.
    """
    pop_names: Dict[int, str] = {}
    for pid in range(ts.num_populations):
        pop = ts.population(pid)
        name = None
        if hasattr(pop, "name") and getattr(pop, "name", None):
            name = pop.name
        elif hasattr(pop, "metadata") and isinstance(pop.metadata, dict):
            name = pop.metadata.get("name")
        pop_names[pid] = name or f"pop_{pid}"

    sample_set = {int(u) for u in ts.samples()}
    individual_rows = [
        ind for ind in ts.individuals() if any(int(n) in sample_set for n in ind.nodes)
    ]

    lines = ["sample\tpop"]
    if individual_rows:
        for i, ind in enumerate(individual_rows):
            node_id = next(int(n) for n in ind.nodes if int(n) in sample_set)
            pid = ts.node(node_id).population
            lines.append(f"tsk_{i}\t{pop_names[pid]}")
    else:
        for i, node_id in enumerate(ts.samples()):
            pid = ts.node(int(node_id)).population
            lines.append(f"tsk_{i}\t{pop_names[pid]}")

    samples_out = out_dir / "samples.txt"
    samples_out.write_text("\n".join(lines) + "\n")
    return samples_out


def _vcf_bounds(vcf_gz: Path) -> Tuple[str, int, int]:
    chrom = (
        subprocess.check_output(
            f"bcftools query -f '%CHROM\n' '{vcf_gz}' | head -n 1", shell=True
        )
        .decode()
        .strip()
    )
    first_pos = int(
        subprocess.check_output(
            f"bcftools query -f '%POS\n' '{vcf_gz}' | head -n 1", shell=True
        )
    )
    last_pos = int(
        subprocess.check_output(
            f"bcftools query -f '%POS\n' '{vcf_gz}' | tail -n 1", shell=True
        )
    )
    return chrom, first_pos, last_pos


def window_vcf(
    input_vcf: PathLike,
    out_dir: PathLike,
    *,
    window_size: int,
    num_windows: int,
    popfile: Optional[PathLike] = None,
    recomb_rate: float = 1e-8,
    window_index: Optional[int] = None,
) -> List[Path]:
    """
    Chop an existing VCF into `num_windows` equally-spaced, possibly-
    overlapping windows of `window_size` bp via `bcftools view -r`.

    `popfile` (sampleID popID, whitespace-separated) is copied into
    samples.txt if given; otherwise samples.txt is not written.

    Returns the window_<i>.vcf.gz path(s) written.
    """
    input_vcf = Path(input_vcf)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tbi = Path(str(input_vcf) + ".tbi")
    csi = Path(str(input_vcf) + ".csi")
    if not tbi.exists() and not csi.exists():
        _run(f"bcftools index -t '{input_vcf}'")

    chrom, first_pos, last_pos = _vcf_bounds(input_vcf)
    bounds = compute_window_bounds(first_pos, last_pos, window_size, num_windows)
    indices = _resolve_indices(bounds, num_windows, window_index)

    written: List[Path] = []
    for i in indices:
        w_start, w_end = bounds[i]
        win_vcf = out_dir / f"window_{i}.vcf.gz"
        _run(
            f"bcftools view -r {chrom}:{w_start}-{w_end} -O z -o '{win_vcf}' '{input_vcf}'"
        )
        _run(f"bcftools index -t '{win_vcf}'")
        written.append(win_vcf)

    if popfile is not None:
        _write_samples_from_popfile(Path(popfile), out_dir)
    _write_flat_map(out_dir, first_pos, last_pos, recomb_rate)
    return written


def window_trees(
    input_trees: Union[PathLike, tskit.TreeSequence],
    out_dir: PathLike,
    *,
    window_size: int,
    num_windows: int,
    recomb_rate: float = 1e-8,
    window_index: Optional[int] = None,
) -> List[Dict[str, Path]]:
    """
    Chop a tree sequence into `num_windows` equally-spaced, possibly-
    overlapping windows of `window_size` bp at the tree-sequence level:
    each window is sliced with `ts.keep_intervals([[start, end]])` then
    `.ltrim().rtrim()`, re-basing its coordinates to start at 0. The
    window's VCF is written directly from that trimmed tree sequence, so
    window_<i>.trees and window_<i>.vcf.gz always agree exactly.

    Returns a list of {"trees": Path, "vcf_gz": Path} dicts, one per
    window written.
    """
    ts = (
        input_trees
        if isinstance(input_trees, tskit.TreeSequence)
        else tskit.load(str(input_trees))
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = compute_window_bounds(0, ts.sequence_length, window_size, num_windows)
    indices = _resolve_indices(bounds, num_windows, window_index)

    written: List[Dict[str, Path]] = []
    for i in indices:
        w_start, w_end = bounds[i]
        sub = ts.keep_intervals([[w_start, w_end]], simplify=True).ltrim().rtrim()

        trees_out = out_dir / f"window_{i}.trees"
        sub.dump(str(trees_out))

        raw_vcf = out_dir / f"window_{i}.vcf"
        with raw_vcf.open("w") as fh:
            sub.write_vcf(fh, allow_position_zero=True)
        vcf_gz = out_dir / f"window_{i}.vcf.gz"
        with raw_vcf.open("rb") as f_in, gzip.open(vcf_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        raw_vcf.unlink()

        written.append({"trees": trees_out, "vcf_gz": vcf_gz})

    _write_samples_from_ts(ts, out_dir)
    _write_flat_map(out_dir, 0, window_size, recomb_rate)
    return written


def _write_region_genetic_map(rec_cfg: dict, out_dir: Path) -> Path:
    """
    genetic_map.txt for a simulation run on a real-map region (config
    "recombination": {"type": "map", "file", "region": [start, end]}):
    the real map file's rows inside the region, shifted so the region starts at
    0 (the simulation's coordinates) -- the same map the real-data LD step bins
    with, so simulated and real LD stats are binned identically.
    """
    import numpy as np

    start, end = (float(x) for x in rec_cfg["region"])
    arr = np.loadtxt(rec_cfg["file"], skiprows=1)
    pos, cm = arr[:, 0], arr[:, 1]
    inside = (pos > start) & (pos < end)
    xs = np.concatenate([[start], pos[inside], [end]])
    ys = np.interp(xs, pos, cm)
    lines = ["pos\tMap(cM)"] + [f"{x - start:.0f}\t{y - ys[0]:.6f}" for x, y in zip(xs, ys)]
    map_out = out_dir / "genetic_map.txt"
    map_out.write_text("\n".join(lines) + "\n")
    return map_out


def _pair_haploid_vcf(raw_vcf: Path, ts: tskit.TreeSequence, out_dir: Path) -> None:
    """
    Rewrite a haploid-sample VCF (one column per haploid individual, as
    ts.write_vcf writes for sample_ploidy=1) in place as phased pseudo-diploids:
    consecutive haploids within each population become one "a|b" sample, an odd
    one out is dropped -- the same pairing the real data gets
    (make_pseudodiploid_pairs.py + recode_haploid_to_diploid.py; simulated
    haploids are exchangeable, so which ones are paired doesn't matter).
    samples.txt is rewritten with the pair names.
    """
    pop_names = {}
    for pid in range(ts.num_populations):
        md = ts.population(pid).metadata
        pop_names[pid] = (md.get("name") if isinstance(md, dict) else None) or f"pop_{pid}"

    sample_set = {int(u) for u in ts.samples()}
    cols_by_pop: Dict[str, List[int]] = {}
    col = 0
    for ind in ts.individuals():
        nodes = [int(n) for n in ind.nodes if int(n) in sample_set]
        if not nodes:
            continue
        if len(nodes) != 1:
            raise ValueError("pseudo-diploid pairing expects haploid individuals (sample_ploidy=1)")
        cols_by_pop.setdefault(pop_names[ts.node(nodes[0]).population], []).append(9 + col)
        col += 1

    pairs = []  # (name, pop, col_a, col_b)
    for pop, cols in cols_by_pop.items():
        for k in range(len(cols) // 2):
            pairs.append((f"{pop}_pair{k}", pop, cols[2 * k], cols[2 * k + 1]))

    paired = raw_vcf.with_suffix(".paired.vcf")
    with raw_vcf.open() as fin, paired.open("w") as fout:
        for line in fin:
            if line.startswith("##"):
                fout.write(line)
                continue
            f = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                fout.write("\t".join(f[:9] + [p[0] for p in pairs]) + "\n")
                continue
            fout.write("\t".join(f[:8] + ["GT"] + [f"{f[a]}|{f[b]}" for _, _, a, b in pairs]) + "\n")
    paired.replace(raw_vcf)

    (out_dir / "samples.txt").write_text(
        "sample\tpop\n" + "".join(f"{name}\t{pop}\n" for name, pop, _, _ in pairs)
    )


def materialize_full_vcf(
    input_trees: Union[PathLike, tskit.TreeSequence],
    out_dir: PathLike,
    cfg: Optional[dict] = None,
) -> Path:
    """
    Write the WHOLE tree sequence to one bgzipped, tabix-indexed VCF, plus
    samples.txt (same sample-naming convention as window_trees).

    No keep_intervals/simplify at all -- this does a single, full-genome
    `write_vcf()` pass, then leaves per-window extraction to `window_vcf()`'s
    `bcftools view -r` slicing against the finished, indexed file. Measured
    ~4.6x faster than window_trees()'s per-window keep_intervals(simplify=True)
    approach on a test tree sequence, and unlike that approach, slicing here
    is read-only against a finished file, so it's safe to parallelize across
    windows again (each window can be its own independent job).

    With the experiment config `cfg`, the LD inputs are made to match the
    real data's: sample_ploidy 1 -> the haploid samples are paired into phased
    pseudo-diploids (LD then computed in haplotype mode), and a
    "recombination": {"type": "map"} region -> genetic_map.txt (that region of
    the real map, shifted to start at 0) is written for LD binning.

    Returns the path to the indexed full_genome.vcf.gz.
    """
    ts = (
        input_trees
        if isinstance(input_trees, tskit.TreeSequence)
        else tskit.load(str(input_trees))
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or {}

    raw_vcf = out_dir / "full_genome.vcf"
    vcf_gz = out_dir / "full_genome.vcf.gz"

    with raw_vcf.open("w") as fh:
        ts.write_vcf(fh, allow_position_zero=True)

    _write_samples_from_ts(ts, out_dir)
    if int(cfg.get("sample_ploidy", 2)) == 1:
        _pair_haploid_vcf(raw_vcf, ts, out_dir)
    rec = cfg.get("recombination") or {}
    if rec.get("type", "flat") == "map":
        _write_region_genetic_map(rec, out_dir)

    _run(f"bgzip -f '{raw_vcf}'")
    _run(f"bcftools index -t '{vcf_gz}'")

    return vcf_gz


def window_sequence(input_path: PathLike, out_dir: PathLike, **kwargs):
    """Dispatch to window_trees or window_vcf based on `input_path`'s suffix."""
    input_path = Path(input_path)
    suffixes = "".join(input_path.suffixes).lower()
    if suffixes.endswith(".trees") or suffixes.endswith(".ts"):
        return window_trees(input_path, out_dir, **kwargs)
    if suffixes.endswith(".vcf") or suffixes.endswith(".vcf.gz"):
        return window_vcf(input_path, out_dir, **kwargs)
    raise ValueError(f"Unrecognized input suffix for windowing: {input_path}")


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Chop a .trees or .vcf(.gz) file into windows for LD analysis."
    )
    p.add_argument("--input", required=True, type=Path, help="Input .trees or .vcf(.gz)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--window-size", type=int, default=1_000_000)
    p.add_argument("--num-windows", type=int, default=100)
    p.add_argument(
        "--popfile",
        type=Path,
        default=None,
        help="sampleID popID file (VCF input only; ignored for .trees)",
    )
    p.add_argument("--recomb-rate", type=float, default=1e-8)
    p.add_argument("--window-index", type=int, default=None)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    kwargs = dict(
        window_size=args.window_size,
        num_windows=args.num_windows,
        recomb_rate=args.recomb_rate,
        window_index=args.window_index,
    )
    if args.input.suffix != ".trees" and not str(args.input).endswith(".ts"):
        kwargs["popfile"] = args.popfile
    written = window_sequence(args.input, args.out_dir, **kwargs)
    print(f"Wrote {len(written)} window(s) to {args.out_dir}")


if __name__ == "__main__":
    main()

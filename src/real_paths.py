"""
Single source of truth for real-data (Drosophila) output paths.

Imported by the Snakefile; bash drivers can call it too:
    eval "$(python -m src.real_paths --config <experiment_config.json> --shell)"

Folder names are built from the settings that actually differ between runs,
so they stay short and two different runs never share a folder. Everything a
name does NOT spell out (exact trim coordinates, r-bins, pairing seed, ...) is
recorded in a settings.json inside that folder; check_settings() refuses to
reuse a folder whose settings.json disagrees with the current config.

Layout (e.g. trimmed data, Chr3L, 100 kb windows):
    real_data_analysis/data/drosophila/                    region-independent inputs
        popfile.txt, pseudodiploid_*.txt, genetic_maps/{chrom}/genetic_map.txt
    real_data_analysis/data/drosophila_trimmed/            processed data
        settings.json
        {chrom}/polarized.vcf.gz, polarized.diploidGT.vcf.gz, unfolded.sfs.pkl
        combined/autosomes.unfolded.sfs.pkl
        ld/Chr3L_100kb/                             one LD run
            settings.json
            {arm}/windows/, {arm}/LD_stats/, {arm}/means.varcovs.pkl
    experiments/{model}/real_trimmed/                      fits on that data
        runs/, inferences/                                 combined-autosome SFS fits (+ pooled MomentsLD)
        {chrom}/runs/, {chrom}/inferences/                 per-arm fits
            inferences/moments/, inferences/MomentsLD_100kb/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

DROSO_BASE_DIR = "real_data_analysis/data/drosophila"
PSEUDODIPLOID_SEED = 42


def _bp_label(bp: int) -> str:
    if bp % 1_000_000 == 0:
        return f"{bp // 1_000_000}Mb"
    if bp % 1_000 == 0:
        return f"{bp // 1_000}kb"
    return f"{bp}bp"


def _arms_label(arms) -> str:
    # ["Chr3L"] -> "Chr3L";  ["Chr2L","Chr3L","ChrX"] -> "Chr2L-3L-X"
    short = [a[3:] if a.startswith("Chr") else a for a in arms]
    return "Chr" + "-".join(short)


def real_paths(cfg: Dict[str, Any], model: str, r_bins: str = "") -> Dict[str, Any]:
    rd = cfg.get("real_data_analysis", {})
    trim = {c: [int(v[0]), int(v[1])] for c, v in sorted(rd.get("trim_region", {}).items())}
    arms = list(rd.get("arms", ["Chr3L"]))
    window_bp = int(rd.get("window_size_bp", 10_000_000))
    nw = rd.get("num_windows", 100)
    nw = "auto" if str(nw).lower() == "auto" else int(nw)
    # Haploid samples left out before pairing (e.g. inversion carriers).
    excl = sorted(rd.get("exclude_samples", []))
    excl_tag = ("excl-" + "-".join(excl)) if excl else ""
    # Drop LD windows within this many bp of any inversion breakpoint on the arm.
    bp_buffer = int(rd.get("exclude_breakpoint_buffer_bp", 0) or 0)

    data_label = "trimmed" if trim else "untrimmed"
    droso_dir = f"{DROSO_BASE_DIR}_trimmed" if trim else DROSO_BASE_DIR

    # LD-run name: only what varies between runs. num_windows="auto" means
    # non-overlapping tiling, so it adds nothing; a fixed count is spelled out.
    win = _bp_label(window_bp) + ("" if nw == "auto" else f"x{nw}")
    excl_label = f"_{excl_tag}" if excl else ""
    buf_label = f"_bpbuf{_bp_label(bp_buffer)}" if bp_buffer else ""
    tags = f"{excl_label}{buf_label}"
    ld_name = f"{_arms_label(arms)}_{win}{tags}"
    ld_engine_arm = f"MomentsLD_{win}{tags}"              # per-arm fit subdir
    ld_engine = f"MomentsLD_{ld_name}"                        # pooled fit subdir

    fit_root = f"experiments/{model}/real_{data_label}"

    data_settings = {
        "trim_region": trim,
        "pseudodiploid_seed": PSEUDODIPLOID_SEED,
    }
    ld_settings = {
        **data_settings,
        "exclude_samples": excl,
        "exclude_breakpoint_buffer_bp": bp_buffer,
        "arms": arms,
        "window_size_bp": window_bp,
        "num_windows": nw,
        "r_bins": r_bins,
    }

    return {
        "DROSO_BASE_DIR": DROSO_BASE_DIR,
        "DROSO_DIR": droso_dir,
        "REAL_LD_NAME": ld_name,
        "EXCLUDE_SAMPLES": ",".join(excl),
        "BREAKPOINT_BUFFER_BP": bp_buffer,
        # pairing files + diploid-VCF suffix carry the exclusion so runs with
        # different sample sets never share a file
        "PSEUDODIPLOID_PAIRS": f"{DROSO_BASE_DIR}/pseudodiploid_pairs.seed{PSEUDODIPLOID_SEED}{'.' + excl_tag if excl else ''}.tsv",
        "PSEUDODIPLOID_POPFILE": f"{DROSO_BASE_DIR}/pseudodiploid_popfile.seed{PSEUDODIPLOID_SEED}{'.' + excl_tag if excl else ''}.txt",
        "DIPLOID_SUFFIX": f".{excl_tag}" if excl else "",
        "REAL_LD_ROOT": f"{droso_dir}/ld/{ld_name}",
        "REAL_LD_ENGINE": ld_engine,
        "REAL_LD_ENGINE_ARM": ld_engine_arm,
        "REAL_FIT_ROOT": fit_root,
        "REAL_RUN_ROOT": f"{fit_root}/runs",
        "REAL_INF_ROOT": f"{fit_root}/inferences",
        "REAL_RUN_ROOT_CHROM": f"{fit_root}/{{chrom}}/runs",
        "REAL_INF_ROOT_CHROM": f"{fit_root}/{{chrom}}/inferences",
        "GENMAP_DIR": f"{DROSO_BASE_DIR}/genetic_maps",
        "_settings": {droso_dir: data_settings, f"{droso_dir}/ld/{ld_name}": ld_settings},
    }


def check_settings(settings_by_dir: Dict[str, Dict[str, Any]]) -> None:
    """Raise if an existing folder's settings.json disagrees with the config."""
    for d, want in settings_by_dir.items():
        f = Path(d) / "settings.json"
        if not f.exists():
            continue
        have = json.loads(f.read_text())
        if have != json.loads(json.dumps(want)):
            diff = {k: (have.get(k), want.get(k)) for k in set(have) | set(want) if have.get(k) != want.get(k)}
            raise ValueError(
                f"{d} was made with different settings than the current config "
                f"(setting: (on disk, config)) {diff}. Archive or rename that folder, "
                f"or change the config back."
            )


def write_settings(settings_by_dir: Dict[str, Dict[str, Any]]) -> None:
    """Write settings.json into each folder that exists and doesn't have one yet."""
    for d, want in settings_by_dir.items():
        f = Path(d) / "settings.json"
        if Path(d).is_dir() and not f.exists():
            f.write_text(json.dumps(want, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Print real-data paths for a config.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--shell", action="store_true", help="print KEY=\"value\" lines for bash eval")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    p = real_paths(cfg, cfg["demographic_model"])
    for k, v in p.items():
        if k.startswith("_"):
            continue
        print(f'{k}="{v}"' if args.shell else f"{k}: {v}")


if __name__ == "__main__":
    main()

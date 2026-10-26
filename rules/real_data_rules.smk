##############################################################################
# Real data: VCF prep -> SFS/LD inference -> prediction with trained models
##############################################################################

##############################################################################
# Real-data-only settings and paths
##############################################################################
NUM_REAL_OPTIMS = int(CFG.get("num_optimizations", 3))

USE_GPU_LD = CFG.get("use_gpu_ld", False)

# Optional real-data region trim (per-chrom), e.g. to drop centromere/telomere-
# proximal ends: config_files/*.json -> "real_data_analysis": {"trim_region":
# {"Chr3L": [447386, 18392988]}}. Keyed by chrom so multiple arms can carry
# independent bounds. Absent -> the whole arm (see real_vcf_for_chrom below).
REAL_DATA_CFG = CFG.get("real_data_analysis", {})
REAL_TRIM_REGION = {
    c: (int(v[0]), int(v[1]))
    for c, v in REAL_DATA_CFG.get("trim_region", {}).items()
}

# All real-data folder names come from src/real_paths.py (shared with the bash
# drivers): short names built from the settings that vary between runs, with
# the rest (exact trim coordinates, r-bins, pairing seed) in a settings.json
# inside each folder.
from src.real_paths import real_paths, check_settings, write_settings, PSEUDODIPLOID_SEED
RP = real_paths(CFG, CFG["demographic_model"])

# Per-arm input VCF, built from the original DPGP2 VCF (real_data_analysis.
# original_vcf, a "{chrom}" path pattern; read-only) by rule make_real_vcf.
# Derived output only -- never written next to the raw data.
REAL_ORIGINAL_VCF = REAL_DATA_CFG.get("original_vcf")

def _original_vcf(w):
    if not REAL_ORIGINAL_VCF:
        raise ValueError("real_data_analysis.original_vcf is not set in the experiment config")
    return REAL_ORIGINAL_VCF.format(chrom=w.chrom)
REAL_VCF_DIR = "real_data_analysis/data/real_vcf"

def real_vcf_for_chrom(chrom):
    """Input VCF for this chrom, tagged with its trim region (or "whole")."""
    rng = REAL_TRIM_REGION.get(chrom)
    region = f"trim{rng[0]}-{rng[1]}" if rng else "whole"
    return f"{REAL_VCF_DIR}/{chrom}.{region}.vcf.gz"

# real_data_analysis.sfs_projection, e.g. {"CO": 9, "FR": 8}: haplotypes per
# population the SFS is projected to, so sites with a few missing flies still
# count. Absent -> full sample size (any site with a missing fly is skipped).
REAL_SFS_PROJECTION = REAL_DATA_CFG.get("sfs_projection") or {}

# Make sure these match files that actually exist in your repo
DROSO_BASE_DIR   = RP["DROSO_BASE_DIR"]   # region-independent inputs (popfile, pairing, genetic maps)
DROSO_DIR        = RP["DROSO_DIR"]        # processed data: drosophila_trimmed/ (or drosophila/ when untrimmed)
GENMAP_DIR       = RP["GENMAP_DIR"]       # {GENMAP_DIR}/{chrom}/genetic_map.txt (Comeron, region/model-independent)
AUTOSOMES        = ["Chr2L", "Chr2R", "Chr3L", "Chr3R"]          # all autosomal arms (ChrX excluded: not an autosome)
ANCESTRAL_DIR    = "drosophila_data/dpgp_ancestor"                        # relative to repo root, alongside drosophila_data/data/

# popfile.txt (sample->population map) doesn't depend on the genomic region
# analyzed, so it stays on the untagged base dir -- otherwise every new
# trim_region tag would need its own copy of an identical file.
REAL_POPFILE     = f"{DROSO_BASE_DIR}/popfile.txt"
# Pseudo-diploid pairing of the haploid samples (for MomentsLD). Region-
# independent like popfile.txt, so every chrom/trim uses the same pairs.
PSEUDODIPLOID_PAIRS   = RP["PSEUDODIPLOID_PAIRS"]     # tagged .excl-<ids> when real_data_analysis.exclude_samples is set
PSEUDODIPLOID_POPFILE = RP["PSEUDODIPLOID_POPFILE"]
DIPLOID_SUFFIX        = RP["DIPLOID_SUFFIX"]          # "" or ".excl-<ids>" in the diploid VCF names
SFS_SUFFIX            = RP["SFS_SUFFIX"]              # "" or ".excl-<ids>" in the SFS names (same exclude_samples)
EXCLUDE_SAMPLES       = RP["EXCLUDE_SAMPLES"]
COMBINED_SFS     = f"{DROSO_DIR}/combined/autosomes.unfolded{SFS_SUFFIX}.sfs.pkl"     # summed autosomal SFS; used by SFS inference
COMBINED_SFS_META = f"{DROSO_DIR}/combined/autosomes.unfolded{SFS_SUFFIX}.sfs.meta.json"  # summed sequence_length across AUTOSOMES

# Per-chromosome path helpers (by-chromosome layout)
# Phased (a|b) pseudo-diploids: LD stats come from the two known haplotypes.
def polarized_diploid_vcf(chrom):  return f"{DROSO_DIR}/{chrom}/polarized.diploidGT{DIPLOID_SUFFIX}.phased.vcf.gz"
def per_chrom_sfs(chrom):          return f"{DROSO_DIR}/{chrom}/unfolded{SFS_SUFFIX}.sfs.pkl"
def per_chrom_sfs_meta(chrom):     return f"{DROSO_DIR}/{chrom}/unfolded{SFS_SUFFIX}.sfs.meta.json"
def ancestral_fasta(chrom):        return f"{ANCESTRAL_DIR}/{chrom.replace('Chr', 'chr', 1)}.q30.fa"

# Real-data prediction needs inference features, so raw_* variants are skipped there.
REAL_VARIANTS = [v for v in MODELING_VARIANTS if v in INFERENCE_VARIANTS]

# Which arm(s) the real-data Moments-LD chain runs on, and at what window
# size/count, are config-driven so this can be flipped between e.g. Chr3L-only
# and pooled Chr2L+Chr3L without editing the Snakefile. Deliberately a NEW,
# separate config key from the top-level "num_windows" (used only by the
# simulated-data windowing path above) -- the two must not be conflated.
# num_windows accepts a fixed int (same value applied to every configured arm)
# or the literal string "auto" (per-arm count = arm_span // window_size_bp,
# runtime-determined since the actual VCF POS span isn't known until read --
# mirrors godambe_correction_LRT/Snakefile's `checkpoint tile_nonoverlap`).
# Overlap vs. non-overlap is never chosen explicitly: split_vcf_windows.py's
# existing sliding-window formula produces whichever naturally falls out of
# (window_size, num_windows, actual span) -- e.g. today's Chr3L defaults
# (10 Mb, 100) yield overlapping windows because 100x10Mb can't fit
# non-overlapping in a ~28Mb arm; "auto" at a smaller window_size yields
# non-overlapping windows because the count is chosen to fit exactly.
# Any experiment config JSON without this block falls back to arms=["Chr3L"],
# window_size_bp=10_000_000, num_windows=100 -- byte-identical to the
# pre-existing hardcoded single-arm behavior.
REAL_ARMS       = REAL_DATA_CFG.get("arms", ["Chr3L"])
# "pooled" (default, existing behavior): every configured arm's LD windows are
# flattened together (materialize_real_ld_stats) and fit as ONE combined
# MomentsLD model. "individual": each arm keeps its own windows/LD_stats
# (already arm-separated upstream) AND its own means/varcovs/bootstrap/
# optimization -- the {scope}="<arm>/" case of aggregate_ld_windows_real /
# infer_momentsld_real / aggregate_opts_momentsld_real. Nothing is combined across arms
# in "individual" mode until/unless you explicitly request the pooled target.
REAL_POOLING_MODE = REAL_DATA_CFG.get("pooling_mode", "pooled")
if REAL_POOLING_MODE not in ("pooled", "individual"):
    raise ValueError(f'real_data_analysis.pooling_mode must be "pooled" or "individual", got {REAL_POOLING_MODE!r}')
REAL_WINDOW_BP  = int(REAL_DATA_CFG.get("window_size_bp", 10_000_000))
_raw_real_nw    = REAL_DATA_CFG.get("num_windows", 100)
REAL_NUM_WINDOWS_MODE = "auto" if str(_raw_real_nw).lower() == "auto" else int(_raw_real_nw)
# MomentsLD_real_data.py always needs SOME N_ref to satisfy resolve_n_ref(),
# but only actually USES it for rho-scaling when momentsld_use_scaled_units
# is True -- in absolute-units mode (the current default), rho is rescaled
# from each candidate's own N_ANC instead, so N_ref is otherwise vestigial.
# When scaled units are off, infer_momentsld_real(_by_arm) skip the
# moments/dadi seed fit entirely and pass a placeholder --n-ref instead, so
# MomentsLD can run standalone without requiring moments/dadi to have been
# fit first.
REAL_USE_SCALED_UNITS = bool(CFG.get("momentsld_use_scaled_units", True))
# MomentsLD_real_data.py only reads the moments SFS fit to anchor N_ref (scaled
# units) or to pin a fixed_parameters entry set to "moments_best"; otherwise
# the real MomentsLD rules don't wait on the SFS fit at all.
MOMENTSLD_NEEDS_SFS_FIT = REAL_USE_SCALED_UNITS or any(
    v == "moments_best" for v in CFG.get("fixed_parameters", {}).values()
)
def _momentsld_seed_flag(w, input):
    if not MOMENTSLD_NEEDS_SFS_FIT:
        return "--n-ref 1.0"
    flag = f'--sfs-best-fit-pkl "{input.sfs_best}"'
    return flag if REAL_USE_SCALED_UNITS else f"{flag} --n-ref 1.0"
# LD-run name from arms + window + exclusions, e.g. "Chr3L_100kb" (see
# src/real_paths.py). REAL_LD_ENGINE names the pooled MomentsLD fit subdir,
# REAL_LD_ENGINE_ARM the per-arm one (arm is already in that path).
REAL_LD_ENGINE     = RP["REAL_LD_ENGINE"]
REAL_LD_ENGINE_ARM = RP["REAL_LD_ENGINE_ARM"]

# Real-data LD windows/LD_stats/aggregated means+varcovs are pure functions of
# the VCF data (arms, window size, r_bins) -- not of which demographic model
# you're fitting -- so this lives under DROSO_DIR (shared across every model)
# instead of experiments/{MODEL}/...: switching MODEL never re-triggers the
# window split or the (GPU-bound) per-window LD computation. The MomentsLD
# *fit itself* (aggregate_opts_momentsld_real's best_fit.pkl) is genuinely
# model-specific and lives under REAL_INF_ROOT instead -- see that rule below.
REAL_LD_ROOT = RP["REAL_LD_ROOT"]   # {DROSO_DIR}/ld/{REAL_LD_NAME}
# Real-data LD always converts bp -> genetic distance with the Comeron (R5/dm3)
# recombination map (build_genetic_map_real turns this into GENMAP_DIR files).
COMERON_XLSX       = "drosophila_data/recombination_maps/Comeron_100kb_R5_R6.xlsx"  # input data (like the VCFs), not a pipeline output
# infer_engine_real/aggregate_opts_engine_real fit COMBINED_SFS, which is
# itself trim-tagged (built under DROSO_DIR) -- so these fit outputs need the
# same tag, otherwise a trimmed run would silently overwrite the full-arm
# moments/dadi best_fit.pkl already sitting here. Untagged when trim_region
# is unset, matching every path above.
REAL_FIT_ROOT = RP["REAL_FIT_ROOT"]   # experiments/{MODEL}/real_trimmed (or real_untrimmed)
REAL_INF_ROOT = RP["REAL_INF_ROOT"]   # pooled fits; per-arm ones are {scope}="<arm>/" (see _scope_arm)

# Per-arm restart folder ({chrom} wildcard), read by calibration_simulate_fit.
REAL_RUN_ROOT_CHROM = RP["REAL_RUN_ROOT_CHROM"]

# Number of top replicates the real moments/dadi aggregation keeps.
# Must match the rep count the trained model was built with (moments_*_rep_0..N-1),
# i.e. the sim-side TOP_K -- defaults to TOP_K unless overridden.
REAL_TOP_K    = int(CFG.get("real_top_k", TOP_K))

# ── Real-data prediction (push real fits through a trained model) ───────────
# Generalized over {variant} (the same MODELING_VARIANTS the sim pipeline
# trains) instead of pinning to a single modeling dir via a
# real_predict_modeling_dir config override -- so real-data predictions can
# be compared with/without FIM and SFS-residual features. Each variant's
# outputs live under their own prediction_{variant}/ dir.
REAL_PRED_ROOT = f"{REAL_FIT_ROOT}/prediction_{{variant}}"

def _real_modeling_dir(variant):
    return f"experiments/{MODEL}/modeling_{variant}"

def _real_train_features(variant):
    return f"{_real_modeling_dir(variant)}/datasets/features_df.pkl"

def _real_empirical_norm_stats(variant):
    return f"{_real_modeling_dir(variant)}/datasets/empirical_norm_stats.json"

def _real_norm_train_features(variant):
    return f"{_real_modeling_dir(variant)}/datasets/normalized_train_features.pkl"

# model_key wildcard -> trained mdl_obj.pkl path (rule train_model), for a given variant
def _real_model_objs(variant):
    return {k: f"{TRAINED_MODEL_DIR.format(variant=variant, model_key=k)}/mdl_obj.pkl" for k in MODEL_KEYS}

# model_key set is the same across variants.
REAL_MODEL_KEYS = MODEL_KEYS

# Model-calibration check (posterior-predictive): simulate at the real-data
# fitted params, one job per (variant, model_key, rep) so replicates can run
# as parallel SLURM array tasks (see bash_scripts/simulation/calibration_simulate.sh) --
# each array task builds just its own replicate_{rep} target.
CALIBRATION_ROOT = f"{REAL_FIT_ROOT}/calibration_{{variant}}/{{model_key}}/replicate_{{rep}}"
NUM_CALIBRATION_REPLICATES = int(CFG.get("calibration_n_replicates", 20))
CALIBRATION_REPS = list(range(NUM_CALIBRATION_REPLICATES))

# {scope}: "" for the pooled fit (combined autosomes / all REAL_ARMS together),
# "<arm>/" for one arm's own fit. It slots straight into the existing layouts
# (<root>/runs vs <root>/<arm>/runs, REAL_LD_ROOT vs REAL_LD_ROOT/<arm>), so one
# rule serves both pooling modes.
def _scope_arm(scope):
    return scope.rstrip("/")            # "" when pooled

def _scope_sfs(scope):
    arm = _scope_arm(scope)
    return per_chrom_sfs(arm) if arm else COMBINED_SFS

def _scope_sfs_meta(scope):
    arm = _scope_arm(scope)
    return per_chrom_sfs_meta(arm) if arm else COMBINED_SFS_META

def _ld_engine(scope):
    """MomentsLD fit folder name: the pooled one also encodes the arm set."""
    return REAL_LD_ENGINE_ARM if scope else REAL_LD_ENGINE

def _check_ld_engine(w):
    if w.ld_engine != _ld_engine(w.scope):
        raise ValueError(f"MomentsLD folder {w.ld_engine!r} doesn't belong to scope {w.scope!r}; expected {_ld_engine(w.scope)!r}")

wildcard_constraints:
    chrom      = r"Chr(2L|2R|3L|3R|X)",
    arm        = r"Chr(2L|2R|3L|3R|X)",
    start      = r"\d+",
    end        = r"\d+",
    # TEMPORARY: widened to also allow the raw dadi/moments/momentsLD fits
    # (not just trained ML model keys) as calibration_simulate/calibration_ppc/
    # calibration_ld_ppc model_key values, so those PPC checks can run against
    # the direct SFS/LD fit for comparison against the ML surrogate's PPC.
    # Revert to just REAL_MODEL_KEYS once that comparison is done.
    model_key  = "|".join(REAL_MODEL_KEYS) + "|dadi|moments|momentsLD",
    scope      = r"(Chr(2L|2R|3L|3R|X)/)?",
    ld_engine  = "|".join(re.escape(e) for e in sorted({REAL_LD_ENGINE, REAL_LD_ENGINE_ARM})),
    rep        = r"\d+",

# Refuse to reuse a real-data folder whose settings.json disagrees with the
# current config (e.g. trim coordinates changed but the folder name didn't);
# settings.json is written into whichever of these folders a (non-dry) run
# actually created.
REAL_FOLDER_SETTINGS = real_paths(CFG, CFG["demographic_model"], r_bins=R_BINS_STR)["_settings"]
check_settings(REAL_FOLDER_SETTINGS)

onsuccess:
    write_settings(REAL_FOLDER_SETTINGS)

onerror:
    write_settings(REAL_FOLDER_SETTINGS)



##############################################################################
# RULE make_real_vcf  (per-chromosome)
# Build the arm's haploid input VCF from the original DPGP2 VCF: popfile
# samples only, restricted to the trim region (or the whole arm), "*" (N)
# calls written as missing genotypes, and only sites variable among the
# called flies. Sites with missing flies are KEPT (compute_unfolded_sfs
# projects them). The ##contig length is the region span, so
# compute_unfolded_sfs.py's L (theta = 4*mu*L*N_ANC) is the region analyzed.
##############################################################################
rule make_real_vcf:
    input:
        vcf     = _original_vcf,
        popfile = REAL_POPFILE,
    output:
        vcf = f"{REAL_VCF_DIR}/{{chrom}}.{{region}}.vcf.gz",
        tbi = f"{REAL_VCF_DIR}/{{chrom}}.{{region}}.vcf.gz.tbi",
    wildcard_constraints:
        region = r"whole|trim\d+-\d+",
    params:
        bounds = lambda w: ("" if w.region == "whole" else
                            "--start {} --end {}".format(*w.region[4:].split("-"))),
    threads: 1
    shell:
        r"""
        set -euo pipefail
        python snakemake_scripts/make_real_vcf.py \
          --original-vcf "{input.vcf}" --popfile "{input.popfile}" \
          --chrom {wildcards.chrom} {params.bounds} --out "{output.vcf}"
        """

##############################################################################
# RULE annotate_ancestral_allele  (per-chromosome, autosomes)
# Polarize the raw haploid VCF using the DPGP ML-ancestor FASTA.
# FASTA naming differs from the VCF: "Chr2L" -> "chr2L.q30.fa".
# Adds an AA= INFO field; sites where the ancestral base is N or matches
# neither allele are dropped.  Output is bgzipped + tabix-indexed.
##############################################################################
rule annotate_ancestral_allele:
    input:
        vcf   = lambda wc: real_vcf_for_chrom(wc.chrom),
        tbi   = lambda wc: real_vcf_for_chrom(wc.chrom) + ".tbi",
        fasta = lambda wc: ancestral_fasta(wc.chrom),
    output:
        vcf = f"{DROSO_DIR}/{{chrom}}/polarized.vcf.gz",
        tbi = f"{DROSO_DIR}/{{chrom}}/polarized.vcf.gz.tbi",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname "{output.vcf}")"
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/annotate_ancestral_allele.py \
          --input-vcf       "{input.vcf}" \
          --ancestral-fasta "{input.fasta}" \
          --output-vcf      "{output.vcf}"
        """

##############################################################################
# RULE make_pseudodiploid_pairs
# Shuffle each population's haploid samples (fixed seed) and split them into
# disjoint pairs -- no sample used twice, no cross-population pairs; an odd
# leftover is dropped. Writes the pairing table and the matching popfile
# (names "<hap1>_<hap2>") that the MomentsLD rules use with the diploid VCF.
##############################################################################
rule make_pseudodiploid_pairs:
    input:
        popfile = REAL_POPFILE,
    output:
        pairs   = PSEUDODIPLOID_PAIRS,
        popfile = PSEUDODIPLOID_POPFILE,
    params:
        seed = PSEUDODIPLOID_SEED,
        exclude = EXCLUDE_SAMPLES,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        python snakemake_scripts/make_pseudodiploid_pairs.py \
          --popfile     "{input.popfile}" \
          --seed        {params.seed} \
          --exclude     "{params.exclude}" \
          --out-pairs   "{output.pairs}" \
          --out-popfile "{output.popfile}"
        """

##############################################################################
# RULE recode_polarized_to_diploid  (per-chromosome, autosomes)
# Combine pairs of haploid samples (from make_pseudodiploid_pairs) into phased
# (a|b) pseudo-diploid GTs, so MomentsLD uses exactly the same sites as the
# SFS analysis and computes LD from the two known haplotypes
# (use_genotypes=False). Unpaired haploid samples are dropped.
##############################################################################
rule recode_polarized_to_diploid:
    input:
        vcf   = f"{DROSO_DIR}/{{chrom}}/polarized.vcf.gz",
        tbi   = f"{DROSO_DIR}/{{chrom}}/polarized.vcf.gz.tbi",
        pairs = PSEUDODIPLOID_PAIRS,
    output:
        vcf = f"{DROSO_DIR}/{{chrom}}/polarized.diploidGT{DIPLOID_SUFFIX}.phased.vcf.gz",
        tbi = f"{DROSO_DIR}/{{chrom}}/polarized.diploidGT{DIPLOID_SUFFIX}.phased.vcf.gz.tbi",
    params:
        script = f"{workflow.basedir}/snakemake_scripts/recode_haploid_to_diploid.py",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname "{output.vcf}")"
        out="{output.vcf}"
        tmp_vcf="${{out%.gz}}"
        python "{params.script}" "{input.vcf}" "$tmp_vcf" "{input.pairs}"
        bgzip -f "$tmp_vcf"
        tabix -f -p vcf "{output.vcf}"
        """

##############################################################################
# RULE compute_unfolded_sfs  (per-chromosome, autosomes)
# Build the 2D unfolded SFS directly from the polarized haploid VCF.
# Each sample contributes 1 chromosome (no diploid recoding needed).
##############################################################################
rule compute_unfolded_sfs:
    input:
        vcf     = f"{DROSO_DIR}/{{chrom}}/polarized.vcf.gz",
        tbi     = f"{DROSO_DIR}/{{chrom}}/polarized.vcf.gz.tbi",
        unpolarized = lambda w: real_vcf_for_chrom(w.chrom),
        popfile = REAL_POPFILE,
    output:
        sfs  = per_chrom_sfs("{chrom}"),
        # sequence_length = region length (this VCF's ##contig header) x the
        # fraction of SNPs that made it into the SFS (polarized, and enough
        # called flies to project) -- the real-data
        # inference rules read it back out for theta -> N_ANC, so it always
        # matches whichever VCF actually built the SFS.
        meta = per_chrom_sfs_meta("{chrom}"),
        # Each population's own spectrum vs. the neutral 1/i expectation, plus
        # the joint SFS heatmap (log scale).
        png  = f"{DROSO_DIR}/{{chrom}}/unfolded{SFS_SUFFIX}.sfs.png",
    params:
        # same samples left out as the MomentsLD branch (make_pseudodiploid_pairs)
        exclude = EXCLUDE_SAMPLES,
        project = ",".join(f"{p}={n}" for p, n in REAL_SFS_PROJECTION.items()),
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/compute_unfolded_sfs.py \
          --input-vcf   "{input.vcf}" \
          --unpolarized-vcf "{input.unpolarized}" \
          --popfile     "{input.popfile}" \
          --exclude     "{params.exclude}" \
          --project     "{params.project}" \
          --output-sfs  "{output.sfs}" \
          --output-meta "{output.meta}" \
          --output-png  "{output.png}"
        """

##############################################################################
# RULE combine_autosomal_sfs
# Sum the per-chromosome unfolded SFSs for the four autosomes into one
# genome-wide (autosomal) spectrum (entry-by-entry; fixed-site corners masked).
##############################################################################
rule combine_autosomal_sfs:
    input:
        per_chrom      = [per_chrom_sfs(c) for c in AUTOSOMES],
        per_chrom_meta = [per_chrom_sfs_meta(c) for c in AUTOSOMES],
    output:
        sfs  = COMBINED_SFS,
        meta = COMBINED_SFS_META,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname "{output.sfs}")"
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/combine_sfs.py \
          --in-sfs {input.per_chrom} \
          --output-sfs "{output.sfs}" \
          --in-meta {input.per_chrom_meta} \
          --output-meta "{output.meta}"
        """

##############################################################################
# REAL DATA – NLopt Poisson SFS optimisation (moments / dadi), one restart.
# {scope} "" fits the combined autosomal SFS; "<arm>/" fits that arm's own SFS.
##############################################################################
rule infer_engine_real:
    input:
        sfs  = lambda w: _scope_sfs(w.scope),
        meta = lambda w: _scope_sfs_meta(w.scope),
    output:
        # kept (not temp) so each restart can be inspected after aggregation,
        # and so real_sfs_inference.sh's skip-if-exists check stays valid
        pkl = f"{REAL_FIT_ROOT}/{{scope}}runs/run_{{opt}}/inferences/{{engine}}/best_fit.pkl",
        png = f"{REAL_FIT_ROOT}/{{scope}}runs/run_{{opt}}/inferences/{{engine}}/sfs_fit.png",
        json = f"{REAL_FIT_ROOT}/{{scope}}runs/run_{{opt}}/inferences/{{engine}}/sfs_fit.json",
    params:
        run_dir  = lambda w: f"{REAL_FIT_ROOT}/{w.scope}runs/run_{w.opt}",
        label    = lambda w: f"{_scope_arm(w.scope) or 'autosomes'}, run {w.opt}",
        cfg      = EXP_CFG,
        model_py = (
            f"demes_models:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "demes_models:drosophila_three_epoch"
        ),
    threads: 2
    shell:
        r"""
        set -euo pipefail
        seq_len=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['sequence_length'])" "{input.meta}")
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/moments_dadi_inference_real.py \
          --mode {wildcards.engine} \
          --sfs-file "{input.sfs}" \
          --config "{params.cfg}" \
          --model-py "{params.model_py}" \
          --outdir "{params.run_dir}/inferences" \
          --opt-seed {wildcards.opt} \
          --real-sequence-length "$seq_len" \
          -v
        # observed vs. best-fit SFS for this run (+ its numbers in sfs_fit.json)
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/plot_sfs_fit_real.py \
          --fit-pkl "{output.pkl}" --sfs "{input.sfs}" --config "{params.cfg}" \
          --model-py "{params.model_py}" --label "{params.label}" \
          --out "{output.png}"
        """

# ── REAL DATA: MOMENTS / DADI top-K across restarts (pooled or one arm) ─────
rule aggregate_opts_engine_real:
    input:
        runs = lambda w: [f"{REAL_FIT_ROOT}/{w.scope}runs/run_{o}/inferences/{w.engine}/best_fit.pkl"
                          for o in range(NUM_REAL_OPTIMS)],
        sfs  = lambda w: _scope_sfs(w.scope),
    output:
        pkl = f"{REAL_FIT_ROOT}/{{scope}}inferences/{{engine}}/best_fit.pkl",
        png = f"{REAL_FIT_ROOT}/{{scope}}inferences/{{engine}}/sfs_fit.png",
        json = f"{REAL_FIT_ROOT}/{{scope}}inferences/{{engine}}/sfs_fit.json",
    params:
        cfg      = EXP_CFG,
        model_py = (
            f"demes_models:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "demes_models:drosophila_three_epoch"
        ),
    run:
        import pickle, pathlib
        from src.aggregate_utils import aggregate_top_k

        arm = _scope_arm(wildcards.scope)
        records = [(p, i) for i, p in enumerate(input.runs)]

        best, diag = aggregate_top_k(
            records, REAL_TOP_K,
            extra_fields=("theta_hat", "N_ANC_implied_from_theta"),
        )
        best = {"mode": wildcards.engine, **({"chrom": arm} if arm else {}), **best}

        pathlib.Path(output.pkl).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(best, open(output.pkl, "wb"))

        print(f"✅ [REAL{'/' + arm if arm else ''}] Aggregated {diag['n_entries']} {wildcards.engine} optimization results → {output.pkl}")

        # observed vs. best-fit SFS for the top run (+ its numbers in sfs_fit.json)
        shell(
            'PYTHONPATH={basedir} python snakemake_scripts/plot_sfs_fit_real.py '
            '--fit-pkl "{output.pkl}" --sfs "{input.sfs}" --config "{params.cfg}" '
            '--model-py "{params.model_py}" --label "{label}" --out "{output.png}"',
            basedir=workflow.basedir, label=f"{arm or 'autosomes'}, best of all runs",
        )


##############################################################################
# REAL DATA LD ANALYSIS
##############################################################################

# Per-arm window split. A checkpoint because the window COUNT is only known
# at runtime when REAL_NUM_WINDOWS_MODE == "auto" (arm_span // REAL_WINDOW_BP,
# span read from the VCF itself) -- when it's a fixed int the count is already
# known, but using a checkpoint uniformly for both cases means one rule body
# serves every REAL_ARMS/REAL_NUM_WINDOWS_MODE combination without branching
# into separate "pooled" vs. "single-arm" rules. Mirrors the proven recipe in
# godambe_correction_LRT/Snakefile's `checkpoint tile_nonoverlap`.
checkpoint real_vcf_windows:
    input:
        vcf     = lambda wc: polarized_diploid_vcf(wc.arm),
        popfile = PSEUDODIPLOID_POPFILE,
    output:
        windir = directory(f"{REAL_LD_ROOT}/{{arm}}/windows"),
    params:
        script      = "snakemake_scripts/split_vcf_windows.py",
        window_size = REAL_WINDOW_BP,
        num_windows = REAL_NUM_WINDOWS_MODE,
        # Match the constant rate simulations use (chunk_window reads the same
        # CFG["recombination_rate"] to build msprime truth + its flat map), so
        # real and simulated MomentsLD fits share one assumed rate instead of
        # split_vcf_windows.py's unrelated 1e-8/bp default.
        recomb_rate = float(CFG["recombination_rate"]),
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{output.windir}"

        if [ "{params.num_windows}" = "auto" ]; then
            set +o pipefail
            first=$(bcftools query -f '%POS\n' "{input.vcf}" | head -n1)
            last=$(bcftools query -f '%POS\n' "{input.vcf}" | tail -n1)
            set -o pipefail
            span=$((last - first + 1))
            n=$((span / {params.window_size}))
            if [ "$n" -lt 1 ]; then n=1; fi
        else
            n={params.num_windows}
        fi

        python "{params.script}" \
            --input-vcf "{input.vcf}" \
            --popfile "{input.popfile}" \
            --out-dir "{output.windir}" \
            --window-size "{params.window_size}" \
            --num-windows "$n" \
            --recomb-rate "{params.recomb_rate}"
        """


def _real_arm_window_idxs(arm):
    ck = checkpoints.real_vcf_windows.get(arm=arm)
    return sorted(
        glob_wildcards(os.path.join(ck.output.windir, "window_{i}.vcf.gz")).i,
        key=int,
    )


def gather_all_real_ld_stats(wildcards):
    """Every configured arm's per-window LD_stats pkls, in a stable order."""
    paths = []
    for arm in REAL_ARMS:
        for i in _real_arm_window_idxs(arm):
            paths.append(f"{REAL_LD_ROOT}/{arm}/LD_stats/LD_stats_window_{i}.pkl")
    return paths


def real_flat_ld_stats(wildcards):
    """Resolve materialize_real_ld_stats' checkpoint and return its actual
    flattened file list -- must glob its directory rather than independently
    predict filenames, since a directory() output doesn't let Snakemake match
    individual files inside it against a producer rule."""
    ck = checkpoints.materialize_real_ld_stats.get()
    flat_dir = ck.output.flat_dir
    idxs = glob_wildcards(os.path.join(flat_dir, "LD_stats_window_{j}.pkl")).j
    return expand(f"{REAL_LD_ROOT}/LD_stats/LD_stats_window_{{j}}.pkl", j=idxs)


rule compute_ld_real:
    input:
        vcf_gz = f"{REAL_LD_ROOT}/{{arm}}/windows/window_{{i}}.vcf.gz",
        gmap   = f"{GENMAP_DIR}/{{arm}}/genetic_map.txt",
    output:
        pkl = f"{REAL_LD_ROOT}/{{arm}}/LD_stats/LD_stats_window_{{i}}.pkl"
    resources:
        gpu = 1 if USE_GPU_LD else 0
    params:
        script  = "snakemake_scripts/compute_ld_window.py",
        config  = EXP_CFG,
        sim_dir = lambda wc: f"{REAL_LD_ROOT}/{wc.arm}",
        r_bins  = R_BINS_STR,
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.sim_dir}/LD_stats"

        python "{params.script}" \
            --sim-dir "{params.sim_dir}" \
            --window-index "{wildcards.i}" \
            --config-file "{params.config}" \
            --r-bins "{params.r_bins}" \
            --rec-map-file "{input.gmap}" \
            --haplotypes
        """


# Flatten every configured arm's LD_stats into one directory with fresh
# sequential indices -- LD_inference.py's aggregator globs
# "LD_stats_window_*.pkl" out of a single flat directory with no assumption
# about count/contiguity, so windows from multiple arm subdirectories need to
# be collected into one place first. Arm-count-agnostic: 1 arm or many, same
# rule body. A checkpoint (not a plain rule): its own output *count* is only
# known once gather_all_real_ld_stats resolves the upstream per-arm
# checkpoints, so downstream consumers must glob its directory afterward
# (real_flat_ld_stats below) rather than independently predicting filenames --
# a directory() output alone doesn't let Snakemake match individual files
# inside it against a rule.
checkpoint materialize_real_ld_stats:
    input:
        pkls = gather_all_real_ld_stats,
    output:
        flat_dir = directory(f"{REAL_LD_ROOT}/LD_stats"),
    run:
        import shutil, pathlib
        out = pathlib.Path(output.flat_dir)
        out.mkdir(parents=True, exist_ok=True)
        for j, src in enumerate(input.pkls):
            shutil.copyfile(src, out / f"LD_stats_window_{j}.pkl")

rule build_genetic_map_real:
    input:
        xlsx = COMERON_XLSX,
    output:
        gmap = f"{GENMAP_DIR}/{{chrom}}/genetic_map.txt",
    params:
        script = "snakemake_scripts/build_genetic_map.py",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        python "{params.script}" \
            --xlsx  "{input.xlsx}" \
            --chrom "{wildcards.chrom}" \
            --out   "{output.gmap}"
        """

##############################################################################
# REAL DATA: compute_fim_real – observed FIM at best-LL params for {engine}  #
##############################################################################
rule compute_fim_real:
    input:
        fit = lambda w: f"{REAL_INF_ROOT}/{w.engine}/best_fit.pkl",
        sfs = COMBINED_SFS,
    output:
        fim  = temp(f"{REAL_INF_ROOT}/fim/{{engine}}.fim.npy"),
        summ = f"{REAL_INF_ROOT}/fim/{{engine}}.summary.json",
    params:
        script = "snakemake_scripts/compute_fim.py",
        cfg    = EXP_CFG,
    threads: 2
    shell:
        r"""
        set -euo pipefail
        mkdir -p "$(dirname "{output.fim}")"

        PYTHONPATH={workflow.basedir} \
        python "{params.script}" \
            --engine "{wildcards.engine}" \
            --fit-pkl "{input.fit}" \
            --sfs "{input.sfs}" \
            --config "{params.cfg}" \
            --fim-npy "{output.fim}" \
            --summary-json "{output.summ}"
        """

##############################################################################
# REAL DATA: aggregate_ld_windows_real – LD windows -> means/varcovs (once)   #
# {scope} "" pools every REAL_ARMS window (materialize_real_ld_stats);       #
# "<arm>/" uses that arm's own windows.                                      #
##############################################################################
def _scope_ld_stats(w):
    arm = _scope_arm(w.scope)
    if not arm:
        return real_flat_ld_stats(w)
    return [f"{REAL_LD_ROOT}/{arm}/LD_stats/LD_stats_window_{i}.pkl" for i in _real_arm_window_idxs(arm)]

rule aggregate_ld_windows_real:
    """
    Aggregate per-window LD stats into means/varcovs and write comparison PDF.
    Runs once per scope; the per-opt optimisation rules consume the means.varcovs.pkl output.
    """
    input:
        pkls = _scope_ld_stats,
    output:
        mv   = f"{REAL_LD_ROOT}/{{scope}}means.varcovs.pkl",
        boot = f"{REAL_LD_ROOT}/{{scope}}bootstrap_sets.pkl",
        pdf  = f"{REAL_LD_ROOT}/{{scope}}empirical_vs_theoretical_comparison.pdf",
    params:
        # --run-dir is ignored with --real-data; values kept as before
        run_dir     = lambda w: f"{REAL_INF_ROOT}/{w.scope}".rstrip("/"),
        output_root = lambda w: f"{REAL_LD_ROOT}/{w.scope}".rstrip("/"),
        cfg         = EXP_CFG,
        bins        = R_BINS_STR,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.output_root}"

        PYTHONPATH={workflow.basedir} \
        python "snakemake_scripts/LD_inference.py" \
            --run-dir       "{params.run_dir}" \
            --output-root   "{params.output_root}" \
            --config-file   "{params.cfg}" \
            --r-bins        "{params.bins}" \
            --real-data \
            --skip-optimize
        """


##############################################################################
# REAL DATA: infer_momentsld_real – one LHS start per opt wildcard           #
# {ld_engine} must match {scope}: REAL_LD_ENGINE pooled, REAL_LD_ENGINE_ARM  #
# per arm (_ld_engine).                                                      #
##############################################################################
def _scope_means_varcovs(w):
    _check_ld_engine(w)
    return f"{REAL_LD_ROOT}/{w.scope}means.varcovs.pkl"

rule infer_momentsld_real:
    input:
        mv       = _scope_means_varcovs,
        # The moments best fit of the same scope (pooled or this arm). Needed
        # regardless of momentsld_use_scaled_units: in scaled mode it anchors
        # N_ref; in absolute mode it's still required whenever
        # fixed_parameters pins a param via "moments_best" (e.g. N_ANC) --
        # that lookup reads --sfs-best-fit-pkl independently of N_ref/rho
        # scaling. ancient() so regenerating the moments fit (e.g. changing
        # REAL_TOP_K) doesn't needlessly re-trigger LD inference -- the best
        # (rep_0) moments params are unchanged.
        sfs_best = lambda w: [ancient(f"{REAL_FIT_ROOT}/{w.scope}inferences/moments/best_fit.pkl")] if MOMENTSLD_NEEDS_SFS_FIT else [],
    output:
        # Not temp(): kept per-restart so per-opt N_ANC/T/etc. trends can be
        # inspected without waiting on aggregate_opts_momentsld_real.
        pkl = f"{REAL_FIT_ROOT}/{{scope}}runs/run_{{opt}}/inferences/{{ld_engine}}/best_fit.pkl",
    params:
        outdir = lambda w: f"{REAL_FIT_ROOT}/{w.scope}runs/run_{w.opt}/inferences/{w.ld_engine}",
        cfg    = EXP_CFG,
        bins   = R_BINS_STR,
        seed_flag = _momentsld_seed_flag,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.outdir}"

        PYTHONPATH={workflow.basedir} \
        python "src/MomentsLD_real_data.py" \
            --config           "{params.cfg}" \
            --empirical        "{input.mv}" \
            --outdir           "{params.outdir}" \
            {params.seed_flag} \
            --r-bins           "{params.bins}" \
            --normalization    0 \
            --opt-seed         {wildcards.opt} \
            --verbose

        test -f "{output.pkl}"
        """


##############################################################################
# REAL DATA: aggregate_opts_momentsld_real – pick best across LHS restarts   #
##############################################################################
def _scope_momentsld_runs(w):
    _check_ld_engine(w)
    return [f"{REAL_FIT_ROOT}/{w.scope}runs/run_{o}/inferences/{w.ld_engine}/best_fit.pkl"
            for o in range(NUM_REAL_OPTIMS)]

rule aggregate_opts_momentsld_real:
    input:
        runs = _scope_momentsld_runs,
    output:
        # Model-specific (unlike REAL_LD_ROOT above) -- the fitted MomentsLD
        # result under the active demographic model, next to the moments/dadi
        # best_fit.pkl of the same scope.
        best = f"{REAL_FIT_ROOT}/{{scope}}inferences/{{ld_engine}}/best_fit.pkl",
    run:
        import pickle, pathlib
        from src.aggregate_utils import aggregate_top_k

        arm = _scope_arm(wildcards.scope)
        records = [(p, i) for i, p in enumerate(input.runs)]

        # No theta_hat/N_ANC_implied_from_theta here (unlike moments/dadi real):
        # MomentsLD has no direct access to theta, so there's no absolute-scale
        # implication to carry through.
        best, diag = aggregate_top_k(records, REAL_TOP_K)
        best = {"mode": "momentsLD", **({"arm": arm} if arm else {}), **best}

        pathlib.Path(output.best).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(best, open(output.best, "wb"))

        print(f"✅ [REAL{'/' + arm if arm else ''}] Aggregated {diag['n_entries']} MomentsLD optimization results → {output.best}")


##############################################################################
# REAL DATA: per-arm LD / MomentsLD plots. (The per-arm fits themselves are   #
# the {scope}="<arm>/" case of aggregate_ld_windows_real /                    #
# infer_momentsld_real / aggregate_opts_momentsld_real above.)                #
##############################################################################
# LD decay of the real data alone (same layout as
# empirical_vs_theoretical_comparison.pdf, without a model curve), from the
# same kept windows aggregate_ld_windows_real uses for that arm.
rule plot_ld_decay_real:
    input:
        pkls = lambda w: [
            f"{REAL_LD_ROOT}/{w.arm}/LD_stats/LD_stats_window_{i}.pkl"
            for i in _real_arm_window_idxs(w.arm)
        ],
    output:
        pdf = f"{REAL_LD_ROOT}/{{arm}}/ld_decay.pdf",
    wildcard_constraints:
        arm = r"Chr(2L|2R|3L|3R|X)",
    params:
        ld_dir = lambda w: f"{REAL_LD_ROOT}/{w.arm}/LD_stats",
        bins   = R_BINS_STR,
        seed   = int(CFG.get("seed", 42)),
    threads: 1
    shell:
        r"""
        python snakemake_scripts/plot_ld_decay_real.py \
            --ld-stats-dir "{params.ld_dir}" --r-bins "{params.bins}" --seed {params.seed} --out "{output.pdf}"
        """


# MomentsLD fit vs. data (empirical_vs_theoretical_comparison.pdf layout) for
# one restart's best_fit.pkl or the aggregated one; also prints the
# recomputed log-likelihood next to the stored one as a check.
def _momentsld_fit_plot_shell():
    return r"""
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/plot_momentsld_fit_real.py \
            --empirical "{input.mv}" --best-fit "{input.fit}" --config "{params.cfg}" \
            --r-bins "{params.bins}" --out "{output.pdf}"
        """

rule plot_momentsld_fit_real_run:
    input:
        mv  = f"{REAL_LD_ROOT}/{{arm}}/means.varcovs.pkl",
        fit = f"{REAL_FIT_ROOT}/{{arm}}/runs/run_{{opt}}/inferences/{REAL_LD_ENGINE_ARM}/best_fit.pkl",
    output:
        pdf = f"{REAL_FIT_ROOT}/{{arm}}/runs/run_{{opt}}/inferences/{REAL_LD_ENGINE_ARM}/fit_comparison.pdf",
    wildcard_constraints:
        arm = r"Chr(2L|2R|3L|3R|X)",
    params:
        cfg  = EXP_CFG,
        bins = R_BINS_STR,
    threads: 1
    shell: _momentsld_fit_plot_shell()

rule plot_momentsld_fit_real_best:
    input:
        mv  = f"{REAL_LD_ROOT}/{{arm}}/means.varcovs.pkl",
        fit = f"{REAL_FIT_ROOT}/{{arm}}/inferences/{REAL_LD_ENGINE_ARM}/best_fit.pkl",
    output:
        pdf = f"{REAL_FIT_ROOT}/{{arm}}/inferences/{REAL_LD_ENGINE_ARM}/fit_comparison.pdf",
    wildcard_constraints:
        arm = r"Chr(2L|2R|3L|3R|X)",
    params:
        cfg  = EXP_CFG,
        bins = R_BINS_STR,
    threads: 1
    shell: _momentsld_fit_plot_shell()

##############################################################################
# REAL DATA: sfs_residuals_real – best-fit model SFS − observed SFS          #
##############################################################################
rule sfs_residuals_real:
    input:
        obs_sfs = COMBINED_SFS,
        # (not strictly needed by the script, but keeps DAG honest)
        agg_fit = lambda w: f"{REAL_INF_ROOT}/{w.engine}/best_fit.pkl",
    output:
        res_arr   = f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/residuals.npy",
        res_flat  = temp(f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/residuals_flat.npy"),
        meta_json = temp(f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/meta.json"),
        hist_png  = f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/residuals_histogram.png",
        **gs_outputs(f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}"),
    params:
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL not in ["drosophila_three_epoch", "OOA_three_pop_Gutenkunst"]
            else f"src.simulation:{MODEL}"
        ),
        # for real data: inference-dir is the REAL_INF_ROOT, not sim_{sid}
        inf_dir  = REAL_INF_ROOT,
        out_dir  = lambda w: f"{REAL_INF_ROOT}/sfs_residuals/{w.engine}",
        n_bins   = CFG.get("sfs_n_bins", ""),  # empty string if not specified
        script   = "bash_scripts/lib/run_sfs_residuals.sh",
    threads: 1
    shell:
        r"""
        bash "{params.script}" \
            "{wildcards.engine}" "{params.cfg}" "{params.model_py}" "{input.obs_sfs}" \
            "{params.inf_dir}" "{params.out_dir}" "{USE_GS}" "{RESID_SCRIPT}" \
            "{workflow.basedir}" "{params.n_bins}"
        """


##############################################################################
# REAL DATA: combine_results_real – merge fits + attach FIM + residuals      #
##############################################################################
rule combine_results_real:
    input:
        cfg     = EXP_CFG,

        moments   = f"{REAL_INF_ROOT}/moments/best_fit.pkl",
        dadi      = f"{REAL_INF_ROOT}/dadi/best_fit.pkl",
        momentsLD = f"{REAL_INF_ROOT}/{REAL_LD_ENGINE}/best_fit.pkl",

        # FIMs (upper-tri flattened) for whatever engines you computed
        fims = lambda w: [
            f"{REAL_INF_ROOT}/fim/{eng}.fim.npy"
            for eng in FIM_ENGINES
        ],

        # residual vectors for whichever engines you want to include
        resid_vecs = lambda w: [
            f"{REAL_INF_ROOT}/sfs_residuals/{eng}/{_resid_vector_fname()}"
            for eng in RESIDUAL_ENGINES
        ],
        resid_meta = lambda w: [
            f"{REAL_INF_ROOT}/sfs_residuals/{eng}/meta.json"
            for eng in RESIDUAL_ENGINES
        ],
    output:
        combo = f"{REAL_INF_ROOT}/all_inferences.pkl",
    run:
        import pickle, pathlib
        from src.combine_payloads import build_fim_payload, build_residual_payload

        outdir = pathlib.Path(output.combo).parent
        outdir.mkdir(parents=True, exist_ok=True)

        summary = {}
        summary["moments"]   = pickle.load(open(input.moments, "rb"))
        summary["dadi"]      = pickle.load(open(input.dadi, "rb"))
        summary["momentsLD"] = pickle.load(open(input.momentsLD, "rb"))

        fim_payload = build_fim_payload(input.fims)
        if fim_payload:
            summary["FIM"] = fim_payload

        resid_payload = build_residual_payload(input.resid_vecs)
        if resid_payload:
            summary["SFS_residuals"] = resid_payload

        pickle.dump(summary, open(output.combo, "wb"))
        print(f"✓ combined REAL → {output.combo}")

##############################################################################
# REAL DATA: build_real_prediction_dataset                                   #
# Assemble the real dadi / moments / MomentsLD fits -- plus the same FIM /    #
# SFS-residual payloads combine_results_real attaches for sims -- into a      #
# single feature row formatted exactly like the training features_df, then    #
# z-score normalize by the priors so it can be pushed through a trained       #
# model. One dataset per {variant} (matches MODELING_VARIANTS) since features #
# differ by whether FIM / SFS-residual columns are included.                 #
#                                                                             #
#   snakemake experiments/<MODEL>/real_trimmed/prediction_<variant>/real_features_df.pkl
##############################################################################
rule build_real_prediction_dataset:
    input:
        cfg            = EXP_CFG,
        moments        = f"{REAL_INF_ROOT}/moments/best_fit.pkl",
        dadi           = f"{REAL_INF_ROOT}/dadi/best_fit.pkl",
        ld             = f"{REAL_INF_ROOT}/{REAL_LD_ENGINE}/best_fit.pkl",
        train_features = lambda w: _real_train_features(w.variant),
        empirical_norm_stats = lambda w: _real_empirical_norm_stats(w.variant),
        fims = lambda w: [
            f"{REAL_INF_ROOT}/fim/{eng}.fim.npy"
            for eng in FIM_ENGINES
        ],
        resid_vecs = lambda w: [
            f"{REAL_INF_ROOT}/sfs_residuals/{eng}/{_resid_vector_fname()}"
            for eng in RESIDUAL_ENGINES
        ],
        resid_meta = lambda w: [
            f"{REAL_INF_ROOT}/sfs_residuals/{eng}/meta.json"
            for eng in RESIDUAL_ENGINES
        ],
    output:
        feats = f"{REAL_PRED_ROOT}/real_features_df.pkl",
        raw   = f"{REAL_PRED_ROOT}/real_features_raw_df.pkl",
        meta  = f"{REAL_PRED_ROOT}/real_dataset_meta.json",
    params:
        real_inf_dir = REAL_INF_ROOT,
        # NOT the REAL_PRED_ROOT constant -- that still holds the literal,
        # unsubstituted "{variant}" text; params (unlike input/output) don't
        # get auto-filled from wildcards, so this must build the real path
        # from w.variant directly.
        out_dir      = lambda w: f"{REAL_FIT_ROOT}/prediction_{w.variant}",
        ld_engine_subdir = REAL_LD_ENGINE,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/build_real_prediction_dataset.py \
            --config         "{input.cfg}" \
            --real-inf-dir   "{params.real_inf_dir}" \
            --ld-engine-subdir "{params.ld_engine_subdir}" \
            --train-features "{input.train_features}" \
            --empirical-norm-stats "{input.empirical_norm_stats}" \
            --out-dir        "{params.out_dir}" \
            --fim-paths      {input.fims} \
            --resid-vec-paths {input.resid_vecs} \
            --allow-missing
        """

##############################################################################
# REAL DATA: predict_real_data – push real features through a trained model  #
# {model_key} ∈ random_forest | xgboost | linear_standard | linear_ridge |    #
#              linear_lasso | linear_elasticnet                               #
# {variant} selects which modeling_{variant}-trained model to use (matches   #
# MODELING_VARIANTS) -- outputs land under the matching prediction_{variant}/ #
#                                                                             #
#   snakemake experiments/<MODEL>/real_trimmed/prediction_<variant>/predictions_random_forest.json
##############################################################################
rule predict_real_data:
    input:
        feats          = f"{REAL_PRED_ROOT}/real_features_df.pkl",
        model          = lambda w: _real_model_objs(w.variant)[w.model_key],
        train_features = lambda w: _real_train_features(w.variant),
        cfg            = EXP_CFG,
    output:
        json = f"{REAL_PRED_ROOT}/predictions_{{model_key}}.json",
        csv  = f"{REAL_PRED_ROOT}/predictions_{{model_key}}.csv",
    params:
        out_prefix = lambda w: f"{REAL_FIT_ROOT}/prediction_{w.variant}/predictions_{w.model_key}",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/predict_real_data.py \
            --model-obj      "{input.model}" \
            --real-features  "{input.feats}" \
            --train-features "{input.train_features}" \
            --config         "{input.cfg}" \
            --out-prefix     "{params.out_prefix}" \
            --model-key      "{wildcards.model_key}"
        """

##############################################################################
# REAL DATA: shap_real_data -- explain a trained model's real-data prediction #
# with SHAP, one explainer per target parameter. {variant}/{model_key} match  #
# predict_real_data above; background = that variant's normalized training   #
# features (normalized_train_features.pkl), same space real_features_df.pkl  #
# is normalized into.                                                        #
#                                                                             #
#   snakemake experiments/<MODEL>/real_trimmed/prediction_<variant>/shap_summary_random_forest.json
##############################################################################
rule shap_real_data:
    input:
        feats              = f"{REAL_PRED_ROOT}/real_features_df.pkl",
        background_features = lambda w: _real_norm_train_features(w.variant),
        model              = lambda w: _real_model_objs(w.variant)[w.model_key],
        train_features     = lambda w: _real_train_features(w.variant),
    output:
        json = f"{REAL_PRED_ROOT}/shap_summary_{{model_key}}.json",
        pkl  = f"{REAL_PRED_ROOT}/shap_values_{{model_key}}.pkl",
        csv  = f"{REAL_PRED_ROOT}/shap_values_{{model_key}}.csv",
        png  = f"{REAL_PRED_ROOT}/shap_summary_{{model_key}}.png",
    params:
        out_dir = lambda w: f"{REAL_FIT_ROOT}/prediction_{w.variant}",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/shap_real_data.py \
            --model-obj           "{input.model}" \
            --real-features       "{input.feats}" \
            --background-features "{input.background_features}" \
            --train-features      "{input.train_features}" \
            --out-dir             "{params.out_dir}" \
            --model-key           "{wildcards.model_key}"
        """

##############################################################################
# REAL DATA: calibration_simulate – model calibration / posterior-predictive #
# check. Simulate at the real-data fitted params (predictions_{model_key}    #
# from predict_real_data), one job per {variant}/{model_key}/{rep} so        #
# replicates run as parallel SLURM array tasks                              #
# (bash_scripts/simulation/calibration_simulate.sh) instead of a sequential  #
# loop in one job. Saves the tree sequence + SFS per replicate for           #
# comparison against the real observed SFS.                                 #
#                                                                             #
#   snakemake experiments/<MODEL>/real_trimmed/calibration_<variant>/<model_key>/replicate_<rep>/SFS.pkl
##############################################################################
rule calibration_simulate:
    input:
        cfg         = EXP_CFG,
        predictions = lambda w: f"{REAL_FIT_ROOT}/prediction_{w.variant}/predictions_{w.model_key}.json",
    output:
        trees  = f"{CALIBRATION_ROOT}/tree_sequence.trees",
        sfs    = f"{CALIBRATION_ROOT}/SFS.pkl",
        meta   = f"{CALIBRATION_ROOT}/meta.json",
    params:
        out_dir = lambda w: f"{REAL_FIT_ROOT}/calibration_{w.variant}/{w.model_key}",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/calibration_simulate.py \
            --config               "{input.cfg}" \
            --params-json          "{input.predictions}" \
            --out-dir              "{params.out_dir}" \
            --n-replicates         1 \
            --start-replicate-index {wildcards.rep}
        """

# Convenience target: all calibration_n_replicates replicates for one
# {variant}/{model_key} (reads the count from the config's calibration_n_replicates
# key). bash_scripts/simulation/calibration_simulate.sh reads the same key itself to size its
# SLURM --array range -- this rule is for driving the same count through plain
# snakemake (e.g. local runs) without SLURM.
rule calibration_simulate_all_reps:
    input:
        lambda w: expand(
            f"{REAL_FIT_ROOT}/calibration_{w.variant}/{w.model_key}/replicate_{{rep}}/SFS.pkl",
            rep=CALIBRATION_REPS,
        )
    output:
        touch(f"{REAL_FIT_ROOT}/calibration_{{variant}}/{{model_key}}/.all_reps_done")

##############################################################################
# REAL DATA: calibration_ppc – model calibration / posterior-predictive check #
# plots + summary. Compares SFS-derived pi/Tajima's D/FST and SFS shape       #
# between the real observed COMBINED_SFS and all calibration_n_replicates     #
# calibration_simulate replicates for one {variant}/{model_key}. Requires all #
# replicates (calibration_simulate) to already exist.                        #
#                                                                             #
#   snakemake experiments/<MODEL>/real_trimmed/calibration_<variant>/<model_key>/ppc/calibration_ppc.png
##############################################################################
rule calibration_ppc:
    input:
        cfg  = EXP_CFG,
        sfs  = COMBINED_SFS,
        meta = COMBINED_SFS_META,
        reps = lambda w: expand(
            f"{REAL_FIT_ROOT}/calibration_{w.variant}/{w.model_key}/replicate_{{rep}}/SFS.pkl",
            rep=CALIBRATION_REPS,
        ),
    output:
        png     = f"{REAL_FIT_ROOT}/calibration_{{variant}}/{{model_key}}/ppc/calibration_ppc.png",
        summary = f"{REAL_FIT_ROOT}/calibration_{{variant}}/{{model_key}}/ppc/calibration_ppc_summary.json",
    params:
        calibration_dir = lambda w: f"{REAL_FIT_ROOT}/calibration_{w.variant}/{w.model_key}",
        out_dir         = lambda w: f"{REAL_FIT_ROOT}/calibration_{w.variant}/{w.model_key}/ppc",
        title           = lambda w: f"{MODEL} ({w.variant}/{w.model_key})",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/calibration_ppc.py \
            --config             "{input.cfg}" \
            --calibration-dir    "{params.calibration_dir}" \
            --combined-sfs       "{input.sfs}" \
            --combined-sfs-meta  "{input.meta}" \
            --out-dir            "{params.out_dir}" \
            --title              "{params.title}"
        """

##############################################################################
# REAL DATA: calibration_ld_ppc – model calibration / posterior-predictive   #
# check, LD version. Does the theoretical LD decay curve at the fitted       #
# params (predictions_{model_key} from predict_real_data) reproduce the      #
# empirical LD decay measured directly from the real data? Purely analytic   #
# (moments.Demes.LD at the fitted params vs REAL_LD_ROOT/means.varcovs.pkl)  #
# -- no calibration_simulate replicates needed, so this doesn't depend on    #
# the (SLURM-array) tree-sequence simulations at all.                        #
#                                                                             #
#   snakemake experiments/<MODEL>/real_trimmed/calibration_<variant>/<model_key>/ppc/calibration_ld_ppc.pdf
##############################################################################
rule calibration_ld_ppc:
    input:
        cfg         = EXP_CFG,
        real_ld     = f"{REAL_LD_ROOT}/means.varcovs.pkl",
        predictions = lambda w: f"{REAL_FIT_ROOT}/prediction_{w.variant}/predictions_{w.model_key}.json",
    output:
        pdf = f"{REAL_FIT_ROOT}/calibration_{{variant}}/{{model_key}}/ppc/calibration_ld_ppc.pdf",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/calibration_ld_ppc.py \
            --config        "{input.cfg}" \
            --real-ld-pkl   "{input.real_ld}" \
            --params-json   "{input.predictions}" \
            --out-path      "{output.pdf}"
        """

##############################################################################
# REAL DATA: SFS calibration check for ONE per-arm moments/dadi run          #
# Simulate the arm under that run's fitted params (sfs_fit.json), with the   #
# config's sample_ploidy and recombination map (same as every simulation),  #
# the observed SFS's sample sizes, and the mutation rate scaled by the fraction #
# of sites the real data kept -- then compare against that arm's observed    #
# SFS with calibration_ppc.py. One calibration_simulate_fit job per          #
# replicate (SLURM array: bash_scripts/real_data/real_calibration_simulate.sh).
#   snakemake ".../real_trimmed_excl-FR217-FR361/Chr3L/calibration/moments_run11/ppc/calibration_ppc.png"
##############################################################################
CALIB_FIT_ROOT = f"{REAL_FIT_ROOT}/{{chrom}}/calibration/{{engine}}_run{{opt}}"

def _calibration_map_matches(w):
    """The recombination map is per-arm -- refuse to simulate one arm with
    another arm's map."""
    rec = CFG.get("recombination") or {}
    if rec.get("type") == "map" and f"/{w.chrom}/" not in rec["file"]:
        raise ValueError(f"recombination.file {rec['file']} is not {w.chrom}'s map")
    return EXP_CFG

rule calibration_simulate_fit:
    input:
        cfg  = _calibration_map_matches,
        fit  = f"{REAL_RUN_ROOT_CHROM}/run_{{opt}}/inferences/{{engine}}/sfs_fit.json",
        sfs  = lambda w: per_chrom_sfs(w.chrom),
        meta = lambda w: per_chrom_sfs_meta(w.chrom),
    output:
        sfs  = f"{CALIB_FIT_ROOT}/replicate_{{rep}}/SFS.pkl",
        meta = f"{CALIB_FIT_ROOT}/replicate_{{rep}}/meta.json",
    params:
        out_dir = lambda w: CALIB_FIT_ROOT.format(chrom=w.chrom, engine=w.engine, opt=w.opt),
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/calibration_simulate.py \
            --config                "{input.cfg}" \
            --params-json           "{input.fit}" \
            --observed-sfs          "{input.sfs}" \
            --sfs-meta              "{input.meta}" \
            --out-dir               "{params.out_dir}" \
            --n-replicates          1 \
            --start-replicate-index {wildcards.rep} \
            --no-trees
        """

rule calibration_ppc_fit:
    input:
        cfg  = EXP_CFG,
        sfs  = lambda w: per_chrom_sfs(w.chrom),
        meta = lambda w: per_chrom_sfs_meta(w.chrom),
        reps = lambda w: expand(
            f"{CALIB_FIT_ROOT.format(chrom=w.chrom, engine=w.engine, opt=w.opt)}/replicate_{{rep}}/SFS.pkl",
            rep=CALIBRATION_REPS,
        ),
    output:
        png     = f"{CALIB_FIT_ROOT}/ppc/calibration_ppc.png",
        summary = f"{CALIB_FIT_ROOT}/ppc/calibration_ppc_summary.json",
    params:
        calibration_dir = lambda w: CALIB_FIT_ROOT.format(chrom=w.chrom, engine=w.engine, opt=w.opt),
        title           = lambda w: f"{MODEL} {w.chrom} ({w.engine} run {w.opt})",
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python snakemake_scripts/calibration_ppc.py \
            --config             "{input.cfg}" \
            --calibration-dir    "{params.calibration_dir}" \
            --combined-sfs       "{input.sfs}" \
            --combined-sfs-meta  "{input.meta}" \
            --out-dir            "$(dirname "{output.png}")" \
            --title              "{params.title}"
        """

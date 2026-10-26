##############################################################################
# Real data: VCF prep -> SFS/LD inference -> prediction with trained models
##############################################################################


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
# REAL DATA – NLopt Poisson SFS optimisation (moments / dadi)
##############################################################################
rule infer_engine_real:
    input:
        sfs  = COMBINED_SFS,
        meta = COMBINED_SFS_META,
    output:
        # TEMPORARY: temp() removed so per-replicate best_fit.pkl files stick
        # around to inspect individual optimizations after the trim-region
        # comparison run. Restore temp(...) once done inspecting.
        pkl = f"{REAL_RUN_ROOT}/run_{{opt}}/inferences/{{engine}}/best_fit.pkl",
        png = f"{REAL_RUN_ROOT}/run_{{opt}}/inferences/{{engine}}/sfs_fit.png",
        json = f"{REAL_RUN_ROOT}/run_{{opt}}/inferences/{{engine}}/sfs_fit.json",
    params:
        run_dir  = lambda w: f"{REAL_RUN_ROOT}/run_{w.opt}",
        label    = lambda w: f"autosomes, run {w.opt}",
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

# ── REAL DATA: MOMENTS / DADI ───────────────────────────────────────────────
rule aggregate_opts_engine_real:
    input:
        runs = lambda w: [f"{REAL_RUN_ROOT}/run_{o}/inferences/{w.engine}/best_fit.pkl"
                          for o in range(NUM_REAL_OPTIMS)],
        sfs  = COMBINED_SFS,
    output:
        pkl = f"{REAL_INF_ROOT}/{{engine}}/best_fit.pkl",
        png = f"{REAL_INF_ROOT}/{{engine}}/sfs_fit.png",
        json = f"{REAL_INF_ROOT}/{{engine}}/sfs_fit.json",
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

        records = [(p, i) for i, p in enumerate(input.runs)]

        best, diag = aggregate_top_k(
            records, REAL_TOP_K,
            extra_fields=("theta_hat", "N_ANC_implied_from_theta"),
        )
        best = {"mode": wildcards.engine, **best}

        pathlib.Path(output.pkl).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(best, open(output.pkl, "wb"))

        print(f"✅ [REAL] Aggregated {diag['n_entries']} {wildcards.engine} optimization results → {output.pkl}")

        # observed vs. best-fit SFS for the top run (+ its numbers in sfs_fit.json)
        shell(
            'PYTHONPATH={basedir} python snakemake_scripts/plot_sfs_fit_real.py '
            '--fit-pkl "{output.pkl}" --sfs "{input.sfs}" --config "{params.cfg}" '
            '--model-py "{params.model_py}" --label "{label}" --out "{output.png}"',
            basedir=workflow.basedir, label="autosomes, best of all runs",
        )


##############################################################################
# REAL DATA (single chromosome) – NLopt Poisson SFS optimisation             #
# (moments / dadi), fit against one chromosome's SFS instead of the         #
# combined-autosome COMBINED_SFS used by infer_engine_real above.           #
##############################################################################
rule infer_engine_real_chrom:
    input:
        sfs  = lambda w: per_chrom_sfs(w.chrom),
        meta = lambda w: per_chrom_sfs_meta(w.chrom),
    output:
        # kept (not temp) so each restart can be inspected after aggregation,
        # and so real_sfs_inference.sh's skip-if-exists check stays valid
        pkl = f"{REAL_RUN_ROOT_CHROM}/run_{{opt}}/inferences/{{engine}}/best_fit.pkl",
        png = f"{REAL_RUN_ROOT_CHROM}/run_{{opt}}/inferences/{{engine}}/sfs_fit.png",
        json = f"{REAL_RUN_ROOT_CHROM}/run_{{opt}}/inferences/{{engine}}/sfs_fit.json",
    params:
        run_dir  = lambda w: f"{REAL_RUN_ROOT_CHROM.format(chrom=w.chrom)}/run_{w.opt}",
        label    = lambda w: f"{w.chrom}, run {w.opt}",
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

rule aggregate_opts_engine_real_chrom:
    input:
        runs = lambda w: [
            f"{REAL_RUN_ROOT_CHROM.format(chrom=w.chrom)}/run_{o}/inferences/{w.engine}/best_fit.pkl"
            for o in range(NUM_REAL_OPTIMS)
        ],
        sfs  = lambda w: per_chrom_sfs(w.chrom),
    output:
        pkl = f"{REAL_INF_ROOT_CHROM}/{{engine}}/best_fit.pkl",
        png = f"{REAL_INF_ROOT_CHROM}/{{engine}}/sfs_fit.png",
        json = f"{REAL_INF_ROOT_CHROM}/{{engine}}/sfs_fit.json",
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

        records = [(p, i) for i, p in enumerate(input.runs)]

        best, diag = aggregate_top_k(
            records, REAL_TOP_K,
            extra_fields=("theta_hat", "N_ANC_implied_from_theta"),
        )
        best = {"mode": wildcards.engine, "chrom": wildcards.chrom, **best}

        pathlib.Path(output.pkl).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(best, open(output.pkl, "wb"))

        print(f"✅ [REAL/{wildcards.chrom}] Aggregated {diag['n_entries']} {wildcards.engine} optimization results → {output.pkl}")

        # observed vs. best-fit SFS for the top run (+ its numbers in sfs_fit.json)
        shell(
            'PYTHONPATH={basedir} python snakemake_scripts/plot_sfs_fit_real.py '
            '--fit-pkl "{output.pkl}" --sfs "{input.sfs}" --config "{params.cfg}" '
            '--model-py "{params.model_py}" --label "{label}" --out "{output.png}"',
            basedir=workflow.basedir, label=f"{wildcards.chrom}, best of all runs",
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
# REAL DATA: aggregate_ld_windows_real – aggregate LD windows (once)         #
##############################################################################
rule aggregate_ld_windows_real:
    """
    Aggregate per-window LD stats into means/varcovs and write comparison PDF.
    Runs once; the per-opt optimisation rules consume the means.varcovs.pkl output.
    """
    input:
        pkls = real_flat_ld_stats,
    output:
        mv   = f"{REAL_LD_ROOT}/means.varcovs.pkl",
        boot = f"{REAL_LD_ROOT}/bootstrap_sets.pkl",
        pdf  = f"{REAL_LD_ROOT}/empirical_vs_theoretical_comparison.pdf",
    params:
        run_dir     = REAL_INF_ROOT,
        output_root = REAL_LD_ROOT,
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
##############################################################################
rule infer_momentsld_real:
    input:
        mv       = f"{REAL_LD_ROOT}/means.varcovs.pkl",
        # Always sourced from the moments best-fit. Needed regardless of
        # momentsld_use_scaled_units: in scaled mode it anchors N_ref; in
        # absolute mode it's still required whenever fixed_parameters pins a
        # param via "moments_best" (e.g. N_ANC) -- that lookup reads
        # --sfs-best-fit-pkl independently of N_ref/rho scaling, which in
        # absolute mode tracks the optimizer's own current N_ANC guess
        # instead. Marked ancient() so regenerating the moments fit (e.g.
        # changing REAL_TOP_K) does not needlessly re-trigger LD inference --
        # the best (rep_0) moments params are unchanged.
        sfs_best = [ancient(f"{REAL_INF_ROOT}/moments/best_fit.pkl")] if MOMENTSLD_NEEDS_SFS_FIT else [],
    output:
        # Not temp(): kept per-restart so per-opt N_ANC/T/etc. trends can be
        # inspected directly (as we've been doing) without waiting on
        # aggregate_opts_momentsld_real to finish.
        pkl = f"{REAL_RUN_ROOT}/run_{{opt}}/inferences/{REAL_LD_ENGINE}/best_fit.pkl",
    params:
        outdir = lambda w: f"{REAL_RUN_ROOT}/run_{w.opt}/inferences/{REAL_LD_ENGINE}",
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
rule aggregate_opts_momentsld_real:
    input:
        runs = [f"{REAL_RUN_ROOT}/run_{o}/inferences/{REAL_LD_ENGINE}/best_fit.pkl"
                for o in range(NUM_REAL_OPTIMS)],
    output:
        # Model-specific (unlike REAL_LD_ROOT above) -- this is the fitted
        # MomentsLD result under the active demographic model, so it belongs
        # under REAL_INF_ROOT alongside the moments/dadi best_fit.pkl.
        best = f"{REAL_INF_ROOT}/{REAL_LD_ENGINE}/best_fit.pkl",
    run:
        import pickle, pathlib
        from src.aggregate_utils import aggregate_top_k

        records = [(p, i) for i, p in enumerate(input.runs)]

        # No theta_hat/N_ANC_implied_from_theta here (unlike moments/dadi real):
        # MomentsLD has no direct access to theta, so there's no absolute-scale
        # implication to carry through.
        best, diag = aggregate_top_k(records, REAL_TOP_K)
        best = {"mode": "momentsLD", **best}

        pathlib.Path(output.best).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(best, open(output.best, "wb"))

        print(f"✅ [REAL] Aggregated {diag['n_entries']} MomentsLD optimization results → {output.best}")


##############################################################################
# REAL DATA (pooling_mode="individual"): per-arm MomentsLD, no cross-arm      #
# pooling. Each arm's own windows/LD_stats already live separately under     #
# {REAL_LD_ROOT}/{arm}/LD_stats/ (compute_ld_real, arm-count-agnostic) --     #
# these three rules add an arm-scoped means/varcovs+bootstrap and an         #
# arm-scoped MomentsLD optimization on top of that, so nothing is combined   #
# across arms anywhere in this path. Mirrors aggregate_ld_windows_real/      #
# infer_momentsld_real/aggregate_opts_momentsld_real above, but with an      #
# {arm} wildcard instead of REAL_TAG-pooling all of REAL_ARMS together.      #
##############################################################################
# LD decay of the real data alone (same layout as
# empirical_vs_theoretical_comparison.pdf, without a model curve), from the
# same kept windows aggregate_ld_windows_real_by_arm uses.
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


rule aggregate_ld_windows_real_by_arm:
    input:
        pkls = lambda w: [
            f"{REAL_LD_ROOT}/{w.arm}/LD_stats/LD_stats_window_{i}.pkl"
            for i in _real_arm_window_idxs(w.arm)
        ],
    output:
        mv   = f"{REAL_LD_ROOT}/{{arm}}/means.varcovs.pkl",
        boot = f"{REAL_LD_ROOT}/{{arm}}/bootstrap_sets.pkl",
        pdf  = f"{REAL_LD_ROOT}/{{arm}}/empirical_vs_theoretical_comparison.pdf",
    wildcard_constraints:
        arm = r"Chr(2L|2R|3L|3R|X)",
    params:
        run_dir     = lambda w: f"{REAL_INF_ROOT}/{w.arm}",
        output_root = lambda w: f"{REAL_LD_ROOT}/{w.arm}",
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


rule infer_momentsld_real_by_arm:
    input:
        mv       = f"{REAL_LD_ROOT}/{{arm}}/means.varcovs.pkl",
        # Always sourced from this SAME arm's own (unpooled) moments fit --
        # see infer_engine_real_chrom/aggregate_opts_engine_real_chrom, which
        # fit per_chrom_sfs(arm) directly rather than the pooled COMBINED_SFS.
        # Needed regardless of momentsld_use_scaled_units: in scaled mode it
        # anchors N_ref; in absolute mode it's still required whenever
        # fixed_parameters pins a param via "moments_best" (e.g. N_ANC) --
        # that lookup reads --sfs-best-fit-pkl independently of N_ref/rho
        # scaling, which in absolute mode tracks the optimizer's own current
        # N_ANC guess instead (see infer_momentsld_real above).
        sfs_best = lambda w: [ancient(f"{REAL_FIT_ROOT}/{w.arm}/inferences/moments/best_fit.pkl")] if MOMENTSLD_NEEDS_SFS_FIT else [],
    output:
        pkl = f"{REAL_FIT_ROOT}/{{arm}}/runs/run_{{opt}}/inferences/{REAL_LD_ENGINE_ARM}/best_fit.pkl",
    wildcard_constraints:
        arm = r"Chr(2L|2R|3L|3R|X)",
    params:
        outdir = lambda w: f"{REAL_FIT_ROOT}/{w.arm}/runs/run_{w.opt}/inferences/{REAL_LD_ENGINE_ARM}",
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


rule aggregate_opts_momentsld_real_by_arm:
    input:
        runs = lambda w: [
            f"{REAL_FIT_ROOT}/{w.arm}/runs/run_{o}/inferences/{REAL_LD_ENGINE_ARM}/best_fit.pkl"
            for o in range(NUM_REAL_OPTIMS)
        ],
    output:
        best = f"{REAL_FIT_ROOT}/{{arm}}/inferences/{REAL_LD_ENGINE_ARM}/best_fit.pkl",
    wildcard_constraints:
        arm = r"Chr(2L|2R|3L|3R|X)",
    run:
        import pickle, pathlib
        from src.aggregate_utils import aggregate_top_k

        records = [(p, i) for i, p in enumerate(input.runs)]
        best, diag = aggregate_top_k(records, REAL_TOP_K)
        best = {"mode": "momentsLD", "arm": wildcards.arm, **best}

        pathlib.Path(output.best).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(best, open(output.best, "wb"))

        print(f"✅ [REAL/{wildcards.arm}] Aggregated {diag['n_entries']} MomentsLD optimization results → {output.best}")


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

rule all_momentsld_fit_plots_real:
    """Convenience target: fit-vs-data plot for every restart and the aggregated best, every arm."""
    input:
        expand(f"{REAL_FIT_ROOT}/{{arm}}/runs/run_{{opt}}/inferences/{REAL_LD_ENGINE_ARM}/fit_comparison.pdf",
               arm=REAL_ARMS, opt=range(NUM_REAL_OPTIMS)),
        expand(f"{REAL_FIT_ROOT}/{{arm}}/inferences/{REAL_LD_ENGINE_ARM}/fit_comparison.pdf", arm=REAL_ARMS),


##############################################################################
# REAL DATA: MomentsLD profile likelihoods (per arm) -- opt-in               #
# Only defined when real_data_analysis.profile_likelihood is in the config:  #
#   "profile_likelihood": {                                                 #
#     "grids":    {"T": [5e4, 1e6, 13], "N_CO0": [1e5, 3e7, 13]},           #
#     "n_starts": 8                                                         #
#   }                                                                       #
# Each grid is [min, max, n_points], log-spaced. Pin one parameter at each  #
# grid value and re-optimize every other one from n_starts starts (start 0  #
# from the unrestricted best fit, the rest LHS draws); the collect rule     #
# keeps the best start per grid value. Grids may extend past the prior      #
# (e.g. to see whether a railed N_CO0 keeps improving beyond its bound).    #
# Target: all_momentsld_profiles_real, or e.g.                              #
#   {REAL_FIT_ROOT}/Chr3L/profiles/{REAL_LD_ENGINE_ARM}/T/profile_T.png     #
##############################################################################
import numpy as _np
_PROFILE_CFG = REAL_DATA_CFG.get("profile_likelihood") or {}
PROFILE_GRIDS = {
    p: list(_np.logspace(_np.log10(float(lo)), _np.log10(float(hi)), int(n)))
    for p, (lo, hi, n) in _PROFILE_CFG.get("grids", {}).items()
}
PROFILE_N_STARTS = int(_PROFILE_CFG.get("n_starts", 8))

if PROFILE_GRIDS:
    def _profile_root(w):
        return f"{REAL_FIT_ROOT}/{w.arm}/profiles/{REAL_LD_ENGINE_ARM}/{w.pparam}"

    rule momentsld_profile_point_real_by_arm:
        input:
            mv   = f"{REAL_LD_ROOT}/{{arm}}/means.varcovs.pkl",
            best = f"{REAL_FIT_ROOT}/{{arm}}/inferences/{REAL_LD_ENGINE_ARM}/best_fit.pkl",
        output:
            pkl = f"{REAL_FIT_ROOT}/{{arm}}/profiles/{REAL_LD_ENGINE_ARM}/{{pparam}}/pt{{k}}/start{{s}}/best_fit.pkl",
        wildcard_constraints:
            arm    = r"Chr(2L|2R|3L|3R|X)",
            pparam = "|".join(PROFILE_GRIDS),
            k      = r"\d+",
            s      = r"\d+",
        params:
            outdir = lambda w: f"{_profile_root(w)}/pt{w.k}/start{w.s}",
            cfg    = EXP_CFG,
            bins   = R_BINS_STR,
            fix    = lambda w: f"{w.pparam}={PROFILE_GRIDS[w.pparam][int(w.k)]:.6g}",
            start  = lambda w, input: (f'--x0-pkl "{input.best}"' if int(w.s) == 0
                                       else f"--opt-seed {w.s}"),
            seed_flag = "--n-ref 1.0" if not MOMENTSLD_NEEDS_SFS_FIT else "",
        threads: 1
        shell:
            r"""
            set -euo pipefail
            mkdir -p "{params.outdir}"
            PYTHONPATH={workflow.basedir} \
            python "src/MomentsLD_real_data.py" \
                --config        "{params.cfg}" \
                --empirical     "{input.mv}" \
                --outdir        "{params.outdir}" \
                {params.seed_flag} \
                --r-bins        "{params.bins}" \
                --normalization 0 \
                --fix           "{params.fix}" \
                {params.start}
            test -f "{output.pkl}"
            """

    rule momentsld_profile_real_by_arm:
        input:
            fits = lambda w: [f"{_profile_root(w)}/pt{k}/start{s}/best_fit.pkl"
                              for k in range(len(PROFILE_GRIDS[w.pparam])) for s in range(PROFILE_N_STARTS)],
            best = f"{REAL_FIT_ROOT}/{{arm}}/inferences/{REAL_LD_ENGINE_ARM}/best_fit.pkl",
        output:
            tsv = f"{REAL_FIT_ROOT}/{{arm}}/profiles/{REAL_LD_ENGINE_ARM}/{{pparam}}/profile_{{pparam}}.tsv",
            png = f"{REAL_FIT_ROOT}/{{arm}}/profiles/{REAL_LD_ENGINE_ARM}/{{pparam}}/profile_{{pparam}}.png",
        wildcard_constraints:
            arm    = r"Chr(2L|2R|3L|3R|X)",
            pparam = "|".join(PROFILE_GRIDS),
        params:
            out   = lambda w: f"{_profile_root(w)}/profile_{w.pparam}",
            cfg   = EXP_CFG,
            title = lambda w: f"{w.arm} {REAL_LD_ENGINE_ARM}",
        threads: 1
        shell:
            r"""
            set -euo pipefail
            python snakemake_scripts/plot_momentsld_profile.py \
                --param {wildcards.pparam} --fits {input.fits} --overall-best "{input.best}" \
                --config "{params.cfg}" --title "{params.title}" --out "{params.out}"
            """

    rule all_momentsld_profiles_real:
        """Convenience target: every configured profile, for every configured arm."""
        input:
            expand(
                f"{REAL_FIT_ROOT}/{{arm}}/profiles/{REAL_LD_ENGINE_ARM}/{{pparam}}/profile_{{pparam}}.png",
                arm=REAL_ARMS, pparam=list(PROFILE_GRIDS),
            ),


##############################################################################
# REAL DATA: moments SFS profile likelihoods (per arm) -- opt-in             #
# Only defined when real_data_analysis.sfs_profile_likelihood is set:        #
#   "sfs_profile_likelihood": {                                             #
#     "grids":    {"N_FR0": [0.002, 5, 13]},                                #
#     "n_starts": 8                                                         #
#   }                                                                       #
# Same recipe as the MomentsLD profiles above, but for the per-arm moments   #
# SFS fit -- whose free parameters are SCALED (sizes are N_*/N_ANC, T is     #
# T/(2 N_ANC), migration is 2 N_ANC m; N_ANC itself comes from theta), so    #
# grids are in those units. Each grid value pins the parameter and          #
# re-optimizes the rest from n_starts starts: start 0 warm-starts from the   #
# arm's aggregated moments best fit, the others are LHS draws.              #
# Target: all_sfs_profiles_real, or e.g.                                     #
#   {REAL_FIT_ROOT}/Chr2R/profiles/moments/N_FR0/profile_N_FR0.png          #
##############################################################################
_SFS_PROFILE_CFG = REAL_DATA_CFG.get("sfs_profile_likelihood") or {}
SFS_PROFILE_GRIDS = {
    p: list(_np.logspace(_np.log10(float(lo)), _np.log10(float(hi)), int(n)))
    for p, (lo, hi, n) in _SFS_PROFILE_CFG.get("grids", {}).items()
}
SFS_PROFILE_N_STARTS = int(_SFS_PROFILE_CFG.get("n_starts", 8))

if SFS_PROFILE_GRIDS:
    def _sfs_profile_root(w):
        return f"{REAL_FIT_ROOT}/{w.arm}/profiles/moments/{w.pparam}"

    rule sfs_profile_point_real_by_arm:
        input:
            sfs  = lambda w: per_chrom_sfs(w.arm),
            meta = lambda w: per_chrom_sfs_meta(w.arm),
            best = f"{REAL_FIT_ROOT}/{{arm}}/inferences/moments/best_fit.pkl",
        output:
            pkl = f"{REAL_FIT_ROOT}/{{arm}}/profiles/moments/{{pparam}}/pt{{k}}/start{{s}}/moments/best_fit.pkl",
        wildcard_constraints:
            arm    = r"Chr(2L|2R|3L|3R|X)",
            pparam = "|".join(SFS_PROFILE_GRIDS),
            k      = r"\d+",
            s      = r"\d+",
        params:
            outdir   = lambda w: f"{_sfs_profile_root(w)}/pt{w.k}/start{w.s}",
            cfg      = EXP_CFG,
            model_py = (
                f"demes_models:{MODEL}_model"
                if MODEL != "drosophila_three_epoch"
                else "demes_models:drosophila_three_epoch"
            ),
            fix      = lambda w: f"{w.pparam}={SFS_PROFILE_GRIDS[w.pparam][int(w.k)]:.6g}",
            start    = lambda w, input: (f'--x0-from "{input.best}"' if int(w.s) == 0
                                         else f"--opt-seed {w.s}"),
        threads: 2
        shell:
            r"""
            set -euo pipefail
            seq_len=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['sequence_length'])" "{input.meta}")
            PYTHONPATH={workflow.basedir} \
            python snakemake_scripts/moments_dadi_inference_real.py \
              --mode moments \
              --sfs-file "{input.sfs}" \
              --config "{params.cfg}" \
              --model-py "{params.model_py}" \
              --outdir "{params.outdir}" \
              --real-sequence-length "$seq_len" \
              --fix "{params.fix}" \
              {params.start}
            test -f "{output.pkl}"
            """

    rule sfs_profile_real_by_arm:
        input:
            fits = lambda w: [f"{_sfs_profile_root(w)}/pt{k}/start{s}/moments/best_fit.pkl"
                              for k in range(len(SFS_PROFILE_GRIDS[w.pparam]))
                              for s in range(SFS_PROFILE_N_STARTS)],
            best = f"{REAL_FIT_ROOT}/{{arm}}/inferences/moments/best_fit.pkl",
        output:
            tsv = f"{REAL_FIT_ROOT}/{{arm}}/profiles/moments/{{pparam}}/profile_{{pparam}}.tsv",
            png = f"{REAL_FIT_ROOT}/{{arm}}/profiles/moments/{{pparam}}/profile_{{pparam}}.png",
        wildcard_constraints:
            arm    = r"Chr(2L|2R|3L|3R|X)",
            pparam = "|".join(SFS_PROFILE_GRIDS),
        params:
            out   = lambda w: f"{_sfs_profile_root(w)}/profile_{w.pparam}",
            cfg   = EXP_CFG,
            title = lambda w: f"{w.arm} moments SFS",
        threads: 1
        shell:
            r"""
            set -euo pipefail
            python snakemake_scripts/plot_momentsld_profile.py --scaled \
                --param {wildcards.pparam} --fits {input.fits} --overall-best "{input.best}" \
                --config "{params.cfg}" --title "{params.title}" --out "{params.out}"
            """

    rule all_sfs_profiles_real:
        """Convenience target: every configured SFS profile, for every configured arm."""
        input:
            expand(
                f"{REAL_FIT_ROOT}/{{arm}}/profiles/moments/{{pparam}}/profile_{{pparam}}.png",
                arm=REAL_ARMS, pparam=list(SFS_PROFILE_GRIDS),
            ),


rule all_momentsld_real_individual:
    """Convenience target: every configured arm's independent (unpooled) MomentsLD fit."""
    input:
        expand(
            f"{REAL_FIT_ROOT}/{{arm}}/inferences/{REAL_LD_ENGINE_ARM}/best_fit.pkl",
            arm=REAL_ARMS,
        ),


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

        # Only required when gram_schmidt=true; otherwise create temp sentinels
        gs_coeffs = (
            temp(f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/residuals_gs_coeffs.npy")
            if USE_GS
            else temp(f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/.gs_disabled")
        ),
        gs_basis = (
            f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/residuals_gs_basis.npy"
            if USE_GS
            else temp(f"{REAL_INF_ROOT}/sfs_residuals/{{engine}}/.gs_basis_disabled")
        ),
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

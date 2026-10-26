##############################################################################
# Simulated data: simulate -> SFS/LD inference -> combine_results
##############################################################################

##############################################################################
# Simulation-only settings and paths
##############################################################################
# External scripts
SIM_SCRIPT   = "snakemake_scripts/simulation.py"
WIN_SCRIPT   = "snakemake_scripts/simulate_window_replicate.py"
LD_SCRIPT    = "snakemake_scripts/compute_ld_window.py"

NUM_WINDOWS   = int(CFG.get("num_windows", 100))

# window_mode: "replicates" (default) independently re-simulates each of the
# NUM_WINDOWS windows for a sid at chunk_genome_length bp each
# (simulate_one_window_replicate, one msprime/SLiM run per window). "chunked"
# simulates ONE original tree sequence per sid at genome_length bp and chops
# it into NUM_WINDOWS overlapping windows of chunk_genome_length bp each via
# src/windowing.py::window_trees. chunk_genome_length falls back to
# genome_length when not set in the config (today's un-chunked behavior).
WINDOW_MODE = CFG.get("window_mode", "replicates")
if WINDOW_MODE not in ("replicates", "chunked"):
    raise ValueError(f"window_mode must be 'replicates' or 'chunked', got {WINDOW_MODE!r}")

WINDOWS = range(NUM_WINDOWS)
OPTIMS  = list(range(NUM_OPTIMS))

# Canonical path builders
SIM_BASEDIR = f"experiments/{MODEL}/simulations"
RUN_DIR     = lambda sid, opt: f"experiments/{MODEL}/runs/run_{sid}_{opt}"
LD_ROOT     = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD"


# Optional pruning of each LD window before LD stats are computed: one
# setting at a time, "prune_mode" ("off" | "fraction" | "count") +
# "prune_keep_values" ([one value]). With pruning on, every LD output for a
# sim lives under MomentsLD/pruning/<tag>/ instead of MomentsLD/ -- see
# src/ld_paths.py. Raw (unthinned) windows always stay in MomentsLD/windows/.
from src.ld_paths import prune_tag, prune_flag, momentsld_subdir
PRUNE_TAG  = prune_tag(CFG)             # e.g. "n5000", or None
LD_SUBDIR  = momentsld_subdir(CFG)      # "MomentsLD" or "MomentsLD/pruning/<tag>"
LD_DIR     = f"experiments/{MODEL}/inferences/sim_{{sid}}/{LD_SUBDIR}"          # LD stats, means/varcovs, aggregated fit
LD_RUN_DIR = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/{LD_SUBDIR}"  # one optimizer restart

# Aggregated (top-K) fit per tool and sim (rule aggregate_opts). {fit} is
# moments, dadi or LD_SUBDIR, so the MomentsLD one is LD_DIR/best_fit.pkl.
SIM_FITS = ["moments", "dadi", LD_SUBDIR]
AGG_FIT  = f"experiments/{MODEL}/inferences/sim_{{sid}}/{{fit}}/best_fit.pkl"
MOMENTSLD_BEST_FIT = AGG_FIT.replace("{fit}", LD_SUBDIR)

wildcard_constraints:
    fit = "|".join(re.escape(f) for f in SIM_FITS),


##############################################################################
# RULE simulate – one complete tree‑sequence + SFS
##############################################################################
rule simulate:
    output:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
        fig    = f"{SIM_BASEDIR}/{{sid}}/demes.png",
        meta   = f"{SIM_BASEDIR}/{{sid}}/bgs.meta.json",
        # window_mode="chunked": chunk_window (a separate, later job) slices
        # this same tree sequence into windows, so it must persist past this
        # rule's own job. window_mode="replicates" never consumes it, so it
        # stays temp (auto-deleted) there, same as always.
        ts     = (f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees" if WINDOW_MODE == "chunked"
                  else temp(f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees")),
        done   = protected(f"{SIM_BASEDIR}/{{sid}}/.done"),
    params:
        sim_dir = SIM_BASEDIR,
        cfg     = EXP_CFG,
        model   = MODEL
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python "{SIM_SCRIPT}" \
          --simulation-dir "{params.sim_dir}" \
          --experiment-config "{params.cfg}" \
          --model-type "{params.model}" \
          --simulation-number {wildcards.sid}

        # ensure expected outputs exist, then create sentinel
        test -f "{output.sfs}"    && \
        test -f "{output.params}" && \
        test -f "{output.fig}"    && \
        test -f "{output.meta}"
        touch "{output.done}"
        """


##############################################################################
# RULE infer_engine – one optimizer restart of moments or dadi ({engine}) on the SFS
##############################################################################
rule infer_engine:
    input:
        sfs    = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        params = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",   # not read by moments; kept for DAG clarity
        cfg    = EXP_CFG
    output:
        pkl = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/{{engine}}/best_fit.pkl"
    params:
        run_dir  = lambda w: RUN_DIR(w.sid, w.opt),
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
        fix      = ""     # e.g. '--fix N0=10000 --fix m12=0.0'
    # No internal multi-threaded code path in moments_dadi_inference.py --
    # OMP/MKL threads only apply to whatever BLAS backend numpy uses, and at
    # this grid size (pts_base up to ~40) that overhead outweighs any benefit.
    # threads:1 also lets moments.sh/dadi.sh run several restarts concurrently
    # per array task (via -j) without oversubscribing cores.
    threads: 1
    shell:
        r"""
        set -euo pipefail

        # Skip already-completed opts even if Snakemake scheduled this job (e.g.
        # because the config file's mtime changed on a git pull) — don't waste
        # time/compute re-running an optimization that already finished.
        if [ -s "{output.pkl}" ]; then
            echo "SKIP: {output.pkl} already exists and is non-empty"
            exit 0
        fi

        echo "===== infer_{wildcards.engine} ENV ====="
        echo "sid={wildcards.sid} opt={wildcards.opt}"
        echo "SLURM_JOB_ID=${{SLURM_JOB_ID:-unset}} SLURM_ARRAY_TASK_ID=${{SLURM_ARRAY_TASK_ID:-unset}}"
        echo "CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-unset}}"
        echo "PYCUDA_CACHE_DIR=${{PYCUDA_CACHE_DIR:-unset}}"
        echo "CUDAHOSTCXX=${{CUDAHOSTCXX:-unset}}"
        command -v nvcc >/dev/null 2>&1 && echo "nvcc=$(command -v nvcc)" || echo "nvcc=NOT_FOUND"
        nvidia-smi -L || true

        # Keep threading sane
        export OMP_NUM_THREADS={threads}
        export MKL_NUM_THREADS={threads}

        # Ensure output dirs exist
        mkdir -p "{params.run_dir}/inferences/{wildcards.engine}"

        PYTHONPATH={workflow.basedir} \
        python "snakemake_scripts/moments_dadi_inference.py" \
          --mode {wildcards.engine} \
          --sfs-file "{input.sfs}" \
          --config "{params.cfg}" \
          --model-py "{params.model_py}" \
          --ground-truth "{input.params}" \
          --outdir "{params.run_dir}/inferences" \
          --opt-seed {wildcards.opt} {params.fix}
        """

##############################################################################
# RULE aggregate_opts – keep the top-K optimizer restarts for one tool/sim   #
# {fit} is moments, dadi, or LD_SUBDIR (MomentsLD[/pruning/<tag>]); reads   #
# runs/run_<sid>_<opt>/inferences/<fit>/best_fit.pkl and writes            #
# inferences/sim_<sid>/<fit>/best_fit.pkl.                                  #
# No hard dependency on the per-restart files on purpose: aggregation works #
# on however many restarts currently exist (found by glob below), so it     #
# never re-triggers restarts that were removed. aggregate_opts.sh's FORCE=1 #
# re-aggregates once more restarts have finished.                           #
##############################################################################
rule aggregate_opts:
    input:
        cfg = EXP_CFG,
    output:
        pkl = AGG_FIT,
    run:
        import pickle, pathlib, re
        from src.aggregate_utils import discover_opt_pkls, aggregate_top_k

        sid, fit = wildcards.sid, wildcards.fit
        tool = fit.split("/")[0]   # moments | dadi | MomentsLD
        MIN_FILES = int(CFG.get("aggregate_min_replicates", 5))

        records = discover_opt_pkls(
            f"experiments/{MODEL}/runs/run_{sid}_*/inferences/{fit}/best_fit.pkl",
            rf"/run_{sid}_(\d+)/inferences/{re.escape(fit)}/best_fit\.pkl$",
        )

        best, diag = aggregate_top_k(
            records, TOP_K, min_nonempty=MIN_FILES,
            err_label=f"aggregate_opts ({fit})", err_engine=tool, err_context=f"for sid={sid}",
        )
        best["n_files_found"] = diag["n_records"]
        best["n_nonempty"]    = diag["n_nonempty"]
        best["min_required"]  = int(TOP_K)

        pathlib.Path(output.pkl).parent.mkdir(parents=True, exist_ok=True)
        with open(output.pkl, "wb") as fh:
            pickle.dump(best, fh)

        print(f"✅ {fit}: found {diag['n_records']} files, aggregated {diag['n_entries']} entries → {output.pkl}")
        print(f"✅ {fit}: kept top-{TOP_K} opts={sorted(set(best.get('opt_index', [])))}")

##############################################################################
# RULE simulate_window_replicate / chunk_window – one VCF window
# window_mode="replicates": simulate_window_replicate independently
# re-simulates each window from scratch, at chunk_genome_length bp
# (falls back to genome_length if chunk_genome_length is unset).
# window_mode="chunked": chunk_window chops rule simulate's own
# tree_sequence.trees (genome_length bp, the original/full sequence
# already simulated for the SFS) into NUM_WINDOWS overlapping windows of
# chunk_genome_length bp each (src/windowing.py) – no simulation happens
# in this rule, just slicing. This reuses simulate's tree sequence
# instead of re-simulating a second one, so there's no redundant
# simulation between the SFS path and the LD/windowing path.
# Both branches produce the same output path so downstream rules
# (ld_window) and callers don't need to know which mode is active.
##############################################################################
if WINDOW_MODE == "replicates":
    rule simulate_window_replicate:
        input:
            params   = f"{SIM_BASEDIR}/{{sid}}/sampled_params.pkl",
            metafile = f"{SIM_BASEDIR}/{{sid}}/bgs.meta.json",
            done     = f"{SIM_BASEDIR}/{{sid}}/.done"
        output:
            vcf_gz = f"{LD_ROOT}/windows/window_{{win}}.vcf.gz"
        params:
            base_sim   = lambda w: f"{SIM_BASEDIR}/{w.sid}",
            out_winDir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/windows",
            rep_idx    = "{win}",
            cfg        = EXP_CFG
        threads: 1
        shell:
            r"""
            set -euo pipefail
            PYTHONPATH={workflow.basedir} \
            python "{WIN_SCRIPT}" \
                --sim-dir      "{params.base_sim}" \
                --rep-index    {params.rep_idx} \
                --config-file  "{params.cfg}" \
                --meta-file    "{input.metafile}" \
                --out-dir      "{params.out_winDir}"
            """
else:
    # Two rules instead of one: materialize_sim_vcf writes the WHOLE genome
    # to one bgzipped, indexed VCF (one per simulation, no keep_intervals/
    # simplify at all -- see src.windowing.materialize_full_vcf). chunk_window
    # then does a per-window `bcftools view -r` slice against that finished
    # file. Measured ~4.6x faster per window than the tree-sequence
    # keep_intervals(simplify=True) approach on a test tree sequence, and
    # since slicing is read-only against an already-finished file, every
    # window's chunk_window job is independent again -- safe to run fully in
    # parallel across windows, same as before any of this was batched.
    rule materialize_sim_vcf:
        input:
            trees = f"{SIM_BASEDIR}/{{sid}}/tree_sequence.trees",
        output:
            # Not temp(): chunk_window's --allowed-rules restriction (see
            # build_windows.sh) means it can't rebuild these if a later
            # retry/rerun of any single window needs them again after
            # Snakemake's temp-cleanup removed them -- MissingInputException,
            # stuck. One whole-genome VCF kept per simulation is cheap
            # relative to everything else in this pipeline.
            vcf_gz  = f"{LD_ROOT}/windows/full_genome.vcf.gz",
            vcf_tbi = f"{LD_ROOT}/windows/full_genome.vcf.gz.tbi",
            samples = f"{LD_ROOT}/windows/samples.txt",
        params:
            out_winDir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/windows",
        threads: 1
        run:
            from src.windowing import materialize_full_vcf

            # cfg: sample_ploidy 1 -> phased pseudo-diploid pairs; a "map"
            # recombination config -> windows/genetic_map.txt for LD binning.
            materialize_full_vcf(Path(input.trees), Path(params.out_winDir), cfg=CFG)

    rule chunk_window:
        input:
            vcf_gz  = f"{LD_ROOT}/windows/full_genome.vcf.gz",
            vcf_tbi = f"{LD_ROOT}/windows/full_genome.vcf.gz.tbi",
        output:
            vcf_gz = f"{LD_ROOT}/windows/window_{{win}}.vcf.gz"
        params:
            out_winDir  = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/windows",
            cfg         = EXP_CFG,
            num_windows = NUM_WINDOWS,
        threads: 1
        run:
            import json
            from src.windowing import window_vcf

            cfg = json.loads(Path(params.cfg).read_text())
            window_size = int(cfg.get("ld_sequence_length", cfg["sequence_length"]))
            recomb_rate = float(cfg["recombination_rate"])

            window_vcf(
                Path(input.vcf_gz),
                Path(params.out_winDir),
                window_size=window_size,
                num_windows=params.num_windows,
                recomb_rate=recomb_rate,
                window_index=int(wildcards.win),
            )

##############################################################################
# RULE prune_window – thin one raw window (only defined when pruning is on)  #
# src/prune_vcf.py writes <out-dir>/<tag>/window_<win>.vcf.gz, i.e. into    #
# LD_DIR/windows/. temp(): only ld_window reads it.                         #
##############################################################################
if PRUNE_TAG:
    rule prune_window:
        input:
            vcf_gz = f"{LD_ROOT}/windows/window_{{win}}.vcf.gz",
        output:
            pruned_vcf = temp(f"{LD_DIR}/windows/window_{{win}}.vcf.gz"),
        params:
            pruning_dir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning",
            prune_flag  = prune_flag(CFG),
        threads: 1
        shell:
            """
            set -euo pipefail
            PYTHONPATH={workflow.basedir} \
            python "src/prune_vcf.py" \
                --vcf            "{input.vcf_gz}" \
                --out-dir        "{params.pruning_dir}" \
                {params.prune_flag}                     \
                --no-unpruned                           \
                --workers        1
            """

##############################################################################
# RULE ld_window – LD statistics for one window                              #
# Reads LD_DIR/windows/window_<win>.vcf.gz: the raw window when pruning is  #
# off (LD_DIR == LD_ROOT), the thinned one from prune_window when it's on.  #
##############################################################################
rule ld_window:
    input:
        vcf_gz = f"{LD_DIR}/windows/window_{{win}}.vcf.gz",
    output:
        pkl = f"{LD_DIR}/LD_stats/LD_stats_window_{{win}}.pkl"
    params:
        sim_dir = lambda w: LD_DIR.format(sid=w.sid),
        bins    = R_BINS_STR,
        cfg     = EXP_CFG
    # No threading/multiprocessing code in moments.LD.Parsing.compute_ld_statistics
    # (the CPU path used when use_gpu_ld=false) -- threads:1 so
    # LD_stats_windows.sh's batched Snakemake call can run several windows
    # concurrently per array task.
    threads: 1
    resources:
        ld_cores = 1,
        gpu      = 1
    shell:
        """
        set -euo pipefail
        python "{LD_SCRIPT}" \
            --sim-dir      {params.sim_dir} \
            --window-index {wildcards.win} \
            --config-file  {params.cfg} \
            --r-bins       "{params.bins}"

        # .h5 is written by the LD script but not declared as a Snakemake output;
        # remove it here so it doesn't accumulate across windows.
        rm -f {params.sim_dir}/windows/window_{wildcards.win}.h5
        """

##############################################################################
# RULE aggregate_ld_stats – aggregate LD windows into means/varcovs + PDF   #
# Runs once per sid; every optimization restart (infer_momentsld, below)    #
# reuses this cached means.varcovs.pkl instead of re-aggregating.           #
##############################################################################
rule aggregate_ld_stats:
    input:
        pkls = lambda w: expand(
            f"{LD_DIR}/LD_stats/LD_stats_window_{{win}}.pkl",
            sid=[w.sid],
            win=WINDOWS
        ),
        cfg = EXP_CFG,
    output:
        mv   = f"{LD_DIR}/means.varcovs.pkl",
        # Not temp(): aggregate_ld_statistics() checks for this file (alongside
        # means.varcovs.pkl) to short-circuit re-aggregating LD_stats/*.pkl on
        # every downstream infer_momentsld restart. Deleting it right after this
        # rule finished (former temp()) silently broke that cache -- every one
        # of the num_optimizations restarts was re-reading and re-aggregating
        # all window LD_stats from scratch instead of reusing this.
        boot = f"{LD_DIR}/bootstrap_sets.pkl",
        pdf  = f"{LD_DIR}/empirical_vs_theoretical_comparison.pdf",
    params:
        sim_dir     = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        output_root = lambda w: LD_DIR.format(sid=w.sid),
        bins = R_BINS_STR,
        cfg = EXP_CFG,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python "snakemake_scripts/LD_inference.py" \
            --run-dir     "{params.sim_dir}" \
            --output-root "{params.output_root}" \
            --config-file "{params.cfg}" \
            --r-bins      "{params.bins}" \
            --skip-optimize
        """

##############################################################################
# RULE infer_momentsld – one LHS-seeded MomentsLD restart                   #
# Mirrors infer_engine: one Snakemake job per {opt}, each a single nlopt    #
# run from a distinct start point keyed by opt_seed.                        #
##############################################################################
rule infer_momentsld:
    input:
        mv  = f"{LD_DIR}/means.varcovs.pkl",
        cfg = EXP_CFG,
    output:
        pkl = f"{LD_RUN_DIR}/best_fit.pkl"
    params:
        sim_dir     = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        output_root = lambda w: LD_DIR.format(sid=w.sid),
        results_dir = lambda w: LD_RUN_DIR.format(sid=w.sid, opt=w.opt),
        bins = R_BINS_STR,
        cfg = EXP_CFG,
    threads: 1
    shell:
        r"""
        set -euo pipefail
        mkdir -p "{params.results_dir}"

        PYTHONPATH={workflow.basedir} \
        python "snakemake_scripts/LD_inference.py" \
            --run-dir     "{params.sim_dir}" \
            --output-root "{params.output_root}" \
            --results-dir "{params.results_dir}" \
            --config-file "{params.cfg}" \
            --r-bins      "{params.bins}" \
            --opt-seed    {wildcards.opt}

        test -f "{output.pkl}"
        """

##############################################################################
# RULE compute_fim – observed FIM at best-LL params for {engine}             #
##############################################################################
rule compute_fim:
    input:
        fit = lambda w: AGG_FIT.format(sid=w.sid, fit=w.engine),
        sfs = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl"
    output:
        fim  = temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/fim/{{engine}}.fim.npy"),
        summ = f"experiments/{MODEL}/inferences/sim_{{sid}}/fim/{{engine}}.summary.json",
    params:
        script = "snakemake_scripts/compute_fim.py",
        cfg    = EXP_CFG
    threads: 2
    shell:
        r"""
        PYTHONPATH={workflow.basedir} \
        python {params.script} \
            --engine {wildcards.engine} \
            --fit-pkl {input.fit} \
            --sfs {input.sfs} \
            --config {params.cfg} \
            --fim-npy {output.fim} \
            --summary-json {output.summ}
        """


##############################################################################
# RULE sfs_residuals – optimized (best-fit) SFS − observed SFS               #
##############################################################################
rule sfs_residuals:
    input:
        obs_sfs = f"{SIM_BASEDIR}/{{sid}}/SFS.pkl",
        agg_fit = lambda w: AGG_FIT.format(sid=w.sid, fit=w.engine),
    output:
        res_arr   = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals.npy",
        res_flat  = temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_flat.npy"),
        meta_json = temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/meta.json"),
        hist_png  = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_histogram.png",
        **gs_outputs(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}"),
    params:
        cfg      = EXP_CFG,
        model_py = (
            f"src.simulation:{MODEL}_model"
            if MODEL != "drosophila_three_epoch"
            else "src.simulation:drosophila_three_epoch"
        ),
        inf_dir  = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}",
        out_dir  = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/sfs_residuals/{w.engine}",
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
# RULE combine_results – merge dadi / moments / moments-LD fits per sim      #
# + attach flattened upper-triangular FIM and residual SFS payloads          #
##############################################################################
rule combine_results:
    input:
        cfg       = EXP_CFG,
        dadi      = lambda w: ancient(AGG_FIT.format(sid=w.sid, fit="dadi"))
                    if os.path.exists(AGG_FIT.format(sid=w.sid, fit="dadi")) else [],
        moments   = lambda w: ancient(AGG_FIT.format(sid=w.sid, fit="moments"))
                    if os.path.exists(AGG_FIT.format(sid=w.sid, fit="moments")) else [],
        momentsLD = lambda w: ancient(MOMENTSLD_BEST_FIT.format(sid=w.sid))
                    if os.path.exists(MOMENTSLD_BEST_FIT.format(sid=w.sid)) else [],
        # FIM/residuals are always computed (see FIM_ENGINES/RESIDUAL_ENGINES
        # above); use_fim_features/use_residuals only control whether
        # feature_extraction.py later uses them as modeling features.
        fims      = lambda w: [
            f"experiments/{MODEL}/inferences/sim_{w.sid}/fim/{eng}.fim.npy"
            for eng in FIM_ENGINES
        ],
        resid_vecs = lambda w: [
            f"experiments/{MODEL}/inferences/sim_{w.sid}/sfs_residuals/{eng}/{_resid_vector_fname()}"
            for eng in RESIDUAL_ENGINES
        ],
        resid_meta = lambda w: [
            f"experiments/{MODEL}/inferences/sim_{w.sid}/sfs_residuals/{eng}/meta.json"
            for eng in RESIDUAL_ENGINES
        ],
    output:
        combo = f"experiments/{MODEL}/inferences/sim_{{sid}}/all_inferences.pkl"
    run:
        import pickle, pathlib
        from src.combine_payloads import build_fim_payload, build_residual_payload

        if not input.dadi or not input.moments or not input.momentsLD:
            missing = [k for k, v in [("dadi", input.dadi), ("moments", input.moments), ("momentsLD", input.momentsLD)] if not v]
            raise RuntimeError(f"sim_{wildcards.sid}: skipping — prerequisites not ready: {missing}")

        outdir = pathlib.Path(output.combo).parent
        outdir.mkdir(parents=True, exist_ok=True)

        summary = {}
        summary["moments"]   = pickle.load(open(input.moments, "rb"))
        summary["dadi"]      = pickle.load(open(input.dadi, "rb"))
        summary["momentsLD"] = pickle.load(open(input.momentsLD, "rb"))

        fim_payload = build_fim_payload(input.fims)
        if fim_payload:
            summary["FIM"] = fim_payload

        # Either residuals_flat.npy (raw) or residuals_gs_coeffs.npy (GS reduced).
        resid_payload = build_residual_payload(input.resid_vecs)
        if resid_payload:
            summary["SFS_residuals"] = resid_payload

        pickle.dump(summary, open(output.combo, "wb"))
        print(f"✓ combined → {output.combo}")

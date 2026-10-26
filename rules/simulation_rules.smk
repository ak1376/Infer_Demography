##############################################################################
# Simulated data: simulate -> SFS/LD inference -> combine_results
##############################################################################

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
# RULE infer_moments  – custom NLopt Poisson SFS optimisation (moments)
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

# ── MOMENTS / DADI (sim) ────────────────────────────────────────────────────
rule aggregate_opts_engine:
    input:
        cfg = EXP_CFG,
        # No hard dependency on the full OPTIMS range here on purpose: aggregation
        # should work on however many per-opt files currently exist (discovered via
        # glob in the run: block below), not force every optimization to finish
        # first. aggregate_moments_dadi.sh's FORCE=1 is what triggers a fresh
        # aggregation pass once more opts have completed.
    output:
        pkl = f"experiments/{MODEL}/inferences/sim_{{sid}}/{{engine}}/fit_params.pkl"
    run:
        import pickle, pathlib
        from src.aggregate_utils import discover_opt_pkls, aggregate_top_k

        sid = wildcards.sid
        engine = wildcards.engine
        MIN_FILES = int(CFG.get("aggregate_min_replicates", 5))

        records = discover_opt_pkls(
            f"experiments/{MODEL}/runs/run_{sid}_*/inferences/{engine}/best_fit.pkl",
            rf"/run_{sid}_(\d+)/inferences/{engine}/best_fit\.pkl$",
        )

        best, diag = aggregate_top_k(
            records, TOP_K, min_nonempty=MIN_FILES,
            err_label=f"aggregate_opts_{engine}", err_engine=engine, err_context=f"for sid={sid}",
        )
        best["n_files_found"] = diag["n_records"]
        best["n_nonempty"]    = diag["n_nonempty"]
        best["min_required"]  = int(TOP_K)

        pathlib.Path(output.pkl).parent.mkdir(parents=True, exist_ok=True)
        with open(output.pkl, "wb") as fh:
            pickle.dump(best, fh)

        print(f"✅ {engine}: found {diag['n_records']} files, aggregated {diag['n_entries']} entries → {output.pkl}")
        print(f"✅ {engine}: kept top-{TOP_K} opts={sorted(set(best.get('opt_index', [])))}")

# ── CLEANUP RULE: Remove non-top-K optimization runs after all three
#    engines (dadi, moments, MomentsLD) have finished aggregating. Must wait
#    on MomentsLD too -- its top-K opt indices (by LD likelihood) don't
#    necessarily overlap with dadi/moments' (by SFS likelihood), so deleting
#    a run dir just because it wasn't in the dadi/moments keep-set can throw
#    away a restart MomentsLD still needed. ─────────────────────────────────
rule cleanup_optimization_runs:
    input:
        dadi       = f"experiments/{MODEL}/inferences/sim_{{sid}}/dadi/fit_params.pkl",
        moments    = f"experiments/{MODEL}/inferences/sim_{{sid}}/moments/fit_params.pkl",
        momentsld  = MOMENTSLD_AGG_TARGETS,
    output:
        cleanup_done = f"experiments/{MODEL}/inferences/sim_{{sid}}/cleanup_done.txt"
    run:
        import pickle, pathlib, subprocess

        sid = wildcards.sid

        # --- skip conditions ---
        if NUM_OPTIMS <= 1 or TOP_K >= NUM_OPTIMS:
            pathlib.Path(output.cleanup_done).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(output.cleanup_done).write_text(
                f"Cleanup skipped for simulation {sid} (NUM_OPTIMS={NUM_OPTIMS}, TOP_K={TOP_K}).\n"
            )
            print(f"✅ cleanup skipped sid={sid}")
            return

        # --- existing cleanup logic ---
        with open(input.dadi, "rb") as f:
            dadi_data = pickle.load(f)
        with open(input.moments, "rb") as f:
            moments_data = pickle.load(f)
        momentsld_keep = set()
        for momentsld_path in input.momentsld:
            with open(momentsld_path, "rb") as f:
                momentsld_data = pickle.load(f)
            momentsld_keep |= set((momentsld_data.get("opt_index") or [])[:TOP_K])

        dadi_keep    = set((dadi_data.get("opt_index") or [])[:TOP_K])
        moments_keep = set((moments_data.get("opt_index") or [])[:TOP_K])
        keep_indices = dadi_keep | moments_keep | momentsld_keep

        run_root = pathlib.Path(f"experiments/{MODEL}/runs")
        prefix = f"run_{sid}_"

        cleaned = 0
        for p in run_root.glob(f"{prefix}*"):
            if not p.is_dir():
                continue
            try:
                opt = int(p.name.rsplit("_", 1)[1])
            except Exception:
                continue
            if opt in keep_indices:
                continue

            subprocess.run(["rm", "-rf", str(p)], check=False)
            cleaned += 1

        pathlib.Path(output.cleanup_done).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(output.cleanup_done).write_text(
            f"Cleanup completed for simulation {sid}\n"
            f"Removed {cleaned} optimization directories\n"
            f"Kept optimizations: {sorted(keep_indices)}\n"
        )


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
# RULE ld_window – LD statistics for one window
# prune_mode "off" (default): computes stats directly on the full window
# (unchanged, original behavior).
# prune_mode "fraction"/"count" set: LD pair count scales as
# density^2, so a dense window (many variants) can make the full
# computation prohibitively slow. This branch thins the window first
# (reusing rule prune_window's already-existing pruned VCF) and computes
# stats on THAT instead, writing to the exact same output path - nothing
# downstream (aggregate_ld_stats etc.) needs to know pruning happened.
##############################################################################
if PRUNE_TAGS:
    rule ld_window:
        input:
            vcf_gz = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{PRUNE_TAGS[0]}/windows/window_{{win}}.vcf.gz",
        output:
            pkl = f"{LD_ROOT}/LD_stats/LD_stats_window_{{win}}.pkl"
        params:
            sim_dir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{PRUNE_TAGS[0]}",
            bins    = R_BINS_STR,
            cfg     = EXP_CFG,
        # No threading/multiprocessing code in moments.LD.Parsing.compute_ld_statistics
        # (the CPU path used when use_gpu_ld=false) -- threads:1 (was 4) so
        # LD_stats_windows.sh's batched Snakemake call can actually run
        # multiple windows concurrently per array task instead of one at a
        # time (was previously exactly matching --cpus-per-task=4, so -j
        # never had room for a second job).
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

            mkdir -p "$(dirname "{output.pkl}")"
            mv "{params.sim_dir}/LD_stats/LD_stats_window_{wildcards.win}.pkl" "{output.pkl}"
            rm -f {params.sim_dir}/windows/window_{wildcards.win}.h5
            """
else:
    rule ld_window:
        input:
            vcf_gz = f"{LD_ROOT}/windows/window_{{win}}.vcf.gz",
        output:
            pkl    = f"{LD_ROOT}/LD_stats/LD_stats_window_{{win}}.pkl"
        params:
            sim_dir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD",
            bins    = R_BINS_STR,
            cfg     = EXP_CFG
        # No threading/multiprocessing code in moments.LD.Parsing.compute_ld_statistics
        # (the CPU path used when use_gpu_ld=false) -- threads:1 (was 4) so
        # LD_stats_windows.sh's batched Snakemake call can actually run
        # multiple windows concurrently per array task instead of one at a
        # time (was previously exactly matching --cpus-per-task=4, so -j
        # never had room for a second job).
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
# RULE prune_window – thin one VCF to a keep-fraction                       #
##############################################################################
rule prune_window:
    input:
        vcf_gz = f"{LD_ROOT}/windows/window_{{win}}.vcf.gz",
    output:
        pruned_vcf = temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/windows/window_{{win}}.vcf.gz"),
    params:
        pruning_dir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning",
        windows_dir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/windows",
        # frac_tag encodes which pruning mode produced it: "thinNN" (keep NN%
        # of sites) or "nNNN" (cap at NNN sites, min(n_full, NNN) -- see
        # src/prune_vcf.py's --keep-counts mode).
        prune_flag  = lambda w: (
            f"--keep-counts {int(w.frac_tag[1:])}"
            if w.frac_tag.startswith("n")
            else f"--keep-fractions {int(w.frac_tag.replace('thin', '')) / 100}"
        ),
    threads: 1
    shell:
        """
        set -euo pipefail
        PYTHONPATH={workflow.basedir} \
        python "src/prune_vcf.py" \
            --vcf            "{params.windows_dir}/window_{wildcards.win}.vcf.gz" \
            --out-dir        "{params.pruning_dir}" \
            {params.prune_flag}                     \
            --no-unpruned                           \
            --workers        1
        """

##############################################################################
# RULE ld_window_pruned – LD stats for one pruned window                    #
##############################################################################
rule ld_window_pruned:
    input:
        vcf_gz = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/windows/window_{{win}}.vcf.gz",
    output:
        pkl = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/LD_stats/LD_stats_window_{{win}}.pkl",
    params:
        sim_dir = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{w.frac_tag}",
        bins    = R_BINS_STR,
        cfg     = EXP_CFG,
    # See rule ld_window above: no threading/multiprocessing in
    # moments.LD.Parsing.compute_ld_statistics, so threads:1 (was 4) lets
    # LD_stats_windows.sh's batched call run windows concurrently.
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

        rm -f {params.sim_dir}/windows/window_{wildcards.win}.h5
        """

##############################################################################
# RULE aggregate_ld_stats – aggregate LD windows into means/varcovs + PDF   #
# Works for unpruned-only AND mixed (unpruned primary + pruned fallback).   #
# Runs once per sid; every optimization restart (infer_momentsld, below)    #
# reuses this cached means.varcovs.pkl instead of re-aggregating.           #
##############################################################################
rule aggregate_ld_stats:
    input:
        pkls = lambda w: expand(
            f"{LD_ROOT}/LD_stats/LD_stats_window_{{win}}.pkl",
            sid=[w.sid],
            win=WINDOWS
        ),
        pruned_pkls = lambda w: (
            [
                f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{PRUNE_TAGS[0]}/LD_stats/LD_stats_window_{win}.pkl"
                for win in WINDOWS
                if not Path(f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/LD_stats/LD_stats_window_{win}.pkl").exists()
            ]
            if PRUNE_TAGS else []
        ),
        cfg = EXP_CFG,
    output:
        mv   = f"{LD_ROOT}/means.varcovs.pkl",
        # Not temp(): aggregate_ld_statistics() checks for this file (alongside
        # means.varcovs.pkl) to short-circuit re-aggregating LD_stats/*.pkl on
        # every downstream infer_momentsld restart. Deleting it right after this
        # rule finished (former temp()) silently broke that cache -- every one
        # of the num_optimizations restarts was re-reading and re-aggregating
        # all window LD_stats from scratch instead of reusing this.
        boot = f"{LD_ROOT}/bootstrap_sets.pkl",
        pdf  = f"{LD_ROOT}/empirical_vs_theoretical_comparison.pdf",
    params:
        sim_dir     = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        output_root = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD",
        pruning_dir = lambda w: (
            f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{PRUNE_TAGS[0]}"
            if PRUNE_TAGS else ""
        ),
        bins = R_BINS_STR,
        cfg = EXP_CFG,
    threads: 1
    run:
        import subprocess
        cmd = [
            "python", "snakemake_scripts/LD_inference.py",
            "--run-dir",     params.sim_dir,
            "--output-root", params.output_root,
            "--config-file", params.cfg,
            "--r-bins",      params.bins,
            "--skip-optimize",
        ]
        if params.pruning_dir:
            cmd += ["--fallback-ld-dir", params.pruning_dir]
        env = {**os.environ, "PYTHONPATH": workflow.basedir}
        subprocess.run(cmd, check=True, env=env)

##############################################################################
# RULE infer_momentsld – one LHS-seeded MomentsLD restart                   #
# Mirrors infer_moments/infer_dadi: one Snakemake job per {opt}, each a     #
# single nlopt run from a distinct start point keyed by opt_seed.           #
##############################################################################
rule infer_momentsld:
    input:
        mv  = f"{LD_ROOT}/means.varcovs.pkl",
        cfg = EXP_CFG,
    output:
        pkl = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/MomentsLD/best_fit.pkl"
    params:
        sim_dir     = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        output_root = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD",
        results_dir = lambda w: f"experiments/{MODEL}/runs/run_{w.sid}_{w.opt}/inferences/MomentsLD",
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
# RULE aggregate_opts_momentsld – pick top-K across LHS restarts            #
# Mirrors aggregate_opts_moments/aggregate_opts_dadi exactly (same TOP_K,   #
# same best_params/best_ll/opt_index list schema).                         #
##############################################################################
rule aggregate_opts_momentsld:
    input:
        cfg = EXP_CFG,
        opts = lambda w: expand(
            f"experiments/{MODEL}/runs/run_{w.sid}_{{opt}}/inferences/MomentsLD/best_fit.pkl",
            opt=OPTIMS,
        ),
    output:
        best = f"{LD_ROOT}/best_fit.pkl",
    run:
        import pickle, pathlib
        from src.aggregate_utils import discover_opt_pkls, aggregate_top_k

        sid = wildcards.sid
        MIN_FILES = int(CFG.get("aggregate_min_replicates", 5))

        records = discover_opt_pkls(
            f"experiments/{MODEL}/runs/run_{sid}_*/inferences/MomentsLD/best_fit.pkl",
            rf"/run_{sid}_(\d+)/inferences/MomentsLD/best_fit\.pkl$",
        )

        best, diag = aggregate_top_k(
            records, TOP_K, min_nonempty=MIN_FILES,
            err_label="aggregate_opts_momentsld", err_engine="MomentsLD", err_context=f"for sid={sid}",
        )
        best["n_files_found"] = diag["n_records"]
        best["n_nonempty"]    = diag["n_nonempty"]
        best["min_required"]  = int(TOP_K)

        pathlib.Path(output.best).parent.mkdir(parents=True, exist_ok=True)
        with open(output.best, "wb") as fh:
            pickle.dump(best, fh)

        print(f"✅ momentsLD: found {diag['n_records']} files, aggregated {diag['n_entries']} entries → {output.best}")
        print(f"✅ momentsLD: kept top-{TOP_K} opts={sorted(set(best.get('opt_index', [])))}")

##############################################################################
# RULE aggregate_ld_stats_pruned – aggregate PRUNED per-window LD stats into  #
# means/varcovs (pruned-only; mirrors aggregate_ld_stats but reads pruned    #
# LD_stats and never runs the optimizer itself -- that's now a separate,     #
# multi-restart stage below (infer_momentsld_pruned / aggregate_opts_        #
# momentsld_pruned), same split as the unpruned aggregate_ld_stats /         #
# infer_momentsld / aggregate_opts_momentsld pattern.                        #
##############################################################################
rule aggregate_ld_stats_pruned:
    input:
        pruned_pkls = lambda w: expand(
            f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/LD_stats/LD_stats_window_{{win}}.pkl",
            sid=[w.sid],
            frac_tag=[w.frac_tag],
            win=WINDOWS,
        ),
        cfg = EXP_CFG,
    output:
        mv   = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/means.varcovs.pkl",
        # Not temp() -- see aggregate_ld_stats's identical note above.
        boot = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/bootstrap_sets.pkl",
        pdf  = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/empirical_vs_theoretical_comparison.pdf",
    params:
        sim_dir     = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        output_root = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{w.frac_tag}",
        bins = R_BINS_STR,
        cfg = EXP_CFG,
    threads: 1
    run:
        import subprocess
        # output_root drives everything: LD_inference reads output_root/LD_stats/*.pkl
        # (the pruned stats) and writes means/varcovs/bootstrap/pdf there.
        # No --fallback-ld-dir: in pruned-only mode there are no unpruned stats.
        # --skip-optimize: aggregation only here, same as aggregate_ld_stats;
        # optimization is the separate multi-restart stage below.
        cmd = [
            "python", "snakemake_scripts/LD_inference.py",
            "--run-dir",     params.sim_dir,
            "--output-root", params.output_root,
            "--config-file", params.cfg,
            "--r-bins",      params.bins,
            "--skip-optimize",
        ]
        env = {**os.environ, "PYTHONPATH": workflow.basedir}
        subprocess.run(cmd, check=True, env=env)

##############################################################################
# RULE infer_momentsld_pruned – one LHS-seeded MomentsLD restart,            #
# pruned stats. Mirrors infer_momentsld exactly, keyed additionally by       #
# {frac_tag}.                                                                #
##############################################################################
rule infer_momentsld_pruned:
    input:
        mv  = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/means.varcovs.pkl",
        cfg = EXP_CFG,
    output:
        pkl = f"experiments/{MODEL}/runs/run_{{sid}}_{{opt}}/inferences/MomentsLD/pruning/{{frac_tag}}/best_fit.pkl"
    params:
        sim_dir     = lambda w: f"{SIM_BASEDIR}/{w.sid}",
        output_root = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/MomentsLD/pruning/{w.frac_tag}",
        results_dir = lambda w: f"experiments/{MODEL}/runs/run_{w.sid}_{w.opt}/inferences/MomentsLD/pruning/{w.frac_tag}",
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
# RULE aggregate_opts_momentsld_pruned – pick top-K across LHS               #
# restarts, pruned stats. Mirrors aggregate_opts_momentsld exactly, keyed    #
# additionally by {frac_tag}. Output path is unchanged from the old         #
# optimize_momentsld_mixed target, so MomentsLD.sh's TARGET doesn't need to  #
# change -- only its --allowed-rules list does.                             #
##############################################################################
rule aggregate_opts_momentsld_pruned:
    input:
        cfg = EXP_CFG,
        opts = lambda w: expand(
            f"experiments/{MODEL}/runs/run_{w.sid}_{{opt}}/inferences/MomentsLD/pruning/{w.frac_tag}/best_fit.pkl",
            opt=OPTIMS,
        ),
    output:
        best = f"experiments/{MODEL}/inferences/sim_{{sid}}/MomentsLD/pruning/{{frac_tag}}/best_fit.pkl",
    run:
        import pickle, pathlib
        from src.aggregate_utils import discover_opt_pkls, aggregate_top_k

        sid = wildcards.sid
        frac_tag = wildcards.frac_tag
        MIN_FILES = int(CFG.get("aggregate_min_replicates", 5))

        records = discover_opt_pkls(
            f"experiments/{MODEL}/runs/run_{sid}_*/inferences/MomentsLD/pruning/{frac_tag}/best_fit.pkl",
            rf"/run_{sid}_(\d+)/inferences/MomentsLD/pruning/{frac_tag}/best_fit\.pkl$",
        )

        best, diag = aggregate_top_k(
            records, TOP_K, min_nonempty=MIN_FILES,
            err_label="aggregate_opts_momentsld_pruned", err_engine="MomentsLD",
            err_context=f"for sid={sid} frac_tag={frac_tag}",
        )
        best["n_files_found"] = diag["n_records"]
        best["n_nonempty"]    = diag["n_nonempty"]
        best["min_required"]  = int(TOP_K)

        pathlib.Path(output.best).parent.mkdir(parents=True, exist_ok=True)
        with open(output.best, "wb") as fh:
            pickle.dump(best, fh)

        print(f"✅ momentsLD (pruned, {frac_tag}): found {diag['n_records']} files, aggregated {diag['n_entries']} entries → {output.best}")
        print(f"✅ momentsLD (pruned, {frac_tag}): kept top-{TOP_K} opts={sorted(set(best.get('opt_index', [])))}")

##############################################################################
# RULE compute_fim – observed FIM at best-LL params for {engine}             #
##############################################################################
rule compute_fim:
    input:
        fit = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/{w.engine}/fit_params.pkl",
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
        agg_fit = lambda w: f"experiments/{MODEL}/inferences/sim_{w.sid}/{w.engine}/fit_params.pkl",
    output:
        res_arr   = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals.npy",
        res_flat  = temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_flat.npy"),
        meta_json = temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/meta.json"),
        hist_png  = f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_histogram.png",

        # Only required when gram_schmidt=true; otherwise create temp sentinels
        gs_coeffs = (
            temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_gs_coeffs.npy")
            if USE_GS
            else temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/.gs_disabled")
        ),
        gs_basis = (
            f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/residuals_gs_basis.npy"
            if USE_GS
            else temp(f"experiments/{MODEL}/inferences/sim_{{sid}}/sfs_residuals/{{engine}}/.gs_basis_disabled")
        ),
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
        dadi      = lambda w: ancient(f"experiments/{MODEL}/inferences/sim_{w.sid}/dadi/fit_params.pkl")
                    if os.path.exists(f"experiments/{MODEL}/inferences/sim_{w.sid}/dadi/fit_params.pkl") else [],
        moments   = lambda w: ancient(f"experiments/{MODEL}/inferences/sim_{w.sid}/moments/fit_params.pkl")
                    if os.path.exists(f"experiments/{MODEL}/inferences/sim_{w.sid}/moments/fit_params.pkl") else [],
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

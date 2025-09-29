#!/usr/bin/env python3
# stdpopsim_bgs_split_isolation.py
# Split isolation demography (custom msprime) + tiled BGS via stdpopsim + SLiM.

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import msprime
import stdpopsim as sps
import tskit
import moments
from typing import List



# ────────────────────────── (optional) calm noisy warnings ──────────────────────────
# Comment these out if you want to see all warnings.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", msprime.TimeUnitsMismatchWarning)

def create_SFS(ts: tskit.TreeSequence) -> moments.Spectrum:
    """
    Build a moments.Spectrum from a tskit TreeSequence.
    Uses populations with sampled individuals.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence from stdpopsim/SLiM.

    Returns
    -------
    moments.Spectrum
        Spectrum object with .pop_ids attribute listing population names.
    """
    sample_sets: List[np.ndarray] = []
    pop_ids: List[str] = []

    for pop in ts.populations():
        samps = ts.samples(population=pop.id)
        if len(samps) > 0:
            sample_sets.append(samps)

            # population metadata may or may not have a "name"
            meta = pop.metadata
            if isinstance(meta, dict) and "name" in meta:
                pop_ids.append(meta["name"])
            else:
                pop_ids.append(f"pop{pop.id}")

    if not sample_sets:
        raise ValueError("No sampled populations found in tree sequence.")

    arr = ts.allele_frequency_spectrum(
        sample_sets=sample_sets,
        mode="site",
        polarised=True,
        span_normalise=False,
    )

    sfs = moments.Spectrum(arr)
    sfs.pop_ids = pop_ids
    return sfs



# ────────────────────────── split-isolation model (msprime) ─────────────────────────
class SplitIsolationModel(sps.DemographicModel):
    """
    ANC splits at time T into YRI and CEU; symmetric migration m between them.
    Leaf-first population creation keeps SLiM happy (ANC last).
    """
    def __init__(self, N0: float, N1: float, N2: float, T: float, m: float):
        dem = msprime.Demography()
        dem.add_population(name="YRI", initial_size=float(N1))
        dem.add_population(name="CEU", initial_size=float(N2))
        dem.add_population(name="ANC", initial_size=float(N0))
        m = float(m)
        dem.set_migration_rate("YRI", "CEU", m)
        dem.set_migration_rate("CEU", "YRI", m)
        dem.add_population_split(time=float(T), ancestral="ANC", derived=["YRI", "CEU"])
        super().__init__(
            id="split_isolation",
            description="ANC → (YRI, CEU) at T; symmetric migration m.",
            long_description="Custom msprime demography: split isolation with symmetric migration.",
            model=dem,
            generation_time=1,
        )


# ────────────────────────── BGS tiling helpers ──────────────────────────
def build_tiling_intervals(L: int, exon_bp: int, tile_bp: int) -> np.ndarray:
    """
    Half-open intervals [start, end) of length exon_bp, placed every tile_bp.
    """
    starts = np.arange(0, max(0, L - exon_bp + 1), tile_bp, dtype=int)
    ends = np.minimum(starts + exon_bp, L).astype(int)
    return np.column_stack([starts, ends])


def make_contig_with_tiled_dfe(
    length: int, mu: float, r: float, species: str, dfe_id: str, exon_bp: int, tile_bp: int
):
    """
    Build a synthetic contig (uniform μ, r) and tile a DFE to emulate BGS.
    Falls back if stdpopsim version can't accept recombination_rate.
    """
    sp = sps.get_species(species)

    # Try newer API (supports recombination_rate for synthetic contigs).
    try:
        contig = sp.get_contig(
            chromosome=None,
            length=int(length),
            mutation_rate=float(mu),
            recombination_rate=float(r),
        )
    except TypeError:
        # Older stdpopsim: recombination_rate kwarg not supported.
        print("[warn] stdpopsim.get_contig() does not accept 'recombination_rate' "
              "in this version; using the species default r instead.")
        contig = sp.get_contig(
            chromosome=None,
            length=int(length),
            mutation_rate=float(mu),
        )

    dfe = sp.get_dfe(dfe_id)
    intervals = build_tiling_intervals(int(length), exon_bp, tile_bp)
    contig.add_dfe(intervals=intervals, DFE=dfe)
    return contig, intervals


# ────────────────────────── CLI ──────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Split isolation + tiled BGS with stdpopsim (SLiM engine)."
    )

    # Demography (split-isolation only)
    p.add_argument("--N-anc", type=float, required=True, help="Ancestral size (ANC)")
    p.add_argument("--N1", type=float, required=True, help="Size of YRI")
    p.add_argument("--N2", type=float, required=True, help="Size of CEU")
    p.add_argument("--t-split", type=float, required=True, help="Split time (generations)")
    p.add_argument("--m", type=float, default=0.0, help="Symmetric migration rate (YRI↔CEU)")

    # Samples (keys must match the model’s pop names)
    p.add_argument("--samples", default="YRI:20,CEU:20",
                   help="Comma list pop:n, e.g. 'YRI:20,CEU:20'")

    # Species is only for contig + DFE (not demography)
    p.add_argument("--species", default="HomSap", help="stdpopsim species for contig/DFE")
    p.add_argument("--dfe", default="Gamma_K17", help="stdpopsim DFE ID for tiled 'exons'")

    # Genome/rates (synthetic contig)
    p.add_argument("--length", type=int, default=200_000, help="Genome length (bp)")
    p.add_argument("--mu", type=float, default=1e-8, help="Mutation rate per bp/gen")
    p.add_argument("--r", type=float, default=1e-8, help="Recombination rate per bp/gen")

    # BGS tiling
    p.add_argument("--exon-bp", type=int, default=200, help="Exon tile size (bp)")
    p.add_argument("--tile-bp", type=int, default=5000, help="Tile period (bp): one exon every tile_bp")

    # SLiM knobs
    p.add_argument("--slim-scaling", type=float, default=10.0, help="SLiM rescaling factor Q")
    p.add_argument("--slim-burn-in", type=float, default=5.0, help="Burn-in in units of Ne")

    # I/O
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--trees", default="sims/out.trees")
    p.add_argument("--vcf", default="", help="Optional VCF path")

    return p.parse_args()


# ────────────────────────── main ──────────────────────────
def main():
    a = parse_args()

    # 1) Demography (split-isolation)
    model = SplitIsolationModel(N0=a.N_anc, N1=a.N1, N2=a.N2, T=a.t_split, m=a.m)

    # 2) Contig + tiled DFE (BGS)
    contig, intervals = make_contig_with_tiled_dfe(
        length=a.length, mu=a.mu, r=a.r, species=a.species, dfe_id=a.dfe,
        exon_bp=a.exon_bp, tile_bp=a.tile_bp,
    )

    # 3) Samples
    try:
        samples = {k.strip(): int(v) for k, v in (x.split(":") for x in a.samples.split(","))}
    except Exception as e:
        raise ValueError(f"Could not parse --samples '{a.samples}'. Use form 'YRI:20,CEU:20'.") from e

    # 4) SLiM simulation via stdpopsim engine
    engine = sps.get_engine("slim")
    ts = engine.simulate(
        model, contig, samples,
        seed=a.seed,
        slim_scaling_factor=a.slim_scaling,
        slim_burn_in=a.slim_burn_in,
    )

    # make an SFS and save it
    sfs = create_SFS(ts)
    sfs_file = Path(a.trees).with_suffix(".sfs.npy")
    np.save(sfs_file, sfs.data)   # save raw numpy array
    print(f"Wrote {sfs_file}")

    # 5) Ensure output dirs exist
    Path(a.trees).parent.mkdir(parents=True, exist_ok=True)
    if a.vcf:
        Path(a.vcf).parent.mkdir(parents=True, exist_ok=True)

    # 6) Write outputs
    ts.dump(a.trees)

    if a.vcf:
        with open(a.vcf, "w") as f:
            ts.write_vcf(f)

    outroot = Path(a.trees).with_suffix("")   # e.g., sims/out
    meta_path = outroot.with_suffix(".meta.json")
    bed_path  = outroot.with_suffix(".exons.bed")

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    bed_path.parent.mkdir(parents=True, exist_ok=True)

    meta = dict(
        model="split_isolation",
        params=dict(N_anc=a.N_anc, N1=a.N1, N2=a.N2, t_split=a.t_split, m=a.m),
        species=a.species,
        samples=a.samples,
        length=a.length, mu=a.mu, r=a.r,
        dfe=a.dfe, exon_bp=a.exon_bp, tile_bp=a.tile_bp,
        slim_scaling=a.slim_scaling, slim_burn_in=a.slim_burn_in,
        seed=a.seed, trees=a.trees, vcf=a.vcf,
        sites=int(ts.num_sites), sequence_length=int(ts.sequence_length),
    )
    meta_path.write_text(json.dumps(meta, indent=2))

    with open(bed_path, "w") as bed:
        for s, e in intervals:
            bed.write(f"chr1\t{s}\t{e}\n")

    # 7) Done
    print(f"Wrote {a.trees}; sites={ts.num_sites}; length={ts.sequence_length:.0f} bp")
    if a.vcf:
        print(f"Wrote {a.vcf}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {bed_path}")


if __name__ == "__main__":
    main()

# src/stdpopsim_wrappers.py
"""
stdpopsim demographic model wrappers for SLiM simulations.
Each model is defined as a subclass of stdpopsim.DemographicModel, which wraps an msprime.Demography object.
"""
from typing import Dict
import demes
import msprime
import stdpopsim as sps
import numpy as np


class _ModelFromDemes(sps.DemographicModel):
    """Wrap a demes.Graph so stdpopsim engines can simulate it (bottleneck, drosophila)."""

    def __init__(
        self,
        g: demes.Graph,
        model_id: str = "custom_from_demes",
        desc: str = "custom demes",
    ):
        model = msprime.Demography.from_demes(g)
        super().__init__(
            id=model_id,
            description=desc,
            long_description=desc,
            model=model,
            generation_time=1,
        )


# Leaf-first stdpopsim models for SLiM (avoid p0=ANC extinction at split)
class _IM_Symmetric(sps.DemographicModel):
    """
    Isolation-with-migration, symmetric: YRI <-> CEU with rate m; split at time T from ANC.
    Populations are added as leaves first so p0/p1 are YRI/CEU (not ANC), avoiding zero-size errors.
    """

    def __init__(self, N0, N1, N2, T, m):
        dem = msprime.Demography()
        dem.add_population(name="YRI", initial_size=float(N1))
        dem.add_population(name="CEU", initial_size=float(N2))
        dem.add_population(name="ANC", initial_size=float(N0))
        m = float(m)
        dem.set_migration_rate(source="YRI", dest="CEU", rate=m)
        dem.set_migration_rate(source="CEU", dest="YRI", rate=m)
        dem.add_population_split(time=float(T), ancestral="ANC", derived=["YRI", "CEU"])
        super().__init__(
            id="IM_sym",
            description="Isolation-with-migration, symmetric",
            long_description="ANC splits at T into YRI and CEU; symmetric migration m.",
            model=dem,
            generation_time=1,
        )


class _IM_Asymmetric(sps.DemographicModel):
    """Isolation-with-migration, asymmetric: YRI→CEU rate m12; CEU→YRI rate m21."""

    def __init__(self, N0, N1, N2, T, m12, m21):
        dem = msprime.Demography()

        # ✅ Add leaves first so that p0/p1 are extant pops, not ANC
        dem.add_population(name="YRI", initial_size=float(N1))
        dem.add_population(name="CEU", initial_size=float(N2))
        dem.add_population(name="ANC", initial_size=float(N0))

        # asymmetric migration
        # Forward-time: m12 = YRI→CEU, m21 = CEU→YRI
        # Backward-time encoding for msprime:
        dem.set_migration_rate(
            source="CEU", dest="YRI", rate=float(m12)
        )  # encode YRI→CEU
        dem.set_migration_rate(
            source="YRI", dest="CEU", rate=float(m21)
        )  # encode CEU→YRI

        # split backward in time
        dem.add_population_split(time=float(T), ancestral="ANC", derived=["YRI", "CEU"])

        super().__init__(
            id="IM_asym",
            description="Isolation-with-migration, asymmetric",
            long_description=(
                "ANC splits at T into YRI and CEU; asymmetric migration m12 and m21."
            ),
            model=dem,
            generation_time=1,
        )


class _Bottleneck(sps.DemographicModel):
    """
    Single-population bottleneck implemented directly in msprime.Demography.
    Times in generations before present (t_start > t_end >= 0).
    """

    def __init__(
        self, N0, N_bottleneck, N_recover, t_bottleneck_start, t_bottleneck_end
    ):
        t_start = float(t_bottleneck_start)
        t_end = float(t_bottleneck_end)
        if not (t_start > t_end >= 0):
            raise ValueError("Require t_bottleneck_start > t_bottleneck_end >= 0.")

        dem = msprime.Demography()
        dem.add_population(name="ANC", initial_size=float(N0))

        # At t_start, drop to the bottleneck size
        dem.add_population_parameters_change(
            time=t_start, population="ANC", initial_size=float(N_bottleneck)
        )
        # At t_end, recover to N_recover (constant to present)
        dem.add_population_parameters_change(
            time=t_end, population="ANC", initial_size=float(N_recover)
        )

        dem.sort_events()

        super().__init__(
            id="bottleneck",
            description="Single-population bottleneck (N0 → N_bottleneck → N_recover).",
            long_description=(
                "One population with ancestral size N0 until t_bottleneck_start, "
                "then a bottleneck of size N_bottleneck until t_bottleneck_end, "
                "then constant size N_recover to the present."
            ),
            model=dem,
            generation_time=1,
        )


class _DrosophilaThreeEpoch(sps.DemographicModel):
    """
    Two-pop Drosophila-style three-epoch model.

    ANC (size N0) splits at T_AFR_EUR_split into:
      - AFR: constant size AFR (AFR_recover in your priors)
      - EUR: bottleneck of size EUR_bottleneck until T_EUR_expansion,
             then recovery to EUR_recover up to the present.

    Populations are added leaf-first so p0/p1 are AFR/EUR (not ANC),
    which plays nicely with SLiM’s population ordering.
    """

    def __init__(
        self,
        N0,
        AFR,
        EUR_bottleneck,
        EUR_recover,
        T_AFR_EUR_split,
        T_EUR_expansion,
    ):
        T_split = float(T_AFR_EUR_split)
        T_exp = float(T_EUR_expansion)

        dem = msprime.Demography()

        # Leaf-first: extant pops first, then ANC
        dem.add_population(name="AFR", initial_size=float(AFR))
        dem.add_population(name="EUR", initial_size=float(EUR_bottleneck))
        dem.add_population(name="ANC", initial_size=float(N0))

        # EUR expansion (bottleneck -> recovery) at T_EUR_expansion
        dem.add_population_parameters_change(
            time=T_exp,
            population="EUR",
            initial_size=float(EUR_recover),
        )

        # Split backward in time at T_AFR_EUR_split: AFR/EUR merge into ANC
        dem.add_population_split(
            time=T_split,
            ancestral="ANC",
            derived=["AFR", "EUR"],
        )

        super().__init__(
            id="drosophila_three_epoch",
            description="Drosophila-style three-epoch AFR/EUR model",
            long_description=(
                "ANC (N0) until T_AFR_EUR_split, then split into AFR and EUR. "
                "AFR stays at AFR; EUR has a bottleneck (EUR_bottleneck) and "
                "expands at T_EUR_expansion to EUR_recover."
            ),
            model=dem,
            generation_time=1,
        )


class _SplitMigrationGrowth(sps.DemographicModel):
    """
    Custom model: CO/FR split from ANC, with growth in FR and asymmetric migration.
    """

    def __init__(self, N_CO, N_FR1, G_FR, N_ANC, m_CO_FR, m_FR_CO, T):
        # N_CO:    Effective population size of the Congolese population (constant).
        # N_FR1:   Effective population size of the French population at present (time 0).
        # G_FR:    Growth rate of the French population (exponential).
        # N_ANC:   Ancestral population size (before split).
        # m_CO_FR: Migration rate FROM Congolese TO French (forward time).
        #          (Fraction of French population replaced by Congolese migrants per generation).
        # m_FR_CO: Migration rate FROM French TO Congolese (forward time).
        #          (Fraction of Congolese population replaced by French migrants per generation).
        # T:       Time of split (generations ago).

        demogr = msprime.Demography()
        demogr.add_population(name="CO", initial_size=float(N_CO))
        demogr.add_population(
            name="FR", initial_size=float(N_FR1), growth_rate=float(G_FR)
        )
        demogr.add_population(name="ANC", initial_size=float(N_ANC))

        # Migration Matrix M[j, k] is rate of lineages moving from j to k (backward time).
        # Lineage j->k (backward) implies Gene Flow k->j (forward).
        # We want m_CO_FR to be Forward CO->FR. This implies Lineages FR->CO.
        # So M[FR, CO] should be m_CO_FR. (Indices: CO=0, FR=1, ANC=2).
        # M[1, 0] = m_CO_FR.
        #
        # We want m_FR_CO to be Forward FR->CO. This implies Lineages CO->FR.
        # So M[CO, FR] should be m_FR_CO.
        # M[0, 1] = m_FR_CO.

        demogr.migration_matrix = np.array(
            [[0, float(m_FR_CO), 0], [float(m_CO_FR), 0, 0], [0, 0, 0]]
        )
        demogr.add_population_split(
            time=float(T), derived=["CO", "FR"], ancestral="ANC"
        )

        super().__init__(
            id="split_migration_growth",
            description="Split with migration and growth (CO/FR)",
            long_description="Custom model with CO/FR split, FR growth, and asymmetric migration.",
            model=demogr,
            generation_time=1,
        )


def define_sps_model(
    model_type: str, g: demes.Graph, sampled_params: Dict[str, float]
) -> sps.DemographicModel:
    """Create appropriate stdpopsim model for SLiM based on model type."""
    if model_type == "split_isolation":
        # Symmetric migration model - extract parameters
        N0 = float(sampled_params.get("N_anc", sampled_params.get("N0")))
        N1 = float(sampled_params.get("N_YRI", sampled_params.get("N1")))
        N2 = float(sampled_params.get("N_CEU", sampled_params.get("N2")))
        T = float(sampled_params.get("T_split", sampled_params.get("t_split")))
        m_keys = ["m", "m_sym", "m12", "m21", "m_YRI_CEU", "m_CEU_YRI"]
        vals = [float(sampled_params[k]) for k in m_keys if k in sampled_params]
        m = float(np.mean(vals)) if vals else 0.0
        return _IM_Symmetric(N0, N1, N2, T, m)

    elif model_type == "split_migration":
        # Asymmetric migration model - extract parameters
        N0 = float(sampled_params.get("N_anc", sampled_params.get("N0")))
        N1 = float(sampled_params.get("N_YRI", sampled_params.get("N1")))
        N2 = float(sampled_params.get("N_CEU", sampled_params.get("N2")))
        T = float(sampled_params.get("T_split", sampled_params.get("t_split")))
        m12 = float(
            sampled_params.get(
                "m_YRI_CEU", sampled_params.get("m12", sampled_params.get("m", 0.0))
            )
        )
        m21 = float(
            sampled_params.get(
                "m_CEU_YRI", sampled_params.get("m21", sampled_params.get("m", 0.0))
            )
        )
        return _IM_Asymmetric(N0, N1, N2, T, m12, m21)

    elif model_type == "bottleneck":
        # Single-population bottleneck model - extract parameters
        N0 = float(sampled_params["N0"])
        N_bottleneck = float(sampled_params["N_bottleneck"])
        N_recover = float(sampled_params["N_recover"])
        t_bottleneck_start = float(sampled_params["t_bottleneck_start"])
        t_bottleneck_end = float(sampled_params["t_bottleneck_end"])
        return _Bottleneck(
            N0, N_bottleneck, N_recover, t_bottleneck_start, t_bottleneck_end
        )

    elif model_type == "drosophila_three_epoch":
        # Two-pop Drosophila three-epoch model
        N0 = float(sampled_params["N0"])
        AFR = float(sampled_params["AFR"])
        EUR_bottleneck = float(sampled_params["EUR_bottleneck"])
        EUR_recover = float(sampled_params["EUR_recover"])
        T_split = float(sampled_params["T_AFR_EUR_split"])
        T_EUR_exp = float(sampled_params["T_EUR_expansion"])

        return _DrosophilaThreeEpoch(
            N0,
            AFR,
            EUR_bottleneck,
            EUR_recover,
            T_split,
            T_EUR_exp,
        )

    elif model_type == "split_migration_growth":
        N_CO = float(sampled_params.get("N_CO", sampled_params.get("N1")))
        N_FR1 = float(sampled_params.get("N_FR1", sampled_params.get("N2")))
        N_ANC = float(sampled_params.get("N_ANC", sampled_params.get("N0")))
        m_CO_FR = float(sampled_params.get("m_CO_FR", 0.0))
        m_FR_CO = float(sampled_params.get("m_FR_CO", 0.0))
        T = float(sampled_params.get("T", sampled_params.get("T_split")))

        if "G_FR" in sampled_params:
            G_FR = float(sampled_params["G_FR"])
        elif "N_FR0" in sampled_params:
            N_FR0 = float(sampled_params["N_FR0"])
            # G = ln(N(0)/N(T)) / T
            G_FR = np.log(N_FR1 / N_FR0) / T
        else:
            G_FR = 0.0

        return _SplitMigrationGrowth(N_CO, N_FR1, G_FR, N_ANC, m_CO_FR, m_FR_CO, T)

    elif model_type == "OOA_three_pop":
        # Use the demes graph we already built and just wrap it.
        return _ModelFromDemes(
            g,
            model_id="OOA_three_pop",
            desc="Three-pop Out-of-Africa (YRI–CEU–CHB, Gutenkunst-style)",
        )

    else:
        # For bottleneck or any other demes-based custom model
        return _ModelFromDemes(g, model_id=f"custom_{model_type}", desc="custom demes")

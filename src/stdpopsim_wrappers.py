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


def define_sps_model(
     g: demes.Graph
) -> sps.DemographicModel:
        return _ModelFromDemes(g, model_id=f"custom_model", desc="custom demes")

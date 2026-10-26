"""
Where a simulation's MomentsLD outputs live, given the experiment config.

Pruning (optional, one level at a time) thins each LD window before LD stats
are computed. Everything derived from the thinned windows -- LD stats,
means/varcovs, optimizer fits -- goes under MomentsLD/pruning/<tag>/ so pruned
and unpruned results never mix:

    prune_mode   prune_keep_values   tag       MomentsLD subdir
    off / null   -                   None      MomentsLD
    "fraction"   [0.15]              thin15    MomentsLD/pruning/thin15
    "count"      [5000]              n5000     MomentsLD/pruning/n5000

fraction keeps that share of sites; count keeps min(n_sites, N) (see
src/prune_vcf.py). Shared by rules/simulation_rules.smk and
src/feature_extraction_helpers.py.
"""
from __future__ import annotations


def prune_setting(cfg: dict) -> tuple[str, float] | None:
    """(mode, value) when pruning is on, else None."""
    mode = cfg.get("prune_mode") or "off"
    if mode == "off":
        return None
    if mode not in ("fraction", "count"):
        raise ValueError(f'prune_mode must be "off", "fraction" or "count", got {mode!r}')
    values = cfg.get("prune_keep_values") or []
    if len(values) != 1:
        raise ValueError(
            f"prune_keep_values must hold exactly one value when pruning is on, got {values!r}"
        )
    return mode, float(values[0])


def prune_tag(cfg: dict) -> str | None:
    """Folder tag for the active pruning setting (thinNN / nNNNN), or None."""
    setting = prune_setting(cfg)
    if setting is None:
        return None
    mode, value = setting
    return f"thin{round(value * 100):02d}" if mode == "fraction" else f"n{int(value)}"


def prune_flag(cfg: dict) -> str:
    """src/prune_vcf.py argument for the active pruning setting ("" if off)."""
    setting = prune_setting(cfg)
    if setting is None:
        return ""
    mode, value = setting
    return f"--keep-fractions {value}" if mode == "fraction" else f"--keep-counts {int(value)}"


def momentsld_subdir(cfg: dict) -> str:
    """MomentsLD or MomentsLD/pruning/<tag>, relative to a sim's inference dir."""
    tag = prune_tag(cfg)
    return f"MomentsLD/pruning/{tag}" if tag else "MomentsLD"

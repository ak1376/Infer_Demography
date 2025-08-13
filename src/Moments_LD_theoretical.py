
from moments.LD.Demographics2D import snm

def split_asym_mig_MomentsLD(params, rho=None, theta=0.001, pop_ids=None):
    """
    Split into two populations of specifed size, which then have their own
    relative constant sizes and symmetric migration between populations.

    - nu1: Size of population 1 after split.
    - nu2: Size of population 2 after split.
    - T: Time in the past of split (in units of 2*Na generations)
    - m: Migration rate between populations (2*Na*m)

    :param params: The input parameters: (nu1, nu2, T, m)
    :param rho: Population-scaled recombination rate (4Nr),
        given as scalar or list of rhos.
    :type rho: float or list of floats, optional
    :param theta: Population-scaled mutation rate (4Nu). Defaults to 0.001.
    :type theta: float
    :param pop_ids: List of population IDs of length 1.
    :type pop_ids: lits of str, optional
    """
    nu1, nu2, m12, m21, T = params

    Y = snm(rho=rho, theta=theta)
    Y.integrate([nu1, nu2], T, rho=rho, theta=theta, m=[[0, m12], [m21, 0]])
    Y.pop_ids = pop_ids
    return Y

def drosophila_three_epoch_MomentsLD(
    params,
    rho=None,
    theta=0.001,
    pop_ids=("AFR", "EUR"),
):
    """
    Out-of-Africa three-epoch model for LD inference.

    Parameters
    ----------
    params : sequence
        (nu_afr, nu_eur_bot, nu_eur_mod,
         T_afr_exp, T_split, T_eur_exp,
         m12, m21, N0)

        • all nus are ratios to N0  
        • all times are already divided by 2·N0  
        • m12 / m21 are 2·N0·m

        The last element (N0) is *accepted* but not required inside the ODE
        – it is kept so you can write the best-fit back to file un-scaled.
    """
    (nu_afr,
     nu_eur_bot,
     nu_eur_mod,
     T_afr_exp,
     T_split,
     T_eur_exp,
     m12,
     m21,
     N0) = params

    # -------- 1  ancestral equilibrium (single pop, size ratio 1) ----------
    nu_anc = 1.0
    Y = Numerics.steady_state([nu_anc], rho=rho, theta=theta)
    Y = LDstats(Y, num_pops=1)

    # -------- 2  African expansion (still one pop) -------------------------
    dur_afr_exp = T_afr_exp - T_split
    if dur_afr_exp < 0:
        raise ValueError("Require T_afr_exp ≥ T_split")
    if dur_afr_exp > 0:
        Y.integrate([nu_afr], dur_afr_exp, rho=rho, theta=theta)

    # -------- 3  split AFR / EUR ------------------------------------------
    Y = Y.split(0)

    # -------- 4  European bottleneck epoch --------------------------------
    dur_eur_bot = T_split - T_eur_exp
    if dur_eur_bot < 0:
        raise ValueError("Require T_split ≥ T_eur_exp")
    if dur_eur_bot > 0:
        Y.integrate(
            [nu_afr, nu_eur_bot],
            dur_eur_bot,
            m=[[0, m12], [m21, 0]],
            rho=rho,
            theta=theta,
        )

    # -------- 5  modern epoch (EUR recovery) -------------------------------
    if T_eur_exp > 0:
        Y.integrate(
            [nu_afr, nu_eur_mod],
            T_eur_exp,
            m=[[0, m12], [m21, 0]],
            rho=rho,
            theta=theta,
        )

    Y.pop_ids = list(pop_ids)
    return Y
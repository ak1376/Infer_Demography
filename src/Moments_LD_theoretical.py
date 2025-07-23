
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
    nu1, nu2, T, m12, m21 = params

    Y = snm(rho=rho, theta=theta)
    Y.integrate([nu1, nu2], T, rho=rho, theta=theta, m=[[0, m12], [m21, 0]])
    Y.pop_ids = pop_ids
    return Y



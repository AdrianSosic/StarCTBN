import numpy as np
import scipy.stats as sta
import matplotlib.pyplot as plt
from scipy.special import softmax
from ctbn import CTBN
from utils import fixed_sum_tuples


class Potts_CTBN(CTBN):
    """CTBN with Potts dynamics."""

    def __init__(self, n_states, beta, tau, obs_std, **kwargs):
        self.beta = beta
        self.tau = tau
        self.obs_std = obs_std
        CTBN.__init__(self, n_states=n_states, **kwargs)
        self._use_stats = True
        self._cache_crms()
        self._cache_stats_values()

    def crm(self, node, parents):
        # implements abstract method of CTBN
        return potts_crm(self.set2stats(parents), self.beta, self.tau)

    def crm_stats(self, stats):
        # implements abstract method of CTBN
        return potts_crm(stats, self.beta, self.tau)

    def obs_likelihood(self, Y, X):
        # implements method of CTBN
        return sta.norm.pdf(X, scale=self.obs_std, loc=Y)

    def obs_rvs(self, X):
        # implements method of CTBN
        return sta.norm.rvs(scale=self.obs_std, loc=X)

    def set2stats(self, states):
        # implements method of CTBN
        return np.bincount(states, minlength=self.n_states)

    def stats_values(self, n_nodes):
        # implements method of CTBN
        return np.array(list(fixed_sum_tuples(self.n_states, n_nodes)))

    @classmethod
    def combine_stats(cls, stats_set1, stats_set2):
        # implements method of CTBN
        return stats_set1 + stats_set2


def potts_crm(spin_counts, beta, tau):
    """
    Computes the conditional rate matrix of a node based on the parent spins using Potts dynamics.

    Parameters
    ----------
    spin_counts : 1-D array, dtype: int
        The nth element counts how many parents have spin n.

    beta : float
        Inverse Potts temperature.

    tau : float
        Potts rate scale.

    Returns
    -------
    out : 2-D array, shape: (S, S)
        Conditional rate matrix.
    """
    # compute transition rates to target spins
    rates = tau * softmax(beta * spin_counts)

    # construct rate matrix by copying the rates and setting the exit rates accordingly
    rates = np.tile(rates, [len(rates), 1])
    np.fill_diagonal(rates, 0)
    np.fill_diagonal(rates, -rates.sum(axis=1))

    # return the rate matrix
    return rates


if __name__ == '__main__':

    # network size
    N = 4

    # number of observations
    n_obs = 100

    # CTBN parameters
    ctbn_params = dict(
        n_states=5,
        adjacency=np.ones((N, N))-np.eye(N),
        beta=1,
        tau=1,
        T=10,
        obs_std=0.75,
    )

    # generate and simulate Potts network
    ctbn = Potts_CTBN(**ctbn_params)
    ctbn.simulate()
    ctbn.emit(n_obs)
    ctbn.plot_trajectory(kind='line'), plt.show()

    # inference
    ctbn.update_rho()
    ctbn.update_Q()
    ctbn.plot_trajectory(kind='line'), plt.show()

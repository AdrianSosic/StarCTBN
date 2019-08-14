import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sta
from src.ctbn import CTBN


class Glauber_CTBN(CTBN):
    """CTBN with Glauber dynamics."""

    def __init__(self, beta, tau, obs_std, **kwargs):
        """
        Parameters
        ----------
        beta : float
            Inverse Glauber temperature.

        tau: float
            Glauber rate scale.
        """
        self.beta = beta
        self.tau = tau
        self.obs_std = obs_std
        CTBN.__init__(self, n_states=2, **kwargs)
        self._use_stats = True
        self._cache_crms()
        self._cache_stats_values()

    def crm(self, node, parents):
        # implements abstract method of CTBN
        return glauber_crm(self.set2stats(parents), self.beta, self.tau)

    def crm_stats(self, stats):
        # implements method of CTBN
        return glauber_crm(stats, self.beta, self.tau)

    @classmethod
    def set2stats(cls, states):
        # implements method of CTBN
        return 2 * np.sum(states) - np.size(states)

    @classmethod
    def stats_values(cls, n_nodes):
        # implements method of CTBN
        return np.arange(-n_nodes, n_nodes + 1, 2)

    @classmethod
    def stats2inds(cls, n_nodes, stats):
        # implements method of CTBN
        return ((stats + n_nodes) / 2).astype(int)

    def obs_likelihood(self, Y, X):
        # implements method of CTBN
        return sta.norm.pdf(X, scale=self.obs_std, loc=Y)

    def obs_rvs(self, X):
        # implements method of CTBN
        return sta.norm.rvs(scale=self.obs_std, loc=X)

    @classmethod
    def combine_stats(cls, stats_set1, stats_set2):
        # implements method of CTBN
        return stats_set1 + stats_set2


def glauber_crm(sum_of_spins, beta, tau):
    """
    Computes the conditional rate matrix of a node based on the parent spins using Glauber dynamics.

    Parameters
    ----------
    sum_of_spins : int
        Sum of parent spins (down spin = -1, up spin = 1)

    beta : float
        (see constructor)

    tau : float
        (see constructor)

    Returns
    -------
    out : 2-D array, shape: (S, S)
        Conditional rate matrix.
    """
    tan = np.tanh(beta * sum_of_spins)
    rate_up = 0.5 * (1 + tan)
    rate_down = 0.5 * (1 - tan)
    return tau * np.array([[-rate_down, rate_down], [rate_up, -rate_up]])


if __name__ == '__main__':

    # network size
    N = 4

    # number of observations
    n_obs = 10

    # CTBN parameters
    ctbn_params = dict(
        adjacency=np.ones((N, N))-np.eye(N),
        beta=1,
        tau=1,
        T=10,
        obs_std=0.1,
    )

    # generate and simulate Glauber network
    ctbn = Glauber_CTBN(**ctbn_params)
    ctbn.simulate()
    ctbn.emit(n_obs)
    ctbn.plot_trajectory(kind='line'), plt.show()

    # inference
    ctbn.update_rho()
    ctbn.update_Q()
    ctbn.plot_trajectory(kind='line'), plt.show()

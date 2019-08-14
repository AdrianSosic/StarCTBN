import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sta
from ctbn import CTBN


class Glauber_CTBN(CTBN):
    """CTBN with Glauber dynamics."""

    def __init__(self, beta, tau, **kwargs):
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
        CTBN.__init__(self, n_states=2, **kwargs)
        self._use_stats = True
        self._cache_crms()
        self._cache_stats_values()

    def crm(self, node, parents):
        return self.glauber_crm(self._set2stats(parents), self.beta, self.tau)

    def crm_stats(self, stats):
        return self.glauber_crm(stats, self.beta, self.tau)

    @staticmethod
    def _set2stats(states):
        return 2 * np.sum(states) - np.size(states)

    @classmethod
    def _stats_values(cls, n_parents):
        return np.arange(-n_parents, n_parents+1, 2)

    def _stats2inds(self, n_parents, stats):
        return ((stats + n_parents) / 2).astype(int)

    @staticmethod
    def obs_likelihood(Y, X):
        return sta.norm.pdf(X, scale=obs_std, loc=Y)

    @staticmethod
    def obs_rvs(X):
        return sta.norm.rvs(scale=obs_std, loc=X)

    @staticmethod
    def _combine_stats(stats_set1, stats_set2):
        return stats_set1 + stats_set2

    @staticmethod
    def glauber_crm(sum_of_spins, beta, tau):
        """
        Computes the conditional rate matrix of a node based on the parent spins according to the Glauber dynamics.

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

    # observation model
    obs_std = 0.1
    obs_means = [0, 1]
    n_obs = 10

    # CTBN parameters
    ctbn_params = dict(
        adjacency=np.ones((N, N))-np.eye(N),
        beta=1,
        tau=1,
        T=10,
        init_state=None,
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

import numpy as np
import matplotlib.pyplot as plt
from potts import Potts_CTBN


class Glauber_CTBN(Potts_CTBN):
    """CTBN with Glauber dynamics."""

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        beta : float
            Inverse Glauber temperature.

        tau: float
            Glauber rate scale.
        """
        Potts_CTBN.__init__(self, n_states=2, **kwargs)

    def crm_stats(self, stats):
        # implements method of CTBN
        return glauber_crm(int(stats), self.beta, self.tau)

    @staticmethod
    def set2stats(states):
        # implements method of CTBN
        return 2 * np.sum(states) - np.size(states)

    @staticmethod
    def stats_values(n_nodes):
        # implements method of CTBN
        return np.arange(-n_nodes, n_nodes + 1, 2)[:, None]

    @staticmethod
    def stats2inds(n_nodes, stats):
        # implements method of CTBN
        return ((stats + n_nodes) / 2).astype(int)


def glauber_crm(sum_of_spins, beta, tau):
    """
    Computes the conditional rate matrix of a node based on the parent spins using Glauber dynamics.

    Parameters
    ----------
    sum_of_spins : int
        Sum of parent spins (down spin = -1, up spin = 1).

    beta : float
        Inverse Glauber temperature

    tau : float
        Glauber rate scale.

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

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class CTBN:
    """
    Base class for continuous-time Bayesian networks (CTBNs).

    Notation:
    N: number of nodes
    S: number of states (equal for all nodes)
    T: simulation horizon
    """

    def __init__(self, adjacency, n_states, T, crm_fun, init_state=None):
        """
        Parameters
        ----------
        adjacency : 2-D array, values in {0, 1}, shape: (N, N)
            Adjacency matrix of the CTBN.

        n_states : int
            Number of states (S) each node can take.

        T : float
            Simulation horizon.

        crm_fun : callable, two parameters
            Provides the conditional rate matrices (CRMs) for the CTBN. When called with parameters (n, pa),
            it returns a 2-D array representing the CRM of node "n" for the parent configuration "pa". "n" is an
            integer in [0, N] and "pa" is a 1-D array containing the parent states ordered from lowest to highest
            parent node number. Each entry in "pa" is an integer in [0, S].

        init_state : optional, {None, 1-D array}
            Initial state of the CTBN.
            * None: all initial node states are drawn uniformly at random
            * 1-D array of integers: specifies the initial states of all nodes
        """
        # store input attributes
        self.adjacency = adjacency
        self.n_states = n_states
        self.T = T
        self.crm_fun = crm_fun
        self.init_state = init_state

        # attributes to store sampled trajectory
        self._states = None
        self._switching_times = None

        # CRM caching
        self._crms = None
        self._crms_cached = False

    @property
    def adjacency(self):
        return self._adjacency

    @adjacency.setter
    def adjacency(self, x):
        # convert and assert adjacency matrix
        x = np.array(x)
        try:
            assert x.ndim == 2
            assert x.shape[0] == x.shape[1]
            assert np.all(np.isin(x, [0, 1]))
            assert np.all(np.diag(x) == 0)  # no self-links (i.e. no node is in its own parent set)
        except AssertionError:
            raise ValueError('invalid adjacency matrix')

        # store adjacency matrix and generate graph
        self._adjacency = np.array(x, dtype=bool)
        self._G = nx.DiGraph(x)

    @property
    def init_state(self):
        return self._init_state

    @init_state.setter
    def init_state(self, x):
        # convert and assert initial states
        if x is not None:
            x = np.array(x)
            try:
                assert x.ndim == 1
                assert len(x) == self.n_nodes
                assert x.dtype == int
            except AssertionError:
                raise ValueError('invalid initial states')

        # store initial state
        self._init_state = x

    @property
    def n_nodes(self):
        """Returns the number of nodes of the CTBN."""
        return self._G.number_of_nodes()

    def parents(self, i):
        """Returns the sorted list of parents of the given node."""
        return sorted(list(self._G.predecessors(i)))

    def children(self, i):
        """Returns the sorted list of children of the given node."""
        return sorted(list(self._G.successors(i)))

    def get_state(self, times):
        """
        Queries the state of the CTBN at different times along the current trajectory.

        Parameters
        ----------
        times : 1-D array, values in [0, T], shape: (M,)
            Contains M query times.

        Returns
        -------
        out : 2-D array, shape: (M, N)
            CTBN states at the given query times.
        """
        t = np.append(self._switching_times, self.T)
        y = np.vstack([self._states, self._states[-1]])
        return interp1d(t, y, axis=0, kind='previous')(times).astype(int)

    def get_crms(self, state):
        """
        Returns all conditional rate matrices for the given state of the CTBN.

        Parameters
        ----------
        state : 1-D array, dtype: int, shape: (N,)
            State of the CTBN for which the rate matrices shall be computed.

        Returns
        -------
        out : 3-D array, shape: (N, S, S)
            N conditional rate matrices, represented as a three-dimensional array.
        """
        return np.array([self.crm(i, state[self.parents(i)]) for i in range(self.n_nodes)])

    def get_rates(self, state, crms=None):
        """
        Returns the rates of all nodes for the given state of the CTBN.

        Parameters
        ----------
        state : 1-D array, dtype: int, shape: (N,)
            State of the CTBN for which the rates shall be computed.

        crms : (optional) 3-D array, shape: (N, S, S)
            Conditional rate matrices for the queried CTBN state.

        Returns
        -------
        out : 2-D array, shape: (N, S)
            N rate vectors, represented as a two-dimensional array.
        """
        if crms is None:
            crms = self.get_crms(state)
        return np.squeeze(np.take_along_axis(crms, state[:, None, None], axis=1))

    def crm(self, node, parent_state):
        """
        Returns the conditional rate matrix of a node for a given parent configuration either from cache or by
        calling the CRM function.

        Parameters
        ----------
        node : int
            (see crm_fun)

        parent_state : 1-D array
            (see crm_fun)

        Returns
        -------
        out : 2-D array, shape: (S, S)
            Conditional rate matrix.
        """
        if self._crms_cached:
            parent_state = tuple(parent_state) if len(parent_state) > 0 else slice(None)
            return self._crms[node][parent_state]
        else:
            return self.crm_fun(node, parent_state)

    def cache_crms(self):
        """
        Caches the conditional rate matrices of all nodes to avoid repeated calls of the CRM function using a
        generic caching strategy, where all possible CRMs of all nodes are evaluated and stored one after another.

        Side Effects
        ------------
        self._crms <-- list of numpy arrays containing the conditional rate matrices of all nodes
        """
        # create empty list to store all CRMs
        self._crms = []

        # iterate over all nodes
        for n in range(self.n_nodes):
            # get the parents of the node
            parents = self.parents(n)

            # if the node has no parents, simply store the node's single (unconditional) rate matrix and continue
            if not parents:
                self._crms.append(self.crm_fun(n, []))
                continue

            # create empty array of appropriate shape to store all conditional rate matrices of the node
            shape = (len(parents) + 2) * [self.n_states]
            self._crms.append(np.zeros(shape))

            # iterate over all parent state configurations
            for parent_state in np.ndindex(*shape[0:-2]):
                # store the CRM of the current parent configuration using the configuration index
                self._crms[n][parent_state] = self.crm_fun(n, parent_state)

        # indicate that CRMs have been cached
        self._crms_cached = True

    def simulate(self):
        """
        Samples a state trajectory using the Gillespie algorithm.

        Side Effects
        ------------
        self._switching_times <-- 1-D array, shape: (L,)
            Time instances representing the switching times of the CTBN state.

        self._states <-- 2-D array, dtype: int, shape: (L, N)
            Sequence of CTBN states (each state is activated at the corresponding switching time).
        """
        # get/sample initial state and save it as first state of the trajectory
        if self.init_state is not None:
            states = self.init_state
        else:
            states = np.random.randint(0, self.n_states, self.n_nodes)
        states = states[None, :]

        # initialize vector containing the switching times
        switching_times = np.array([0], dtype=float)

        # iterate until simulation horizon is exceeded
        while True:
            # get current trajectory state
            ctbn_state = states[-1].copy()

            # get current (exit-)rates of all nodes and compute global switching rate
            rates = self.get_rates(ctbn_state)
            exit_rates = -np.squeeze(np.take_along_axis(rates, ctbn_state[:, None], axis=1))
            switching_rate = exit_rates.sum()

            # randomly select switching node and draw switching time
            switching_node = np.random.choice(range(self.n_nodes), p=exit_rates/switching_rate)
            switching_time = switching_times[-1] + np.random.exponential(1/switching_rate)

            # stop if simulation horizon is exceeded
            if switching_time > self.T:
                break

            # switch state of selected node
            current_node_state = ctbn_state[switching_node]
            candidate_states = np.delete(range(self.n_states), current_node_state)
            candidate_rates = rates[switching_node, candidate_states]
            new_node_state = np.random.choice(candidate_states, p=candidate_rates/candidate_rates.sum())
            ctbn_state[switching_node] = new_node_state

            # save new state
            states = np.vstack([states, ctbn_state])
            switching_times = np.append(switching_times, switching_time)

        # store states and corresponding switching times
        self._states = states
        self._switching_times = switching_times

    def plot_trajectory(self, nodes=None, kind='image'):
        """
        Plots a generated trajectory.

        Parameters
        ----------
        nodes : optional, iterable
            Defines the subset of nodes whose state shall be plotted. Default: all nodes.

        kind : {'image', 'line'}
            Visualize the trajectory as a state matrix or as a set of line plots.
        """

        # select subset of nodes
        if nodes is None:
            nodes = range(self.n_nodes)
        states = self._states[:, nodes]

        # create plot
        if kind == 'image':
            plt.imshow(states.T, extent=[0, self.T, -0.5, self.n_nodes-0.5], aspect='auto')
            plt.ylabel('node')
            plt.xlabel('time')
        elif kind == 'line':
            fig, axs = plt.subplots(self.n_nodes)
            for n, ax in enumerate(axs):
                ax.step(self._switching_times, self._states[:, n])
        else:
            raise ValueError('unknown kind')


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
        crm_fun = lambda i, pa: self.glauber_crm(np.sum(pa), beta, tau)
        CTBN.__init__(self, n_states=2, crm_fun=crm_fun, **kwargs)

    def cache_crms(self):
        """
        Overwrites method in CTBN:
        Caches the conditional rate matrices by storing only one matrix per energy level.
        """
        # number of distinct CRMs = number of energy levels = maximum number of parents in the network + 1
        n_crms = self.adjacency.sum(axis=1) + 1

        # initialize empty array and store the different CRMs
        self._crms = np.zeros([n_crms, 2, 2])
        for s in range(n_crms):
            self._crms[s] = self.glauber_crm(s, self.beta, self.tau)

        # indicate that CRMs have been cached
        self._crms_cached = True

    def crm(self, node, parent_state):
        """
        Overwrites method in CTBN:
        Queries the conditional rate matrices of a node based on the energy level of its parents.
        """
        if self._crms_cached:
            return self._crms[np.sum(parent_state)]
        else:
            return self.crm_fun(node, parent_state)

    @staticmethod
    def glauber_crm(n_up_spins, beta, tau):
        """
        Computes the conditional rate matrix of a node based on the parent spins according to the Glauber dynamics.

        Parameters
        ----------
        n_up_spins : int
            Number of parents with upward spin.

        beta : float
            (see constructor)

        tau : float
            (see constructor)

        Returns
        -------
        out : 2-D array, shape: (S, S)
            Conditional rate matrix.
        """
        tan = np.tanh(beta * n_up_spins)
        rate_up = 0.5 * (1 + tan)
        rate_down = 0.5 * (1 - tan)
        return tau * np.array([[-rate_down, rate_down], [rate_up, -rate_up]])


if __name__ == '__main__':

    # network size
    N = 100

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
    ctbn.plot_trajectory(), plt.show()

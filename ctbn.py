import numpy as np
import scipy.stats as sta
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from utils import PiecewiseFunction


class CTBN:
    """
    Base class for continuous-time Bayesian networks (CTBNs).

    Notation:
    N: number of nodes
    S: number of states (equal for all nodes)
    T: simulation horizon
    """

    def __init__(self, adjacency, n_states, T, crm_fun, obs_model=None, init_state=None):
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

        obs_model : dict with fields {'rvs', likelihood'}
            'rvs' : callable, one parameter
                Generates a random observation of a CTBN state. When called with parameter X, where X is a 1-D array
                of shape (N,) containing integer values in [0, S] representing a CTBN state, it returns a
                corresponding (N,)-array containing noisy observations of each node's state.
            'likelihood' : callable, two parameters
                Computes the likelihood of a given observation. When called with parameters (Y, X), it returns a 2-D
                array containing the likelihood of the CTBN state observation "Y" for given CTBN states "X". "Y" is a
                1-D array of shape (N,) containing noisy observations, as generated through the field 'rvs'. "X" is a
                2-D array of shape (M, N) containing M different CTBN states. The (m,n)th element of the returned array
                contains the likelihood that the nth CTBN node generated observation Y[n] at state X[m,n].

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
        self.obs_model = obs_model
        self.init_state = init_state

        # attributes to store sampled trajectory
        self._states = None
        self._switching_times = None

        # CRM caching
        self._crms = None
        self._crms_cached = False

        # observations
        self.obs_times = None
        self.obs_vals = None

        # initialize marginal distributions (uniform distribution) and Lagrange multipliers (all ones)
        self.Q = [lambda t: np.squeeze(np.full([np.size(t), self.n_states], fill_value=1/self.n_states))] * self.n_nodes
        self._rho = [lambda t: np.squeeze(np.ones([np.size(t), self.n_states]))] * self.n_nodes

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

    def _node_id2parent_index(self, id, child):
        """
        Returns the parent index of a given node in a second node's parent set (provided that the first node is a
        parent of the second). That is, if pa = _node_id2parent_index(id, child), then parents(child)[pa] = id.

        Parameters
        ----------
        id : int
            Node that shall be search for in the parent set of child.

        child :
            Node whose parent set is queried.

        Returns
        -------
        out : int or None
            The parent index of node "id" in the parent set of "child", or "None" if "id" is not a parent of "child".
        """
        # get the parent set and conduct a sorted search
        parents = self.parents(child)
        candidate_index = np.searchsorted(parents, id)

        # if the candidate index is a match, return it, otherwise return None
        return candidate_index if candidate_index < len(parents) and parents[candidate_index] == id else None

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

    def emit(self, n_or_times):
        """
        Generates observations of the current trajectory.

        Parameters
        ----------
        n_or_times : int or 1-D array
            Number of observations to be generated or time instances at which the observations shall be generated.
            * int: emit specified number of observations at uniformly random time instances in [0, T]
            * 1-D array, values in [0, T]: emit emissions at specified time instances

        Side Effects
        ------------
        self.obs_times <-- 1-D array, shape: (M,)
            Stores the M observation times of the CTBN state.

        self.obs_vals <-- 2-D array, shape: (M, N)
            Represents the M noisy observations of the CTBN state.
        """
        # store observation times (if not provided, draw random time instances first)
        if isinstance(n_or_times, int):
            self.obs_times = np.sort(np.random.uniform(0, self.T, n_or_times))
        else:
            times = np.sort(n_or_times)
            assert times[0] > 0 and times[-1] < self.T
            self.obs_times = times

        # get the CTBN state at the observation times
        states = self.get_state(self.obs_times)

        # store emitted observations
        self.obs_vals = self.obs_model['rvs'](states)

    def weighted_rates(self, node, weights, keep_index=None):
        """
        Computes a weighted average of all conditional rate matrices of a node.

        Parameters
        ----------
        node : int
            Node whose CRMs shall be averaged.

        weights : P-D array (P = number of parents), shape: (S, S, ..., S)
            An array containing the weights for all possible parent configurations of the node. The (i,j,..)th element
            of the array contains the weight for the CRM assigned to the configuration where the first parent is in
            state i, the second parent is in state j, and so on.

        keep_index : optional, int
            If provided, the specified index is excluded from the summation and kept as an additional (first) index
            in the returned array.

        Returns
        -------
        out : 2-D array or 3-D array, shape: (S, S) or (S, S, S)
            Average rate matrix/matrices. If "keep_index" is some integer M, the averaging results obtained for
            different states of M-th parent are stored separately along the first dimension of the output array.
        """
        # array to store the averaging results
        rates = np.zeros([self.n_states] * 2) if keep_index is None else np.zeros([self.n_states] * 3)

        # full averaging over all parents:
        # iterate over all parent configurations and add the corresponding weighted CRMs
        if keep_index is None:
            for parent_conf, weight in np.ndenumerate(weights):
                crm = self.crm(node, parent_conf)
                rates += weight * crm

        # compute separate averages for each state of the parent that is located at index "keep_index"
        else:
            for parent_conf, weight in np.ndenumerate(weights):
                crm = self.crm(node, parent_conf)
                rates[parent_conf[keep_index]] += weight * crm

        # return the average rates
        return rates

    def expected_rates(self, node, t, separate=None):
        """
        Computes the "expected rate matrix" of a node at time t, given as the average of its CRMs weighted according to
        the current estimate of its parents' marginal state distributions at time t.

        Parameters
        ----------
        node : int
            Node for which the rates shall be computed.

        t : float, value in [0, T]
            Time instant at which the rates shall be computed.

        separate : optional, int
            Node ID. If provided, the expected rates are computed separately for all states of that node.

        Returns
        -------
        out : 2-D array or 3-D array, shape: (S, S) or (S, S, S)
            Expected rate matrix/matrices. If "separate" is provided, the expected rates obtained for different states
            of the corresponding node are stored separately along the first dimension of the output array.
        """
        # if the node has no parents, the expected CRM is just its (unconditional) rate matrix
        if self.parents(node) is None:
            return self.crm(node, ())

        # get the current estimate of the marginal state distributions of all parents at time t
        parents_marginals = [self.Q[p](t) for p in self.parents(node)]

        # the collection of weights is given by the Cartesian product of all marginals
        product_marginals = np.prod(np.ix_(*parents_marginals))

        # if the expected rates shall be computed separately for all states of a given parent node, exclude the
        # corresponding summation during the averaging procedure
        if separate is not None:
            index = self._node_id2parent_index(separate, node)
            if index is None:
                raise ValueError(f"node {separate} is not a parent of node {node}")
        else:
            index = None

        # average the CRMs according to their weights (= product of parent marginals)
        return self.weighted_rates(node, product_marginals, keep_index=index)

    def update_Q(self):
        """
        Updates the marginal state distributions of all nodes using the current estimate of the Lagrange multipliers
        self._rho by solving Equation (6) forward in time.

        Side Effects
        ------------
        self.Q <-- List of length N, containing callables that represent the marginal state distributions of the nodes.
            Each callable accepts a single parameter t and returns a 1-D array of shape (S,) that provides the state
            probabilities of the corresponding node at time t.
        """
        # TODO: add option to update all Q_n jointly / use updated marginals while iterating over nodes

        def d_Q_n_t(n, Q_n_t, t):
            """
            Implements the right hand side (RHS) of Equation (6)

            Parameters
            ----------
            n : int
                Node ID.

            Q_n_t : 1-D array, shape: (S,)
                Marginal state distribution of the considered node at time t.

            t : float, values in [0, T]
                Time instant at which the RHS of the equation shall be evaluated.

            Returns
            -------
            out : 1-D array, shape: (S,)
                Array containing the time derivative of the node's marginal distribution (= probability flow).
            """
            # get expected rates and Lagrange multipliers of the node
            rates = self.expected_rates(n, t)
            rho_n_t = self._rho[n](t)

            # first term of RHS
            tmp = (Q_n_t / rho_n_t)[:, None] * rates
            np.fill_diagonal(tmp, 0)
            inflow = rho_n_t * np.sum(tmp, axis=0)

            # second term of RHS
            tmp = rates * rho_n_t[None, :]
            np.fill_diagonal(tmp, 0)
            outflow = Q_n_t / rho_n_t * np.sum(tmp, axis=1)

            # add both terms
            return inflow - outflow

        # create empty list and use uniform distribution as initial distribution for all nodes
        Q = []
        Q_0 = np.full(self.n_states, fill_value=1/self.n_states)

        # iterate over all nodes and solve the ODE forward in time
        for n in range(self.n_nodes):
            Q_n = solve_ivp(lambda t, y: d_Q_n_t(n, y, t), [0, self.T], Q_0, dense_output=True).sol
            Q.append(lambda t: Q_n(t).T)  # TODO (optional): rearrange dimensions of Q

        # store the result
        self.Q = Q

    def update_rho(self):
        """
        Updates the Lagrange multipliers of all nodes using the current estimate of the marginal distributions self.Q
        by solving Equation (5) backward in time.

        Side Effects
        ------------
        self._rho <-- List of length N, containing callables that represent the Lagrange multipliers of the nodes.
            Each callable accepts a single parameter t and returns a 1-D array of shape (S,) that provides the
            transformed Lagrange multipliers of the corresponding node at time t.
        """

        def d_rho_n_t(n, rho_n_t, t):
            """
            Implements the right hand side (RHS) of Equation (5).

            Parameters
            ----------
            n : int
                Node ID.

            rho_n_t : 1-D array, shape: (S,)
                Lagrange multipliers of the considered node at time t.

            t : float, values in [0, T]
                Time instant at which the RHS of the equation shall be evaluated.

            Returns
            -------
            out : 1-D array, shape: (S,)
                Array containing the time derivative of the node's Lagrange multipliers.
            """
            rates = self.expected_rates(n, t)
            psi = np.diag(self.psi(n, t))
            return -(rates + psi) @ rho_n_t

        # time grid providing the intervals on which the computed piecewise rho functions are defined
        grid = np.r_[-np.inf, self.obs_times, self.T]

        # list to store the functions
        rho = []

        # iterate over all nodes
        for n in range(self.n_nodes):
            # list to store the function pieces of the function belonging to the current node, filled from right to left
            pieces = []

            # iterate backwards over all time intervals and observations at the end of the intervals
            # (for the rightmost interval, i.e. at the end of the simulation horizon, there is no observation)
            for t1, t2, y_n in zip(grid[-2::-1], grid[-1:0:-1], [None, *self.obs_vals[::-1, n]]):
                # solve only until t=0
                # Note: the grid ranges to -inf because the leftmost interval of rho has infinite support
                t1 = max(t1, 0)

                # set all Lagrange multipliers to 1 at the end of the simulation horizon;
                # for the remaining intervals, use the reset condition described the in paragraph below Equation (7)
                if y_n is None:
                    reset_value = np.ones(self.n_states)
                else:
                    reset_value = pieces[0](t2) * self.obs_model['likelihood'](y_n, range(self.n_states))
                    reset_value = reset_value / reset_value.min()  # renormalize for numerical stability

                # compute function piece and append it to the left side of the list
                f = solve_ivp(lambda t, y: d_rho_n_t(n, y, t), [t2, max(t1, 0)], reset_value, dense_output=True)
                assert f.status == 0
                pieces.insert(0, f.sol)

            # add final rho function of the current node to the list
            rho.append(PiecewiseFunction(grid, pieces, dims=self.n_states))

        # store the result
        self._rho = rho

    def psi(self, node, t):
        """
        Computes the auxiliary values psi (Page 5) for the dynamics of the Lagrange multipliers rho in Equation (5).

        Parameters
        ----------
        node : int
            Node ID

        t : float, values in [0, T]
            Time instant at which the auxiliary variables shall be computed.

        Returns
        -------
        out : 1-D array, shape: (S,)
            Psi vector of the given node at time t.
        """
        # initialize empty array to store the values for each state of the considered node
        psi = np.zeros(self.n_states)

        # get the children of the node
        children = self.children(node)

        # outer sum over all children
        for child in children:
            # get the current marginal estimates and Lagrange multipliers of the child node
            rho_c_t = self._rho[child](t)
            Q_c_t = self.Q[child](t)

            # compute the expected rates of the child node for all states of the target node
            rates = self.expected_rates(child, t, separate=node)

            # evaluate all sums on the right hand side of the equation that belong to the current child node
            psi += np.einsum('j,ijk,k->i', Q_c_t / rho_c_t, rates, rho_c_t)

        # return the auxiliary array
        return psi

    def plot_trajectory(self, nodes=None, kind='image'):
        """
        Plots a generated trajectory, the emitted observations, and the inferred posterior marginal state distributions.

        Notes
        -----
        * Observations and marginal distributions are only shown for kind='line'.
        * Marginal distributions are only shown for S=2.

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
            fig, axs = plt.subplots(len(nodes))
            t = np.linspace(0, self.T, 100)
            for n, ax in zip(nodes, axs):
                ax.step(self._switching_times, self._states[:, n], where='post')
                ax.plot(self.obs_times, self.obs_vals[:, n], 'rx')
                if self.n_states == 2:
                    ax.plot(t, self.Q[n](t)[:, 1], 'g:')
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
        crm_fun = lambda i, pa: self.glauber_crm(self._states2energy(pa), beta, tau)
        CTBN.__init__(self, n_states=2, crm_fun=crm_fun, **kwargs)

    def cache_crms(self):
        """
        Overwrites method in CTBN:
        Caches the conditional rate matrices by storing only one matrix per energy level.
        """
        # number of distinct CRMs = number of energy levels = 2 * maximum number of parents in the network + 1
        n_crms = 2 * self.adjacency.sum(axis=1).max() + 1

        # initialize empty array and store the different CRMs
        self._crms = np.zeros([n_crms, 2, 2])
        for i in range(n_crms):
            self._crms[i] = self.glauber_crm(self._cache_index2energy(i), self.beta, self.tau)

        # indicate that CRMs have been cached
        self._crms_cached = True

    def crm(self, node, parent_state):
        """
        Overwrites method in CTBN:
        Queries the conditional rate matrices of a node based on the energy level of its parents.
        """
        if self._crms_cached:
            cache_index = self._energy2cache_index(self._states2energy(parent_state))
            return self._crms[cache_index]
        else:
            return self.crm_fun(node, parent_state)

    @staticmethod
    def _states2energy(states):
        return 2 * np.sum(states) - np.size(states)

    def _energy2cache_index(self, energy):
        return energy + self.n_nodes - 1

    def _cache_index2energy(self, cache_index):
        return cache_index - self.n_nodes + 1

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
    obs_model = dict(
        likelihood=lambda Y, X: sta.norm.pdf(X, scale=obs_std, loc=Y),
        rvs=lambda X: sta.norm.rvs(scale=obs_std, loc=X)
    )

    # CTBN parameters
    ctbn_params = dict(
        adjacency=np.ones((N, N))-np.eye(N),
        beta=1,
        tau=1,
        T=10,
        init_state=None,
        obs_model=obs_model,
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

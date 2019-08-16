import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from tqdm import trange
from src.utils import PiecewiseFunction, transpose_callable, _to_tuple


class CTBN(ABC):
    """
    Abstract base class for continuous-time Bayesian networks (CTBNs).

    Notation:
    N: number of nodes
    S: number of states (equal for all nodes)
    T: simulation horizon

    The minimum requirement to instantiate and simulate a network is to implement the "crm" function, which defines the
    network's conditional rate matrices.

    For additional features, the following methods can be optionally implemented:

    Generating observations
    -----------------------
    * obs_rvs

    Computing marginal posterior state distributions
    ------------------------------------------------
    * obs_likelihood

    Efficient inference using summary statistics
    --------------------------------------------
    * crm_stats
    * set2stats
    * stats_values
    * combine_stats

    Efficient indexing of summary statistics
    ----------------------------------------
    * stats2inds
    """

    def __init__(self, adjacency, n_states, T, init_state=None, verbose=True):
        """
        Parameters
        ----------
        adjacency : 2-D array, values in {0, 1}, shape: (N, N)
            Adjacency matrix of the CTBN. The nth row of the matrix defines the parent set of the nth node.

        n_states : int
            Number of states (S) each node can take.

        T : float
            Simulation horizon.

        init_state : optional, {None, 1-D array}
            Initial state of the CTBN.
            * None: all initial node states are drawn uniformly at random
            * 1-D array of integers: specifies the initial states of all nodes
        """
        # store input
        self.adjacency = adjacency
        self.n_states = n_states
        self.T = T
        self.init_state = init_state
        self.verbose = verbose

        # variables to store sampled trajectory
        self._states = None
        self._switching_times = None

        # variables to store observations
        self.obs_times = None
        self.obs_vals = None

        # cache for various quantities
        self._use_stats = False
        self._cache = {'node_stats': np.array(self.stats_values(1))}  # TODO: only cache node_stats when use_stats=True

        # initialize marginal distributions (uniform distribution) and Lagrange multipliers (all ones)
        # TODO: write classes to handle these objects
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
        self._G = nx.DiGraph(x.T)

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

    @property
    def max_degree(self):
        """Returns the maximum number of parents in the CTBN."""
        return self.adjacency.sum(axis=1).max()

    def parents(self, node):
        """Returns the sorted list of parents of the given node."""
        return sorted(list(self._G.predecessors(node)))

    def children(self, node):
        """Returns the sorted list of children of the given node."""
        return sorted(list(self._G.successors(node)))

    def _node_id2parent_index(self, node, child):
        """
        Returns the parent index of a given node in a second node's parent set (provided that the first node is a
        parent of the second). That is, if pa = _node_id2parent_index(node, child), then parents(child)[pa] = node.

        Parameters
        ----------
        node : int
            Node that shall be search for in the parent set of "child".

        child :
            Node whose parent set is queried.

        Returns
        -------
        out : int or None
            The parent index of "node" in the parent set of "child", or "None" if "node" is not a parent of "child".
        """
        # get the parent set and conduct a sorted search
        parents = self.parents(child)
        candidate_index = np.searchsorted(parents, node)

        # if the candidate index is a match, return it, otherwise return None
        return candidate_index if candidate_index < len(parents) and parents[candidate_index] == node else None

    @classmethod
    def stats_values(cls, n_nodes):
        """
        Computes all possible summary statistics that can be produced by "n_nodes" nodes.

        Parameters
        ----------
        n_nodes : int
            Number of nodes.

        Returns
        -------
        out : iterable
            Collection of summary statistics.
        """
        raise NotImplementedError

    @classmethod
    def set2stats(cls, state_conf):
        """
        Converts a given state configuration of a set of nodes into the corresponding summary statistic.

        Parameters
        ----------
        state_conf : 1-D array, dtype: int, length: arbitrary
            Contains the states of a given node set.

        Returns
        -------
        out : 1-D array
            Summary statistic of the state configuration.
        """
        raise NotImplementedError

    def get_stats_values(self, n_nodes):
        """Returns all possible summary statistics for "n_nodes" nodes from cache (if available) or by computing them
        from scratch."""
        if 'stats_values' in self._cache:
            return self._cache['stats_values'][n_nodes]
        else:
            return self.stats_values(n_nodes)

    def _cache_stats_values(self):
        """Stores all possible summary statistics for all possible parent set sizes in the cache."""
        self._cache['stats_values'] = {p: self.stats_values(p) for p in range(self.max_degree+1)}

    @classmethod
    def combine_stats(cls, stats_set1, stats_set2):
        """Defines the operation to combine the summary statistics of two node sets."""
        raise NotImplementedError

    def stats2inds(self, n_nodes, stats):
        """Generic method to find the indices of a given set of summary statistics in the list of statistics
        associated with a given set size (typically the size of a parent set). The indices are determined by
        a simple comparison. For improved performance, the method should be overridden in the subclasses to exploit
        the specific structure of the class-specific list of statistics."""
        return np.argwhere(np.all(stats[:, None] == self.get_stats_values(n_nodes), axis=2))[:, 1]

    @abstractmethod
    def crm(self, node, parent_conf):
        """
        Computes the conditional rate matrix of a given node for a certain parent configuration.

        Parameters
        ----------
        node : int
            Node for which the CRM shall be computed.

        parent_conf : 1-D array, dtype: int
            State configuration of the node's parents. The states are ordered from lowest to highest parent node number.

        Returns
        -------
        out : 2-D array, shape: (S, S)
            Conditional rate matrix.
        """
        pass

    def crm_stats(self, parent_stats):
        # TODO: add node dependency
        """
        Computes the conditional rate matrix for a given parent configuration statistic.

        Parameters
        ----------
        parent_stats : float
            Summary statistic of the parents' state configuration.

        Returns
        -------
        out : 2-D array, shape: (S, S)
            Conditional rate matrix.
        """
        raise NotImplementedError

    @classmethod
    def obs_likelihood(cls, Y, X):
        """
        Computes the likelihood of a given CTBN state observation.

        Parameters
        ----------
        Y : 1-D array, shape: (N,)
            Noisy observation of a CTBN state (see obs_rvs).

        X : 2-D array, dtype: int, shape: (M, N)
            Contains M different CTBN states for which the likelihoods of the observation Y shall be computed.

        Returns
        -------
        out : 2-D array, shape: (M, N)
            Likelihoods of the observation. The (m,n)th element of the array contains the likelihood that the nth
            CTBN node generated observation Y[n] at state X[m,n].

        """
        raise NotImplementedError

    @classmethod
    def obs_rvs(cls, X):
        """
        Generates a random observation of a CTBN state.

        Parameters
        ----------
        X : 1-D array, dtype: int, shape: (N,)
            State of the CTBN.

        Returns
        -------
        out : 1-D array, shape: (N,)
            Noisy observation of the CTBN state. The nth element of the array is a noisy observation of the nth node.
        """
        raise NotImplementedError

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

    def get_all_crms(self, state):
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
        return np.array([self.get_crm(i, state[self.parents(i)]) for i in range(self.n_nodes)])

    def get_rates(self, state, crms=None):
        """
        Returns the rates of all nodes for the given state of the CTBN.

        Parameters
        ----------
        state : 1-D array, dtype: int, shape: (N,)
            State of the CTBN for which the rates shall be computed.

        crms : optional, 3-D array, shape: (N, S, S)
            Conditional rate matrices for the queried CTBN state.

        Returns
        -------
        out : 2-D array, shape: (N, S)
            N rate vectors, represented as a two-dimensional array.
        """
        if crms is None:
            crms = self.get_all_crms(state)
        return np.squeeze(np.take_along_axis(crms, state[:, None, None], axis=1))

    def get_crm(self, node, parent_conf):
        """
        Returns the conditional rate matrix of a node for a given parent configuration either from cache or by
        calling the CRM function.

        Parameters
        ----------
        node : int
            Node for which the CRM shall be returned.

        parent_conf : 1-D array, dtype: int
            State configuration of the node's parents. The states are ordered from lowest to highest parent node number.

        Returns
        -------
        out : 2-D array, shape: (S, S)
            Conditional rate matrix.
        """
        if self._use_stats:
            if 'crms_stats' in self._cache:
                return self._cache['crms_stats'][_to_tuple(self.set2stats(parent_conf))]
            else:
                return self.crm_stats(self.set2stats(parent_conf))
        else:
            if 'crms' in self._cache:
                parent_conf = tuple(parent_conf) if len(parent_conf) > 0 else slice(None)
                return self._cache['crms'][node][parent_conf]
            else:
                return self.crm(node, parent_conf)

    def _cache_crms(self):
        """
        Caches the conditional rate matrices of all nodes to avoid repeated calls of the CRM function. The method uses
        generic caching strategy, where all possible CRMs of all nodes are evaluated and stored one after another.

        Side Effects
        ------------
        if self._use_stats:
            self._cache['crms'] <-- List of numpy arrays containing the conditional rate matrices of all nodes. The
            nth numpy array in the list has P+2 dimensions, each of size S, where P is the number of parents of the nth
            node. Each of the first P dimensions corresponds to one parent of the node. These dimensions are ordered
            from lowest to highest parent node number. The last two dimensions store the corresponding CRMs.

        else:
            self.cache['crms_stats'] <-- Dict of numpy arrays containing all possible conditional rate matrices. The
            keys are the summary statistics of the parent configuration.
        """
        if self._use_stats:
            # compute the CRMs for all possible summary statistics and store them in a dictionary
            self._cache['crms_stats'] = {}
            all_stats_values = [s for p in range(self.max_degree+1) for s in self.stats_values(p)]
            for stat in all_stats_values:
                key = _to_tuple(stat)
                if key not in self._cache['crms_stats']:
                    self._cache['crms_stats'][key] = self.crm_stats(stat)

        else:
            # create empty list to store all CRMs
            crms = []

            # iterate over all nodes
            for n in range(self.n_nodes):
                # get the parents of the node
                parents = self.parents(n)

                # if the node has no parents, simply store the node's single (unconditional) rate matrix and continue
                if not parents:
                    crms.append(self.crm(n, []))
                    continue

                # create empty array of appropriate shape to store all conditional rate matrices of the node
                shape = (len(parents) + 2) * [self.n_states]
                crms.append(np.zeros(shape))

                # iterate over all parent state configurations
                for parent_conf in np.ndindex(*shape[0:-2]):
                    # store the CRM of the current parent configuration using the configuration index
                    crms[n][parent_conf] = self.crm(n, parent_conf)

            # store the CRMS in cache
            self._cache['crms'] = crms

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
            Represents the M noisy observations of the CTBN state at the observation times.
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
        self.obs_vals = self.obs_rvs(states)

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
            Average rate matrix/matrices. If "keep_index" is some integer M, the averaging results obtained for the
            different states of the M-th parent are stored separately along the first dimension of the output array.
        """
        # array to store the averaging results
        rates = np.zeros([self.n_states] * 2) if keep_index is None else np.zeros([self.n_states] * 3)

        # full averaging over all parents:
        # iterate over all parent configurations and add the corresponding weighted CRMs
        if keep_index is None:
            for parent_conf, weight in np.ndenumerate(weights):
                crm = self.get_crm(node, parent_conf)
                rates += weight * crm

        # compute separate averages for each state of the parent that is located at index "keep_index"
        else:
            for parent_conf, weight in np.ndenumerate(weights):
                crm = self.get_crm(node, parent_conf)
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
            return self.get_crm(node, ())

        # get the current estimate of the marginal state distributions of all parents at time t
        parents_marginals = [self.Q[p](t) for p in self.parents(node)]

        # if the expected rates shall be computed separately for all states of a given parent node, exclude the
        # corresponding summation during the averaging procedure
        if separate is not None:
            index = self._node_id2parent_index(separate, node)
            if index is None:
                raise ValueError(f"node {separate} is not a parent of node {node}")
        else:
            index = None

        # when statistics are available, compute the rates efficiently using a sum-product procedure
        if self._use_stats:
            return self._sum_product(parents_marginals, keep_index=index)
        else:
            # the full collection of weights is obtained through the Cartesian product of all marginals
            product_marginals = np.prod(np.ix_(*parents_marginals))

            # average the CRMs according to their weights (= product of parent marginals)
            return self.weighted_rates(node, product_marginals, keep_index=index)

    def _sum_product(self, marginals, keep_index=None):
        """
        Efficiently computes the expected rate matrix of a node for the given marginal distributions of its parents
        based on summary statistics using a sum-product procedure.

        Parameters
        ----------
        marginals : list containing 1-D arrays of shape (S,)
            Collection of marginal distributions of the parents' states. The ordering is arbitrary since the
            computation is based on summary statistics, assuming all parents are exchangeable.

        keep_index : optional, int
            If provided, the marginal distribution at the specified index is treated separately. The output array then
            contains an additional (first) index that allows to access the averaging results for all states of the
            associated parent.

        Returns
        -------
        out : 2-D array or 3-D array, shape: (S, S) or (S, S, S)
            Average rate matrix/matrices. If "keep_index" is some integer M, the averaging results obtained for the
            different states of the M-th parent are stored separately along the first dimension of the output array.
        """
        # if a node shall be treated separately, move it to the end of the computation chain
        if keep_index is not None:
            marginals[0], marginals[keep_index] = marginals[keep_index], marginals[0]

        # iterate along the computation chain in reverse order
        for p, marginal in zip(range(len(marginals), -1, -1), reversed(marginals)):

            # get all possible joint statistics of the remaining nodes in the computation chain
            others_stats = self.get_stats_values(p - 1)

            # if the last node is treated separately, stop and return the results for all states of that node separately
            if keep_index is not None and p == 1:
                # weigh the rates with the corresponding marginal state probabilities of the node and return the result
                return result_curr * marginal[:, None, None]

            # compute all possible stats combinations of the current node and the remaining nodes
            joint_stats = self.combine_stats(others_stats[:, None], self._cache['node_stats'])

            # when processing the first node (end of the chain), evaluate the CRMs for all stats values
            if p == len(marginals):
                # TODO: extend get_crm method to allow passing stats
                rates = np.array([[self._cache['crms_stats'][_to_tuple(x)] for x in y] for y in joint_stats])

            # otherwise:
            else:
                # find the correct indices of the joint statistics in the array computed in the previous iteration
                inds = self.stats2inds(p, joint_stats.reshape(-1, len(self._cache['node_stats'])))

                # extract the processed rates and reshape them like the joint statistics array
                rates = result_curr[inds].reshape([*joint_stats.shape[0:2], self.n_states, self.n_states])

            # factor in the marginal of the current node
            # (The variable "result_curr" contains the intermediate result that is obtained by factoring in the
            # marginals of all visited nodes in the computation chain. It stores separate results for all possible
            # statistics of the remaining nodes in the chain.)
            result_curr = np.einsum('ijkl,j->ikl', rates, marginal)

        # return the final rate matrix
        return result_curr[0]

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
            Implements the right hand side (RHS) of Equation (6).

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

        # progress bar
        if self.verbose:
            print("Update Q:")
        pbar = trange(self.n_nodes, disable=not self.verbose)

        # iterate over all nodes and solve the ODE forward in time
        for n in pbar:
            # show progress
            pbar.set_description(f"    Updating node {n}")

            Q_n = solve_ivp(lambda t, y: d_Q_n_t(n, y, t), [0, self.T], Q_0, dense_output=True)
            assert Q_n.status == 0
            Q.append(transpose_callable(Q_n.sol))  # TODO (optional): rearrange dimensions of Q

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

        # progress bar
        if self.verbose:
            print("Update rho:")
        pbar = trange(self.n_nodes, disable=not self.verbose)

        # iterate over all nodes
        for n in pbar:
            # show progress
            pbar.set_description(f"    Updating node {n}")

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
                    reset_value = pieces[0](t2) * self.obs_likelihood(y_n, range(self.n_states))
                    reset_value = reset_value / reset_value.mean()  # renormalize for numerical stability

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
            Node ID.

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

    def plot_trajectory(self, nodes=None, kind='image', n_points=100):
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

        n_points : int
            Number of interpolation points for line plot.
        """

        # select subset of nodes
        if nodes is None:
            nodes = range(self.n_nodes)
        states = self._states[:, nodes]

        # create image plot
        if kind == 'image':
            plt.imshow(states.T, extent=[0, self.T, -0.5, self.n_nodes-0.5], aspect='auto')
            plt.ylabel('node')
            plt.xlabel('time')

        # create line plot
        elif kind == 'line':
            fig, axs = plt.subplots(len(nodes))
            t = np.linspace(0, self.T, n_points)
            times = np.r_[self._switching_times, self.T]
            states = np.vstack([self._states, self._states[-1, :]])
            for n, ax in zip(nodes, axs):
                Q = self.Q[n](t)
                if self.n_states == 2:
                    ax.plot(t, Q[:, 1], 'g:')
                else:
                    for s in range(self.n_states):
                        ax.fill_between(t, s-0.5*Q[:, s], s+0.5*Q[:, s], alpha=0.5)
                ax.step(times, states[:, n], where='post', color='k')
                ax.plot(self.obs_times, self.obs_vals[:, n], 'kx')
                ax.set_xlim(0, self.T)
        else:
            raise ValueError('unknown kind')

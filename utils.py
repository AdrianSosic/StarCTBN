import numpy as np


class PiecewiseFunction:
    """
    Implements a one-dimensional piecewise function consisting of arbitrarily many intervals.

    When called with an array of function arguments, each array element will be assigned to its appropriate interval
    using a sorted search, where it is evaluated using the corresponding callable defining the local piece of the
    function.
    """

    def __call__(self, X):
        """
        Returns the function values for a given set of input values.

        Parameters
        ----------
        X : float or 1-D array of size M
            Input(s) to the function.

        Returns
        -------
        out : 1-D or 2-D array, shape: (D,) or (M, D)
            Array containing the function values for the given inputs.
        """
        # treat input always as array
        X = np.atleast_1d(X)

        # assert that all inputs lie in the defined interval range
        # if self.assign_right:
        #     assert np.all(self.grid[0] <= X) and np.all(X < self.grid[-1])
        # else:
        #     assert np.all(self.grid[0] < X) and np.all(X <= self.grid[-1])

        # create array to store the function values
        y = np.zeros((X.size, self.dims))

        # assign all inputs to their corresponding intervals
        if self.assign_right:
            inds = np.minimum(np.searchsorted(self.grid, X, side='right'), len(self.grid)) - 1
        else:
            inds = np.maximum(np.searchsorted(self.grid, X), 0) - 1
        inds = np.clip(inds, 0, np.size(self.grid)-2)

        # evaluate inputs one by one
        for i, (x, ind) in enumerate(zip(X, inds)):
            y[i] = self.funs[ind](x)

        # if input was a single number, convert output to 1-D array
        if X.size == 1:
            y = y[0]

        # return function values
        return y

    def __init__(self, grid, funs, assign_right=False, dims=None):
        """
        Creates a new piecewise function f:R->R^D consisting of N pieces.

        Parameters
        ----------
        grid : 1-D array, shape: (N+1,)
            Sorted array defining the interval boundaries of the piecewise function.

        funs : list of length N containing callables
            Defines the N pieces of the function.

        assign_right : boolean
            Defines whether input values that coincide with the grid points are assigned to the left or right interval.

        dims : optional, int
            Dimension D of the function's codomain. If not provided, it will be inferred by calling the first callable.
        """
        assert np.all(np.diff(grid) > 0)  # assert that the grid is sorted and contains unique values
        assert np.size(grid) == len(funs) + 1  # assert that there are as many functions as intervals
        self.grid = np.array(grid)
        self.funs = funs
        self.assign_right = assign_right
        self.dims = dims if dims is not None else np.size(funs[0](grid[0]))

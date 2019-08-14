# StarCTBN
A Python library for **approximate inference in continuous-time Bayesian networks (CTBNs)**. The library provides a computational framework for computing marginal posterior state distributions of a network's nodes from noisy time-coarse observation data using a **star approximation**. The underlying theory is described in [this paper](https://papers.nips.cc/paper/8013-cluster-variational-approximations-for-structure-learning-of-continuous-time-bayesian-networks-from-incomplete-data):

```
@incollection{NIPS2018_8013,
title = {Cluster Variational Approximations for Structure Learning of Continuous-Time Bayesian Networks from Incomplete Data},
author = {Linzner, Dominik and Koeppl, Heinz},
booktitle = {Advances in Neural Information Processing Systems 31},
pages = {7880--7890},
year = {2018}
}
```

## Requirements
* Tested with Python 3.6.8

## Example
To see an example, run [./src/glauber.py](./src/glauber.py).

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__) + "/../../src")
from potts import Potts_CTBN
np.random.seed(100)

# network size
N = 4

# number of observations
n_obs = 100

# CTBN parameters
ctbn_params = dict(
    n_states=5,
    adjacency=np.ones((N, N)) - np.eye(N),
    beta=0.5,
    tau=1,
    T=20,
    obs_std=0.75,
)

# generate and simulate Potts network
ctbn = Potts_CTBN(**ctbn_params)
ctbn.simulate()
ctbn.emit(n_obs)
ctbn.plot_trajectory(kind='line')
plt.gcf().set_size_inches((10, 3))
plt.show()

# inference
ctbn.update_rho()
ctbn.update_Q()
ctbn.plot_trajectory(kind='line')

# adjust and export plot
for ax in plt.gcf().get_axes():
    ax.set_xticks([])
    ax.set_yticks([])
plt.gcf().set_size_inches((10, 3))
plt.savefig('logo.png', dpi=400)
plt.show()
os.system('convert logo.png -trim logo.png')
os.system('convert logo.png -bordercolor White -border 20x20 logo.png')

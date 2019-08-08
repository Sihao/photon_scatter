from Objective import Objective
from Medium import Medium

import os
import numpy as np
import plotly
import utils


# Overwrite get_path_length() method in Medium
class _Medium(Medium):
    def get_path_length(self):
        return 10


# Scattering coefficient
# in um^-1
mu_s = 0.01070

# Absorption coefficient
mu_a = 0.0002

# Refractive indices
n_i = 1

# in air
n_e = 1

# Anisotropy factor
g = 1

sample_depth = 250
focus_depth = 170
working_distance = 5940
start_pos = np.array([0, 0, focus_depth])
NA = 0.54
distance_to_sample = working_distance + focus_depth

objective = Objective(NA, working_distance, sample_depth)

medium_shape = np.array([float('inf'), float('inf'), sample_depth])
medium = _Medium(medium_shape, mu_s, mu_a, n_i, n_e, g)

num_photons = 5000

photons = utils.multiple_sim(medium, start_pos, num_photons, single_step=True)

# Plot photon positions
fig = utils.plot_photons(photons, objective)

try:
    plotly.offline.plot(fig, filename='launch_direction/photon_positions.html')
except FileNotFoundError:
    os.mkdir('launch_direction')
    plotly.offline.plot(fig, filename='launch_direction/photon_positions.html')

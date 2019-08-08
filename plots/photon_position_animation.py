from Objective import Objective
from Medium import Medium

import os
import numpy as np
import plotly
import utils


# Scattering coefficient
# in um^-1
mu_s = 0.01070

# Absorption coefficient
mu_a = 0.0002

# Refractive indices
n_i = 1.4

# in air
n_e = 1

# Anisotropy factor
g = 0.94

sample_depth = 1000
focus_depth = 920
working_distance = 5940
start_pos = np.array([0, 0, focus_depth])
NA = 0.54
distance_to_sample = working_distance + focus_depth

objective = Objective(NA, working_distance, sample_depth)

medium_shape = np.array([float('inf'), float('inf'), sample_depth])
medium = Medium(medium_shape, mu_s, mu_a, n_i, n_e, g)

num_photons = 500

photons = utils.multiple_sim(medium, start_pos, num_photons)

# Plot photon positions
fig = utils.animate_photon_positions(photons, objective)

try:
    plotly.offline.plot(fig, filename='photon_position_animation/animation.html', auto_play=False)
except FileNotFoundError:
    os.mkdir('photon_position_animation')
    plotly.offline.plot(fig, filename='photon_position_animation/animation.html', auto_play=False)

from Objective import Objective
from Medium import Medium

import numpy as np
import os
import plotly
import utils


# Scattering coefficient
# in um^-1
mu_s = 0.01070

# Absorption coefficient
mu_a = 0.0002

# Refractive indices
# in tissue
n_i = 1.4

# in air
n_e = 1

# Anisotropy factor
# in white matter
g = 0.94

sample_depth = 250
focus_depth = 170
working_distance = 5940
NA = 0.54
fov = np.linspace(-2500, 2500, 3)

distance_to_sample = working_distance + focus_depth

num_photons = 1000

medium_shape = np.array([float('inf'), float('inf'), sample_depth])
medium = Medium(medium_shape, mu_s, mu_a, n_i, n_e, g)

objective = Objective(NA, working_distance, sample_depth)

photons = utils.fov_sim(medium, fov, num_photons, focus_depth, omit_bottom=True)

acceptance_matrix = utils.calc_acceptance_matrix(photons, objective)
fig = utils.plot_fov_heatmap(acceptance_matrix, fov)

try:
    plotly.offline.plot(fig, filename='fov_heatmap/heatmap.html')
except FileNotFoundError:
    os.mkdir('fov_heatmap')
    plotly.offline.plot(fig, filename='fov_heatmap/heatmap.html')

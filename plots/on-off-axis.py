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

sample_depth = 900
focus_depth = 450
working_distance = 5940
NA = 0.54
distance_to_sample = working_distance + focus_depth

objective = Objective(NA, working_distance, sample_depth)

medium_shape = np.array([float('inf'), float('inf'), sample_depth])
medium = Medium(medium_shape, mu_s, mu_a, n_i, n_e, g)

num_photons = 1000

start_positions = [
    np.array([0, 0, focus_depth]),
    np.array([-2500, -2500, focus_depth]),
    np.array([-2500, 2500, focus_depth]),
    np.array([2500, -2500, focus_depth]),
    np.array([2500, 2500, focus_depth])
]

for start_pos in start_positions:
    photons = utils.multiple_sim(medium, start_pos, num_photons, omit_bottom=True)

    # Plot photon positions
    fig = utils.plot_photons(photons, objective)

    try:
        plotly.offline.plot(fig, filename='on-off-axis/(%s,%s).html' % (start_pos[0], start_pos[1]))
    except FileNotFoundError:
        os.mkdir('on-off-axis')
        plotly.offline.plot(fig, filename='on-off-axis/(%s,%s).html' % (start_pos[0], start_pos[1]))

    # Plot axial paths
    fig = utils.plot_axial_paths(photons, medium, objective)

    try:
        plotly.offline.plot(fig, filename='on-off-axis/axial_path(%s,%s).html' % (start_pos[0], start_pos[1]))
    except FileNotFoundError:
        os.mkdir('on-off-axis')
        plotly.offline.plot(fig, filename='on-off-axis/axial_path(%s,%s).html' % (start_pos[0], start_pos[1]))

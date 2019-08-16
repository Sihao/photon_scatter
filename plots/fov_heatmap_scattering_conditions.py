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
n_i_list = [1.4]

n_e_list = [1.36, 1.4]

# Anisotropy factor
anisotropy_factors = [-1, -0.94, 0, 0.94, 1]

sample_depth = 900
focus_depth = 450
working_distance = 5940
NA = 0.54
distance_to_sample = working_distance + focus_depth


medium_shape = np.array([float('inf'), float('inf'), sample_depth])

num_photons = 1000

start_pos = np.array([0, 0, focus_depth])
fov = np.linspace(-2500, 2500, 10)

for g in anisotropy_factors:
    for n_i in n_i_list:
        for n_e in n_e_list:
            objective = Objective(NA, working_distance, sample_depth, n_e)

            medium = Medium(medium_shape, mu_s, mu_a, n_i, n_e, g)

            photons = utils.fov_sim(medium, fov, num_photons, focus_depth, omit_bottom=True)
            acceptance_matrix = utils.calc_acceptance_matrix(photons, objective)
            # Plot photon positions
            fig = utils.plot_fov_heatmap(acceptance_matrix, fov)

            try:
                plotly.offline.plot(fig, filename='fov_heatmap/n_i=%.2f-n_e=%.2f-g=%.2f.html' % (n_i, n_e, g))
            except FileNotFoundError:
                os.mkdir('fov_heatmap')
                plotly.offline.plot(fig, filename='fov_heatmap/n_i=%.2f-n_e=%.2f-g=%.2f.html' % (n_i, n_e, g))

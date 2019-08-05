from Objective import Objective
from Medium import Medium

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from itertools import compress
import plotly
import utils

mpl.use('Qt5Agg')

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
g = .94

sample_depth = 1000
focus_depth = 170
working_distance = 5940
NA = 0.54
fov = np.linspace(0, 0, 1)

distance_to_sample = working_distance + focus_depth

num_photons = 5000

medium_shape = np.array([float('inf'), float('inf'), sample_depth])
medium = Medium(medium_shape, mu_s, mu_a, n_i, n_e, g)

# xx, yy = np.meshgrid(range(-100, 100), range(-100, 100), sparse=True)
# z_1 = xx * 0 + yy * 0
# z_2 = xx * 0 + yy * 0 + 260

objective = Objective(NA, working_distance, sample_depth)

photons = utils.fov_sim(medium, fov, num_photons, focus_depth, omit_bottom=True)

fig = utils.plot_photons(photons[0][0], objective, show_aperture=True)
plotly.offline.plot(fig, filename='file_1.html')

fig = utils.plot_photon_path(photons[0][0][0])
plotly.offline.plot(fig, filename='file.html')

acceptance_matrix = utils.calc_acceptance_matrix(photons, objective)
fig = utils.plot_fov_heatmap(acceptance_matrix, fov)
plotly.offline.plot(fig, filename='file2.html')

# sns.jointplot(x=positions[:, 0], y=positions[:, 1])
# sns.jointplot(x=positions[:, 0], y=positions[:, 2])
#

#
# plt.show()

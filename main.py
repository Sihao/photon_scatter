from Medium import Medium
from Photon import Photon

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def run_sim(mu_s, n_i, n_e, g, num_photons):
    # Start position
    start_pos = np.array([0, 0, 0])

    medium = Medium(medium_shape, mu_s, n_i, n_e, g)

    photon_positions = np.array([0, 0, 0])

    for i in range(num_photons):
        photon = Photon(start_pos, medium)

        while photon.is_propagating:
            photon.propagate()
        photon_positions = np.vstack((photon_positions, photon.current_pos))

    print(photon_positions)
    photon_positions = np.delete(photon_positions, 0, 0)

    return photon_positions


mpl.use('Qt5Agg')

medium_shape = np.array([float('inf'), float('inf'), 260])

# Scattering coefficient
# in mm^-1
mu_s_list = [1070, 1299, 1430]

# Refractive indices
# in tissue
n_i = 1.4

# in gel
n_e = 1.34

# Anisotropy factor
# in white matter
g = 0.94



# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(photon_positions[:, 0], photon_positions[:, 1], photon_positions[:, 2], s=5)
# # ax.plot(photon_positions[:, 0], photon_positions[:, 1], photon_positions[:, 2])
# plt.show()


for i, mu_s in enumerate(mu_s_list):
    photon_positions = run_sim(mu_s, n_i, n_e, g, 2000)

    sns.jointplot(x=photon_positions[:, 0], y=photon_positions[:, 1], kind='kde')


plt.show()
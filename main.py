from Medium import Medium
from Photon import Photon

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def run_sim(mu_s, mu_a, n_i, n_e, g, num_photons):
    # Start position
    start_pos = np.array([0, 0, 0])

    medium_shape = np.array([float('inf'), float('inf'), 500])

    medium = Medium(medium_shape, mu_s, mu_a, n_i, n_e, g)

    photon_positions = np.array([0, 0, 0])

    for i in range(num_photons):
        photon = Photon(start_pos, medium)

        while photon.is_propagating and not photon.is_absorbed:
            photon.propagate()
        # print('Path length: %f' %photon.total_path)

        if not photon.is_absorbed:
            photon_positions = np.vstack((photon_positions, photon.path[-1]))

    photon_positions = np.delete(photon_positions, 0, 0)

    return photon_positions, photon


def parallel_sim(mu_s, mu_a, n_i, n_e, g, num_photons):
    result = [single_sim(mu_s, mu_a, n_i, n_e, g) for _ in range(num_photons)]

    return result


def single_sim(mu_s, mu_a, n_i, n_e, g):
    # Start position
    start_pos = np.array([0, 0, 750])
    medium_shape = np.array([float('inf'), float('inf'), 1000])

    medium = Medium(medium_shape, mu_s, mu_a, n_i, n_e, g)

    photon = Photon(start_pos, medium)

    while photon.is_propagating and not photon.is_absorbed:
        photon.propagate()
    # print('Path length: %f' %photon.total_path)

    if not photon.is_absorbed:
        return photon
    else:
        return None


mpl.use('Qt5Agg')


# Scattering coefficient
# in um^-1
mu_s = 0.01070

# Absorption coefficient
mu_a = 0.00001

# Refractive indices
# in tissue
n_i = 1.4

# in gel
n_e = 1.34

# Anisotropy factor
# in white matter
g = 0.94
photons = parallel_sim(mu_s, mu_a, n_i, n_e, g, 1000)

positions = np.array([photon.path[-1] for photon in photons])
print(positions.shape)

xx, yy = np.meshgrid(range(-100, 100), range(-100, 100), sparse=True)
z_1 = xx * 0 + yy * 0
z_2 = xx * 0 + yy * 0 + 260

# Scatter plot exited photons
fig_1 = plt.figure()
ax_1 = plt.axes(projection='3d')
ax_1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=5)
# ax.plot(photon_positions[:, 0], photon_positions[:, 1], photon_positions[:, 2])

sns.jointplot(x=positions[:, 0], y=positions[:, 1])
sns.jointplot(x=positions[:, 0], y=positions[:, 2])

fig_2 = plt.figure()
ax_2 = plt.axes(projection='3d')
ax_2.scatter(photons[0].path[:, 0], photons[0].path[:, 1], photons[0].path[:, 2], s=5)
ax_2.plot(photons[0].path[:, 0], photons[0].path[:, 1], photons[0].path[:, 2])

# plt.figure()
# theta_initials = np.array([theta for photon in photons for theta in photon.thetas])
# sns.distplot(theta_initials)

plt.show()

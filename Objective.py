import numpy as np


class Objective:
    def __init__(self, numerical_aperture, working_distance, refractive_index=1):
        self.n = refractive_index
        self.NA = numerical_aperture
        self.working_distance = working_distance

        # Computed properties
        self.front_aperture = self.NA * 2 * self.working_distance
        self.theta = np.arcsin(self.NA / self.n)

    def photon_accepted(self, photon):
        # Determine Z-position of aperture opening
        aperture_z = np.cos(self.theta) * self.working_distance


        # Only consider photons that didn't exit the bottom
        if photon.current_pos[2] > 0:
            # Compute (X, Y) position of photon at height of aperture opening
            photon_x = ((aperture_z - photon.current_pos[2]) / photon.mu_z) * photon.mu_x + photon.current_pos[0]
            photon_y = ((aperture_z - photon.current_pos[2]) / photon.mu_z) * photon.mu_y + photon.current_pos[1]

            # Assume aperture opening is a circle centred at (0, 0, aperture_z)
            # Check if computed photon position is within this circle
            if photon_x ** 2 + photon_y ** 2 < (self.front_aperture / 2) ** 2:
                return True
            else:
                return False
        else:
            return False

import numpy as np


class Objective:
    """
    Class defining properties and methods for Objective object
    """
    def __init__(self, numerical_aperture, working_distance, sample_thickness, refractive_index=1):
        """
        :param numerical_aperture: Numerical aperture of the objective
        :param working_distance: Working distance of the objective
        :param sample_thickness: Thickness of the sample being imaged in microns.
        :param refractive_index: Refractive index of the medium that the objective is in. Default `refractive_index=`1
               for air
        """
        self.n = refractive_index
        self.NA = numerical_aperture
        self.working_distance = working_distance
        self.sample_thickness = sample_thickness

        # Computed properties
        self.front_aperture = self.NA * 2 * self.working_distance
        self.theta = np.arcsin(self.NA / self.n)

    def photon_accepted(self, photon):
        """
        Determine whether a photon is absorbed based on its position, direction and the properties of the objective
        :param photon: Photon object
        :return: `True` if the photon is accepted and `False` if it is not accepted.
        """
        # Determine Z-position of aperture opening
        aperture_z = np.cos(self.theta) * self.working_distance + self.sample_thickness

        # Only consider photons that didn't exit the bottom
        if photon.mu_z > 0:
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

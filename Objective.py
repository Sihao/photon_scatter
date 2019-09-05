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

    def theoretical_collection_efficiency(self, deviation, type='sphere'):
        """
        Calcute the theoretical photon collection efficiency based on the distance of the excitation from the optical
        axis. Based on the ratio of the fraction of light emitted reaching the objective. Modeled as a fraction of the
        surface area of a sphere.
        :param deviation: Distance from the optical axis in microns.
        :param type: Either `sphere` or `hemisphere`. `hemisphere` doubles the collection efficiency based on the
        assumption that only one hemispehre is viable for collection.
        :return: Collection efficiency.
        """
        # Rename parameters
        h = self.working_distance
        d = deviation
        r = h * np.tan(self.theta)

        # Calculate properties of oblique cone
        n = np.sqrt(h ** 2 + (d + r) ** 2)
        m = np.sqrt(h ** 2 + (d - r) ** 2)
        l = np.sqrt(h ** 2 + d ** 2)

        beta = ((d - r) ** 2 + m ** 2 - h ** 2) / (2 * (d - r) * m)
        gamma = ((d + r) ** 2 + n ** 2 - h ** 2) / (2 * (d + r) * n)

        if d + r is 0:
            beta = np.pi / 2

        # Calculate collection angle
        theta_emission = (beta - gamma) / 2

        # Calculate efficiency
        efficiency = 0.5 * (1 - np.cos(theta_emission))

        if type is 'sphere':
            return efficiency
        elif 'hemisphere':
            return 2 * efficiency
        else:
            return TypeError('Please choose `sphere` or `hemisphere`.')

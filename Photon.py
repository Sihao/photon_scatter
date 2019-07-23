import numpy as np
from numpy.random import random as rand


class Photon:
    """
    Class defining properties and methods for a propagating Photon.
    """

    def __init__(self, start_pos, medium):
        """
        :param start_pos: Photon object start position, given as Numpy array in [x, y, z] format
        :param medium: Medium object through which Photon is propagating
        """

        # Random initial direction
        self.theta_i = rand() * np.pi
        self.phi = rand() * 2 * np.pi

        # Properties for position and path length
        self.current_pos = start_pos
        self.new_pos = []
        self.total_path = 0
        self.path = start_pos

        # Initial photon weight
        self.W = 1

        # Weight threshold before discarding photon
        self.weight_threshold = 0.001

        # Roulette parameter to maintain photon after crossing weight threshold
        self.m = 10

        # Medium photon is propagating through (defines interface)
        self.medium = medium

        # Reflection probability
        self.P_r = []

        # Photon state
        self.is_propagating = True
        self.is_absorbed = False

    def next_pos(self):
        """
        Sets new_pos depending on current position, direction and path length
        """

        new_x = self.current_pos[0] + (self.medium.path_length * np.sin(self.theta_i) * np.cos(self.phi))
        new_y = self.current_pos[1] + (self.medium.path_length * np.sin(self.theta_i) * np.sin(self.phi))
        new_z = self.current_pos[2] + (self.medium.path_length * np.cos(self.theta_i))

        self.new_pos = np.array([new_x, new_y, new_z])

    def check_boundary(self):
        """
        Check whether Photon is in Medium. Currently checks for all three dimensions eventhough propagate() function
        assumes semi-infinite slab.
        :return: False if new Photon position is within shape boundary defined by Medium object, True otherwise
        """

        (new_x, new_y, new_z) = self.new_pos
        (medium_x, medium_y, medium_z) = self.medium.shape

        if (new_x < -float('inf') or new_x > medium_x or new_y < -float(
                'inf') or new_y > medium_y or new_z < 0 or new_z > medium_z):

            return True
        else:
            return False

    def is_reflected(self):
        """
        Determine whether photon is actually reflected based on reflection probability P_r and angle of incidence
        :return: True if photon is reflected, False otherwise.
        """

        theta_i = self.theta_i

        # Calculate critical angle
        theta_crit = np.arcsin(self.medium.N_e / self.medium.N_i)

        # All angles greater than critical angle will be reflected
        if theta_i > theta_crit:
            self.P_r = 1

        else:
            # Calculate external angle based on Snell's Law
            theta_e = np.arcsin((self.medium.N_i / self.medium.N_e) * np.sin(theta_i))

            # Reflection probability is determined by the Fresnel reflection coefficient
            r_par = (self.medium.N_i * np.cos(theta_i) - self.medium.N_e * np.cos(theta_e)) / (
                    self.medium.N_i * np.cos(theta_i) + self.medium.N_e * np.cos(theta_e))

            r_perp = (self.medium.N_i * np.cos(theta_e) - self.medium.N_e * np.cos(theta_i)) / (
                    self.medium.N_i * np.cos(theta_e) + self.medium.N_e * np.cos(theta_i))

            self.P_r = (r_par ** 2 + r_perp ** 2) / 2

        if rand() < self.P_r:
            return True
        else:
            return False

    def scatter(self):
        """
        Calculate new propagation direction of photon to model scattering.
        Sets theta_i and phi.
        """
        g = self.medium.g

        # If scattering is isotropic then polar angle is randomly distributed between 0 and \pi
        if g == 0:
            self.theta_i = np.cos(2 * rand() - 1)

        else:
            # Polar angle (theta) anisotropic scattering determined by Henyey-Greenstein phase function
            self.theta_i = np.arccos((1 + g ** 2 - ((1 - g ** 2) / (1 + g - 2 * g * rand())) ** 2) / 2 * g)

        # Azimuthal angle (phi) randomly distributed between 0 and 2 * \pi
        self.phi = 2 * np.pi * rand()

    def absorb(self):
        """
        Decrease photon weight based on absorption and scattering coefficient
        Sets photon weight W.
        """
        self.W = self.W - (self.medium.mu_a / self.medium.mu_s) * self.W

        if self.W < self.weight_threshold:
            self.roulette()

    def roulette(self):
        if rand() < 1 / self.m:
            self.W = self.m * self.W
        else:
            self.W = 0
            self.is_absorbed = True

    def propagate(self):
        """
        Perform one propagation step of photon through medium
        Sets current_pos to new_pos, sets new direction depending on scattering, reflection or refraction. Sets
        is_propagating to False if photon exits medium. Reduces photon weight based on absorption and scattering
        coefficient and sets is_absorbed to True when photon is absorbed.
        """

        # Calculate next position
        self.next_pos()

        # Check if new position is in medium
        out_of_bounds = self.check_boundary()

        # Increment total path length
        self.total_path = self.total_path + self.medium.path_length

        if out_of_bounds:
            if self.is_reflected():
                # Position after reflection (formula from Prahl '89)
                # Semi-infinite slab with only depth `t`

                t = self.medium.shape[2]

                if self.new_pos[2] < 0:
                    self.new_pos[2] = -self.new_pos[2]
                elif self.new_pos[2] > t:
                    self.new_pos[2] = 2 * t - self.new_pos[2]

                self.path = np.vstack((self.path, self.new_pos))

                self.current_pos = self.new_pos

                # Change propagation direction after reflection (Prahl '89)
                self.theta_i = np.pi - self.theta_i
                self.absorb()
                self.scatter()

            # If photon leaves the self.medium
            else:
                theta_refracted = np.arcsin(self.medium.N_i / self.medium.N_e * np.sin(self.theta_i))

                # Construct rotation matrix
                rotation = self.theta_i - theta_refracted

                # Rotate final position in accordance with refraction
                R_z = np.array(
                    [[np.cos(rotation), -np.sin(rotation), 0], [np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
                self.new_pos = (np.matmul(R_z, np.array([[self.new_pos[0]], [self.new_pos[1]], [self.new_pos[2]]]))).T

                self.theta_i = theta_refracted
                self.path = np.vstack((self.path, self.new_pos))

                self.current_pos = self.new_pos
                self.is_propagating = False

        # Set current position to new calculated position if photon is still in medium
        else:
            self.path = np.vstack((self.path, self.new_pos))
            self.current_pos = self.new_pos

            self.absorb()
            self.scatter()

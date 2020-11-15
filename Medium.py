import numpy as np
from numpy.random import random as rand


class Medium:
    """
    Class defining a Medium for Photon object to propagate through
    """

    def __init__(self, shape, mu_s, mu_a, n_i, n_e, g):
        """
        :param shape: Boundary coordinates for the medium, Photon currently only supports semi-infinite slab
                      with finite Z defined, in meantime use large values for X and Y. Expects Numpy array in the form
                      of [x, y, z].
        :param mu_s: Scattering coefficient of the medium
        :param mu_a: Absorption coefficient of the medium
        :param n_i: Internal refractive index (of this medium)
        :param n_e: External refractive index (of the outside)
        :param g: Tissue anisotropy factor, determines scattering
        """

        self.shape = shape
        self.mu_s = mu_s
        self.mu_a = mu_a
        self.N_i = n_i
        self.N_e = n_e
        self.g = g

        # TODO: Depth/wavelength dependent absorption
        # https://www.osapublishing.org/boe/fulltext.cfm?uri=boe-9-8-3534&id=395109

        # TODO: Temperature plots
        # https://elifesciences.org/articles/53205 (Methods)

    # Path length calculation based on scattering length
    # Path length distribution follows Beer's law
    def get_path_length(self):
        return - np.log(rand()) / (self.mu_s + self.mu_a)
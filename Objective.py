import numpy as np


class Objective:
    def __init__(self, numerical_aperture, working_distance, focal_length, refractive_index=1):
        self.n = refractive_index
        self.NA = numerical_aperture
        self.working_distance = working_distance

        # Computed properties
        self.front_aperture = self.NA * 2 * self.working_distance
        self.theta = np.arcsin(self.NA / self.n)




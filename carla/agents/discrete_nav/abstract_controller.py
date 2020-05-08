from abc import ABCMeta, abstractmethod
import numpy as np


class Controller(metaclass=ABCMeta):
    @abstractmethod
    def control(self, pts_2D, measurements, depth_array):
        pass

    @staticmethod
    def _calc_closest_dists_and_location(transform, pts_2D):
        location = np.array([
            transform.location.x,
            transform.location.y,
        ])
        dists = np.linalg.norm(pts_2D - location, axis=1)
        which_closest = np.argmin(dists)
        return which_closest, dists, location

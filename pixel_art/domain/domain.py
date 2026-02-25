from abc import ABC, abstractmethod

import numpy as np

from pixel_art.utils.render_grid import render_grid


class Domain(ABC):
    @abstractmethod
    def sample(self, rng, stamps):
        """
        Sample a problem instance.

        Parameters
        ----------
        rng : np.random.RandomState
            The random number generator to use.
        stamps : dict
            The stamps to use.

        Returns
        -------
        image : np.ndarray
            The image.
        symbols : list
            The symbols in the image.
        """
        pass

    def demo_grid(self, stamps, grid_size=4, size=10, grid_height=None, **kwargs):
        render_grid(
            grid_size,
            size,
            lambda i: self.sample(np.random.RandomState(i), stamps),
            grid_height=grid_height,
            **kwargs
        )


def domain_types():
    from .stamp_circle import StampCircleDomain, StampCircleDomainSingleDigit

    return dict(
        StampCircleDomain=StampCircleDomain,
        StampCircleDomainSingleDigit=StampCircleDomainSingleDigit,
    )

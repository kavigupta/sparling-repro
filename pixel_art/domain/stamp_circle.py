import attr
import numpy as np

from .domain import Domain
from .noise import noise
from .stamp import compute_background_pct, compute_max_size


@attr.s
class StampCircleDomain(Domain):
    size = attr.ib()
    random_shift = attr.ib()
    min_radius = attr.ib()
    max_syms = attr.ib()
    pre_noise = attr.ib()
    post_noise = attr.ib()

    def sample_with_metadata(self, rng, stamps):
        max_size = compute_max_size(stamps)
        image = np.zeros((self.size, self.size), dtype=bool)
        noise(image, compute_background_pct(stamps) * self.pre_noise, rng)
        center, radius = sample_circle(
            self.size, self.random_shift + max_size, self.min_radius, rng
        )
        # pick a random number of symbols
        num_syms = rng.randint(3, self.max_syms + 1)
        symbols = list(rng.choice(list(stamps), size=num_syms, replace=False))
        angles = sample_angles(num_syms, rng)
        placed_stamps = []
        for i in range(num_syms):
            point = point_around_circle(center, radius, angles[i])
            point += rng.randint(-self.random_shift, self.random_shift + 1, size=2)
            slices = stamp_image(image, stamps[symbols[i]], point)
            placed_stamps.append(dict(point=point, symbol=symbols[i], slices=slices))
        noise(image, self.post_noise, rng)
        return image, canonicalize(symbols), dict(placed_stamps=placed_stamps)

    def sample(self, rng, stamps):
        image, symbols, _ = self.sample_with_metadata(rng, stamps)
        return image, symbols


@attr.s
class StampCircleDomainSingleDigit(Domain):
    size = attr.ib()
    pre_noise = attr.ib()
    post_noise = attr.ib()

    def sample_with_metadata(self, rng, stamps):
        image = np.zeros((self.size, self.size), dtype=bool)
        noise(image, compute_background_pct(stamps) * self.pre_noise, rng)
        # pick a random number of symbols
        symbol = rng.choice(list(stamps))
        placed_stamps = []
        point = np.array([self.size // 2, self.size // 2])
        slices = stamp_image(image, stamps[symbol], point)
        placed_stamps.append(dict(point=point, symbol=symbol, slices=slices))
        noise(image, self.post_noise, rng)
        return image, canonicalize([symbol]), dict(placed_stamps=placed_stamps)

    def sample(self, rng, stamps):
        image, symbols, _ = self.sample_with_metadata(rng, stamps)
        return image, symbols


def point_around_circle(center, radius, angle):
    """
    Compute a point on a circle.

    Parameters
    ----------
    center : np.ndarray
        The center of the circle.
    radius : int
        The radius of the circle.
    angle : float
        The angle of the point.

    Returns
    -------
    point : np.ndarray
        The point on the circle.
    """
    point = center + radius * np.array([np.cos(angle), np.sin(angle)])
    point = np.round(point).astype(np.int64)
    return point


def stamp_image(image, stamp, point):
    """
    Stamp an image onto another image, centered at a given point.

    Parameters
    ----------
    image : np.ndarray
        The image to stamp onto.
    stamp : np.ndarray
        The image to stamp.
    point : np.ndarray
        The center of the stamp.
    """
    top_left = point - [stamp.shape[0] // 2, stamp.shape[1] // 2]
    slice_x = slice(top_left[0], top_left[0] + stamp.shape[0])
    slice_y = slice(top_left[1], top_left[1] + stamp.shape[1])
    image[slice_x, slice_y] = stamp
    return slice_x, slice_y


def sample_circle(size, padding_size, min_radius, rng):
    """
    Sample a circle (center, radius) that is guaranteed to fit within the
    image, with a given padding on the edges.

    Parameters
    ----------
    size : int
        The size of the image.
    padding_size : int
        The padding on the edges of the image.
    min_radius : int
        The minimum radius of the circle.
    rng : np.random.RandomState
        The random number generator.

    Returns
    -------
    center : np.ndarray
        The center of the circle.
    radius : int
        The radius of the circle.
    """
    # need radius + padding_size <= size - radius - padding_size
    # so radius <= size / 2 - padding_size
    radius = rng.randint(min_radius, size // 2 - padding_size)
    bounding_box = radius + padding_size, size - radius - padding_size
    center = np.array([rng.randint(*bounding_box) for _ in range(2)])
    return center, radius


def canonicalize(symbols):
    """
    Canonicalize a list of symbols, by rotating them so that
    the first symbol is the smallest.

    Parameters
    ----------
    symbols : list
        The symbols to canonicalize.

    Returns
    -------
    canonicalized : list
        The canonicalized symbols.
    """
    i = np.argmin(symbols)
    return symbols[i:] + symbols[:i]


def sample_angles(num_angles, rng):
    """
    Sample a set of angles around a circle.

    Uniform, but conditioned to be at least 1 / (2 * num_angles) apart.
    """
    while True:
        angles = rng.rand(num_angles)
        angles = angles / angles.sum()
        if (angles >= 1 / (2 * num_angles)).all():
            angles = np.cumsum(angles)
            angles += rng.rand()
            angles *= 2 * np.pi
            return angles

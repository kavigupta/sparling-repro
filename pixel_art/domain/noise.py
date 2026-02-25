def noise(image, p, rng):
    """
    Add noise to an image.

    Parameters
    ----------
    image : np.ndarray
        The image to add noise to.
    p : float
        The probability of flipping a pixel.
    rng : np.random.RandomState
        The random number generator to use.
    """
    mask = rng.rand(*image.shape) < p
    image[mask] = ~image[mask]
    return image

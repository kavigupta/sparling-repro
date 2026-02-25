import numpy as np


def bootstrap_mean(values, count=1000, confidence=0.05):
    values = np.array(values)
    values = values[~np.isnan(values)]
    samplings = np.random.RandomState(0).choice(values, size=(count, len(values)))
    means = np.mean(samplings, axis=1)
    return np.quantile(means, [confidence / 2, 1 - confidence / 2])

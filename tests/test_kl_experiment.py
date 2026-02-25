import unittest

from permacache import no_cache_global

from pixel_art.analysis.kl_experiment import (
    compute_motif_density,
    get_step_for_model,
    kl_names,
)


class KLExperimentTest(unittest.TestCase):
    def test_compute_motif_density_regression(self):
        name = kl_names[0.01]
        step = get_step_for_model(name)
        with no_cache_global():
            mean_act, density = compute_motif_density(name, step, samples=10)
        self.assertAlmostEqual(mean_act, 0.46897488832473755, places=4)
        self.assertAlmostEqual(density, 0.276921, places=4)

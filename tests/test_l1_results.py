import unittest

from permacache import no_cache_global

from pixel_art.analysis.l1_results import get_step_for_model, l1_names, latest_density


class L1ResultsTest(unittest.TestCase):
    def test_latest_density_regression(self):
        name = l1_names[0.1]
        step = get_step_for_model(name)
        with no_cache_global():
            density = latest_density(name, step, samples=10)
        self.assertAlmostEqual(density, 0.375559, places=4)

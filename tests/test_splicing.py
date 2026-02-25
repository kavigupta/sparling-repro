import os
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from permacache import no_cache_global

from latex_decompiler.splicing.evaluation import (
    all_motif_errors,
    evaluate_e2e_from_checkpoint,
    evaluate_motifs_from_checkpoint,
    evaluate_motifs_randomized_from_checkpoint,
)


class TestSplicingE2E(unittest.TestCase):
    def test_e2e_from_checkpoint(self):
        with no_cache_global():
            e2e = evaluate_e2e_from_checkpoint("spl-1c2_1", "2.114141e-03")
        self.assertAlmostEqual(e2e, 1 - 20.98 / 100, places=4)


class TestSplicingMotifs(unittest.TestCase):
    def test_motifs_from_checkpoint(self):
        with no_cache_global():
            conf = evaluate_motifs_from_checkpoint(
                "spl-1c2_1", "2.114141e-03", amount=10
            )
        errors = all_motif_errors(conf)
        self.assertAlmostEqual(errors["no_3p5p"]["fne"], 0.6410220214568041, places=4)
        self.assertAlmostEqual(errors["no_3p5p"]["fpe"], 0.365043695380774, places=4)
        self.assertAlmostEqual(errors["no_3p5p"]["ce"], 0.9001179709005112, places=4)
        self.assertAlmostEqual(errors["direct"]["fne"], 0.6006608180471688, places=4)
        self.assertAlmostEqual(errors["direct"]["fpe"], 0.12484394506866417, places=4)
        self.assertAlmostEqual(errors["direct"]["ce"], 0.9141226818830243, places=4)

    def test_motifs_randomized_from_checkpoint(self):
        with no_cache_global():
            conf = evaluate_motifs_randomized_from_checkpoint(
                "spl-1c2_1", "2.114141e-03", seed=0, amount=10
            )
        errors = all_motif_errors(conf)
        self.assertAlmostEqual(errors["no_3p5p"]["fne"], 0.6016374929418408, places=4)
        self.assertAlmostEqual(errors["no_3p5p"]["fpe"], 0.41500829187396354, places=4)
        self.assertAlmostEqual(errors["no_3p5p"]["ce"], 0.9287739192062368, places=4)
        self.assertAlmostEqual(errors["direct"]["fne"], 0.5601002620485359, places=4)
        self.assertAlmostEqual(errors["direct"]["fpe"], 0.19962686567164178, places=4)
        self.assertAlmostEqual(errors["direct"]["ce"], 0.9380989380989381, places=4)

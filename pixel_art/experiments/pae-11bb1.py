from sys import argv

from pixel_art.analysis.pixel_art_experiment import (
    single_digit_spec as pixel_art_single_digit_spec,
)
from pixel_art.experiments.experiment import E2EExperiment, SingleDigitMotifsExperiment

exp_orig = E2EExperiment(__file__, argv[1], dict(type="digit_stamps"))

exp = SingleDigitMotifsExperiment(__file__, argv[1])

exp.architecture = dict(
    type="LaTeXPredictorJustMotifsModelStub",
    motifs_spec=exp_orig.architecture["motifs_spec"],
    channels=64,
)

exp.data_spec = pixel_art_single_digit_spec

exp.batch_size = 200

exp.lr = 1e-5

exp.val_every = 50_000
exp.total_steps = 10**6 + 1

exp.run()

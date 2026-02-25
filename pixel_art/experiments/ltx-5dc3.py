from sys import argv

from latex_decompiler.latex_cfg import LATEX_CFG_SPECS
from pixel_art.analysis.latex_digit_dataset import latex_digits_spec
from pixel_art.experiments.experiment import E2EExperiment, SingleDigitMotifsExperiment

exp_orig = E2EExperiment(__file__, argv[1], dict(type="digit_stamps"))
exp = SingleDigitMotifsExperiment(__file__, argv[1])

spec = LATEX_CFG_SPECS["latex_cfg"]

exp.data_spec = latex_digits_spec

exp.architecture = dict(
    type="LaTeXPredictorJustMotifsModelStub",
    motifs_spec=dict(
        type="ConvolutionalMotifModelMultipleSizes",
        in_channels=1,
        out_channels=64,
        sizes_per_character=1,
        # dummy for compatibility
        dataset_configuration=dict(min_dpi=100, max_dpi=200),
        cfg=spec["cfg"],
        cr=32,
    ),
    channels=64,
)

exp.batch_size = 200

exp.lr = 1e-5

exp.val_every = 50_000
exp.total_steps = 3 * 10**6 + 1

exp.run()

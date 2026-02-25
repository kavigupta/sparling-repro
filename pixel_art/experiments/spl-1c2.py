from sys import argv

from latex_decompiler.latex_cfg import LATEX_CFG_SPECS
from pixel_art.experiments.experiment import E2EExperiment

exp = E2EExperiment(__file__, argv[1], dict(type="digit_stamps"))

spec = LATEX_CFG_SPECS["latex_cfg"]

num_motifs = 80

exp.data_spec = dict(type="SplicingDataset", is_training=True)

exp.architecture = dict(
    type="LaTeXPredictor",
    channels=num_motifs,
    motifs_spec=dict(
        type="SplicingMotifModel",
        motif_width=21,
        motif_fc_layers=5,
        motif_feature_extractor_spec=dict(type="ResidualStack", depth=5),
    ),
    sparsity_spec=dict(
        type="SparseLayerWithBatchNorm",
        underlying_sparsity_spec=dict(type="EnforceSparsityPerChannel1D"),
        affine=True,
        starting_sparsity=0.5,
        input_dimensions=1,
    ),
    encoder_spec=dict(type="Identity"),
    decoder_spec=dict(type="SplicingDownstreamCorrected", window=10_000),
    in_channels=4,
    add_axis=False,
)

exp.validation_spec = dict(type="TopKValidation")

exp.batch_size = 32

exp.lr = 1e-5

exp.done_at_density = 0.18e-2

# decrease by 2% per epoch
exp.suo_spec["threshold_decrease_per_iter"] = 0.02 / 162706

exp.run()

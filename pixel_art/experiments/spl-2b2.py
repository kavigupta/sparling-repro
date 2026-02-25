from sys import argv

from latex_decompiler.latex_cfg import LATEX_CFG_SPECS
from pixel_art.experiments.experiment import E2EExperiment

exp = E2EExperiment(__file__, argv[1], dict(type="digit_stamps"))

spec = LATEX_CFG_SPECS["latex_cfg"]

num_motifs_each = [2, 80]

exp.data_spec = dict(type="SplicingDataset", is_training=True)

exp.architecture = dict(
    type="LaTeXPredictor",
    channels=sum(num_motifs_each),
    motifs_spec=dict(
        type="ParallelMotifModels",
        motif_model_specs=[
            dict(
                type="SplicingLSSI",
                acceptor="model/splicepoint-model-acceptor-1",
                donor="model/splicepoint-donor2-2.sh",
            ),
            dict(
                type="SplicingMotifModel",
                motif_width=21,
                motif_fc_layers=5,
                motif_feature_extractor_spec=dict(type="ResidualStack", depth=5),
            ),
        ],
        out_channels_each=num_motifs_each,
    ),
    sparsity_spec=dict(
        type="ParallelSparsityLayers",
        sparse_model_specs=[
            dict(type="NoSparsity"),
            dict(
                type="SparseLayerWithBatchNorm",
                underlying_sparsity_spec=dict(type="EnforceSparsityPerChannel1D"),
                affine=True,
                input_dimensions=1,
            ),
        ],
        channels_each=num_motifs_each,
        starting_sparsity=0.5,
    ),
    encoder_spec=dict(type="Identity"),
    decoder_spec=dict(type="SplicingDownstream", window=10_000),
    in_channels=4,
    add_axis=False,
)

exp.validation_spec = dict(type="TopKValidation")

exp.batch_size = 8

exp.lr = 1e-5

exp.done_at_density = 0.18e-2

# decrease by 1% per epoch
exp.suo_spec["threshold_decrease_per_iter"] = 0.01 / 162706

exp.run()

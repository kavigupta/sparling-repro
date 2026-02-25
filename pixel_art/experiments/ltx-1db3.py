from sys import argv

from latex_decompiler.latex_cfg import LATEX_CFG_SPECS
from pixel_art.experiments.experiment import E2EExperiment

exp = E2EExperiment(__file__, argv[1], dict(type="digit_stamps"))

spec = LATEX_CFG_SPECS["latex_cfg"]

exp.architecture["sparsity_spec"] = dict(
    type="SparseLayerWithBatchNorm",
    underlying_sparsity_spec=dict(
        type="EnforceSparsityPerChannel2D",
        enforce_sparsity_per_channel_spec=dict(
            type="EnforceSparsityPerChannelAccumulated",
            accumulation_stop_strategy=dict(
                type="StopAtFixedNumberMotifs", num_motifs=10
            ),
        ),
    ),
    affine=False,
    starting_sparsity=0.5,
    channels=10,
)

exp.data_spec = dict(
    type="LaTeXDataset",
    latex_cfg=spec["cfg"],
    font="computer_modern",
    data_config=dict(
        minimal_length=1, maximal_length=spec["maximal_length"], dpi=200, w=360, h=120
    ),
)

exp.architecture["motifs_spec"] = dict(
    type="ConvolutionalMotifModelMultipleSizes",
    in_channels=1,
    out_channels=64,
    sizes_per_character=1,
    # dummy for compatibility
    dataset_configuration=exp.architecture["motifs_spec"]["dataset_configuration"],
    cfg=exp.architecture["motifs_spec"]["cfg"],
    cr=32,
)
exp.architecture["decoder_spec"]["cfg"] = spec["cfg"]

exp.batch_size = 8

exp.lr = 1e-5

exp.done_at_density = 9.64 / (120 * 360 * 10)

exp.set_num_motifs(100, 10)

exp.run()

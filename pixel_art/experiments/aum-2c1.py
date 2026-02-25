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
    affine=True,
    starting_sparsity=0.5,
    channels=10,
)

N_MELS = 64

exp.data_spec = dict(
    type="AudioMNISTDomainDataset",
    domain_spec=dict(
        type="AudioClipDomain",
        digits_per_speaker_limit=1,
        clip_length_seconds=15,
        length_range=(5, 10),
        speaker_set=[1],
        noise_amplitude=-100,
        operation_spec=dict(type="ListOffDigitsOperation"),
    ),
    n_mels=N_MELS,
)

exp.architecture["motifs_spec"] = dict(
    type="AudioConvolutionalMotifModel",
    in_channels=N_MELS,
    out_channels=64,
    cr=16,
)

exp.architecture["post_sparse_spec"]["type"] = "CollapseMotifsAudio"
exp.architecture["encoder_spec"] = dict(
    type="TransformerAudio",
    transformer_encoder_spec=dict(type="TransformerEncoder", nhead=8, layers=6),
)


exp.architecture["decoder_spec"]["cfg"] = spec["cfg"]

exp.batch_size = 8

exp.lr = 1e-5

exp.done_at_density = 7.5 / (642 * 10)

exp.set_num_motifs(100, 10)

exp.run()

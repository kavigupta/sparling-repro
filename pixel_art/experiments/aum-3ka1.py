from sys import argv

from latex_decompiler.latex_cfg import LATEX_CFG_SPECS
from pixel_art.experiments.experiment import E2EExperiment

exp = E2EExperiment(__file__, argv[1], dict(type="digit_stamps"))

spec = LATEX_CFG_SPECS["latex_cfg"]

exp.architecture = dict(
    type="LoadWithFrozenMotifsAndNoBottleneck",
    model_path=f"model/aum-2ka1_{exp.train_seed}",
    at_density=0.0011892044771002475,
)

N_MELS = 64

exp.data_spec = dict(
    type="AudioMNISTDomainDataset",
    domain_spec=dict(
        type="AudioClipDomain",
        digits_per_speaker_limit=None,
        clip_length_seconds=15,
        length_range=(5, 10),
        speaker_set=list(range(1, 1 + 51)),
        noise_amplitude=-10,
        operation_spec=dict(type="ListOffDigitsOperation"),
    ),
    n_mels=N_MELS,
)

exp.batch_size = 8

exp.lr = 1e-5

exp.total_steps = 10**6

exp.run()

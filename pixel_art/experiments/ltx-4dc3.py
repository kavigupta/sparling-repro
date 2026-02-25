from sys import argv

from latex_decompiler.latex_cfg import LATEX_CFG_SPECS
from pixel_art.experiments.experiment import E2EExperiment

exp = E2EExperiment(__file__, argv[1], dict(type="digit_stamps"))

spec = LATEX_CFG_SPECS["latex_cfg"]

exp.architecture = dict(
    type="LoadWithFrozenMotifsAndNoBottleneck",
    model_path=f"model/ltx-2dc3_{exp.train_seed}",
    at_density=2.8252244733928113e-05,
)

exp.data_spec = dict(
    type="NoisyBinaryLaTeXDataset",
    latex_cfg=spec["cfg"],
    font="computer_modern",
    data_config=dict(
        minimal_length=1, maximal_length=spec["maximal_length"], dpi=200, w=360, h=120
    ),
    noise_amount=0.25,
)

exp.batch_size = 8

exp.lr = 1e-5

exp.total_steps = 70_000

exp.run()

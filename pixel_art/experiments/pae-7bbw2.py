from sys import argv

from pixel_art.experiments.experiment import PIXEL_ART_RIGHT_ABOVE_LINE, E2EExperiment

exp = E2EExperiment(__file__, argv[1], dict(type="digit_stamps"))

exp.data_spec["domain_spec"]["pre_noise"] = 0.5
exp.architecture["sparsity_spec"] = dict(
    type="SparseLayerWithBatchNorm",
    underlying_sparsity_spec=dict(type="EnforceSparsityPerChannel2D"),
    affine=True,
    starting_sparsity=1 - PIXEL_ART_RIGHT_ABOVE_LINE / 0.75,
    channels=10,
)

exp.total_steps = 20 * 10**6

exp.run()

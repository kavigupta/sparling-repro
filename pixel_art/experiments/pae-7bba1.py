from sys import argv

from pixel_art.experiments.experiment import E2EExperiment

exp = E2EExperiment(__file__, argv[1], dict(type="digit_stamps"))

exp.data_spec["domain_spec"]["pre_noise"] = 0.5
exp.architecture["sparsity_spec"] = dict(
    type="SparseLayerWithBatchNorm",
    underlying_sparsity_spec=dict(type="EnforceSparsityPerChannel2D"),
    affine=True,
    starting_sparsity=0.5,
    channels=10,
)

exp.suo_spec["threshold_decrease_per_iter"] = 0

exp.total_steps = 600_001

exp.run()

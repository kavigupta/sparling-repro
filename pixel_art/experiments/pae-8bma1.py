from sys import argv

from pixel_art.experiments.experiment import E2EExperiment

exp = E2EExperiment(__file__, argv[1], dict(type="digit_stamps"))

exp.data_spec["domain_spec"]["pre_noise"] = 0.5
exp.architecture["sparsity_spec"] = dict(
    type="SparseLayerWithBatchNorm",
    underlying_sparsity_spec=dict(type="SparsityForL1"),
    affine=True,
    starting_sparsity=0.5,
    channels=10,
)

exp.motif_loss_specs = [dict(type="L1Loss", weight=2)]

exp.done_at_density = -float("inf")

exp.run()

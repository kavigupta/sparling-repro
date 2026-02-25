from sys import argv

from pixel_art.experiments.experiment import PIXEL_ART_RIGHT_ABOVE_LINE, E2EExperiment

exp = E2EExperiment(__file__, argv[1], dict(type="digit_stamps"))

exp.data_spec["domain_spec"]["pre_noise"] = 0.5
exp.architecture["sparsity_spec"] = dict(
    type="SparseLayerWithBatchNorm",
    underlying_sparsity_spec=dict(type="SparsityForKL"),
    affine=True,
    starting_sparsity=0.5,
    channels=10,
)

exp.architecture["post_sparse_spec"] = dict(
    type="ThresholdForKL",
    and_then_spec=exp.architecture["post_sparse_spec"],
)

exp.motif_loss_specs = [
    dict(type="KLLoss", weight=10, target=PIXEL_ART_RIGHT_ABOVE_LINE)
]

exp.done_at_density = 0.4

exp.run()

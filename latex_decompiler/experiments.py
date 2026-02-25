from .latex_cfg import latex_cfg


def e2e_dataset_1():
    return dict(
        type="LaTeXDataset",
        latex_cfg=latex_cfg,
        font="computer_modern",
        data_config=dict(minimal_length=1, maximal_length=30, dpi=200, w=360, h=120),
    )

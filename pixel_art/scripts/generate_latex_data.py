import tqdm.auto as tqdm

from latex_decompiler.dataset import DATA_TYPE_MAP
from latex_decompiler.latex_cfg import LATEX_CFG_SPECS
from latex_decompiler.utils import construct


def generate_data(seed, spec, amount=10**7):
    data_spec = dict(
        type="LaTeXDataset",
        latex_cfg=spec["cfg"],
        font="computer_modern",
        data_config=dict(
            minimal_length=1,
            maximal_length=spec["maximal_length"],
            dpi=200,
            w=360,
            h=120,
        ),
    )

    print("Generating data for seed", seed)
    dset = construct(DATA_TYPE_MAP, data_spec, seed=seed)
    for i in tqdm.trange(0, amount + 1000, 1000):
        dset[i]


def main():
    for seed in [-1, -2, *range(1, 1 + 9)]:
        for spec in LATEX_CFG_SPECS:
            if spec != "latex_cfg":
                continue
            print(seed, spec)
            generate_data(
                seed, LATEX_CFG_SPECS[spec], amount=10**7 if seed > 0 else 10**5
            )


if __name__ == "__main__":
    main()

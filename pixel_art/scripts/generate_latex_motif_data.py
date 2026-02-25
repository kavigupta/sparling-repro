import tqdm.auto as tqdm

from latex_decompiler.dataset import DATA_TYPE_MAP
from latex_decompiler.latex_difficult import difficult_latex_spec
from latex_decompiler.localize_latex_characters import character_bounding_boxes
from latex_decompiler.utils import construct


def generate_data(seed, group, *, amount):
    print("Generating data for seed", seed)
    dset = construct(DATA_TYPE_MAP, difficult_latex_spec(group), seed=seed)
    for i in tqdm.trange(amount):
        _, sample = dset[i]
        character_bounding_boxes(
            sample, font=dset.font_for(i), data_config=dset.data_config
        )


def main():
    for seed in [-1, -2, *range(1, 1 + 9)]:
        for group in ["A", "B"]:
            print(seed, group)
            generate_data(seed, group, amount=5 * 10**5 if seed > 0 else 10**4)


if __name__ == "__main__":
    main()

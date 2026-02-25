import datetime
import time

import fire

from pixel_art.analysis.latex_experiment import (
    all_results,
    models,
    noise_latex_dset_spec,
)


def run(max_above_line=6):
    while True:
        print(datetime.datetime.now())
        # _ = all_results(max_above_line=max_above_line, num_samples=10_000)
        _ = all_results(
            max_above_line=max_above_line,
            num_samples=10_000,
            models=models,
            latex_dset_spec=noise_latex_dset_spec(0.125),
        )
        _ = all_results(
            max_above_line=max_above_line,
            num_samples=10_000,
            models=models,
            latex_dset_spec=noise_latex_dset_spec(0.25),
        )
        print(datetime.datetime.now())
        time.sleep(60 * 60)


if __name__ == "__main__":
    fire.Fire(run)

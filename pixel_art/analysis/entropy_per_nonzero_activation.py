import os

import numpy as np
import torch
import tqdm.auto as tqdm
from permacache import permacache, stable_hash

from latex_decompiler.dataset import DATA_TYPE_MAP
from latex_decompiler.evaluate import compute_accuracy_from_sequences
from latex_decompiler.remapping_pickle import load_with_remapping_pickle
from latex_decompiler.utils import construct, run_batched_fn, strip_start_and_end_tokens
from pixel_art.analysis.evaluate_motifs import evaluate_motifs
from pixel_art.domain.domain import domain_types
from pixel_art.domain.stamp import digit_stamps

SELECTED_CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "selected_checkpoints",
)


def compute_bin(values, bin_maxima):
    assert len(values.shape) == 1
    return (values[:, None] < bin_maxima).argmax(-1)


def compute_bin_maxima_and_means_percentiles(values, num_bins):
    values = values[values != 0]
    if values.size == 0:
        return np.zeros(num_bins), np.zeros(num_bins)
    bin_maxima = np.quantile(values, np.linspace(0, 1, num_bins + 1))[1:]
    bin_idxs = compute_bin(values, bin_maxima)
    counts, sums = np.zeros((2, num_bins))
    np.add.at(counts, bin_idxs, 1)
    np.add.at(sums, bin_idxs, values)
    means_by_bin = sums / counts
    return bin_maxima, means_by_bin


class ThresholdBinner:
    @classmethod
    def calibrate_to_percentiles(cls, mots, num_bins):
        return cls(
            [
                compute_bin_maxima_and_means_percentiles(mots[:, i], num_bins)
                for i in range(mots.shape[1])
            ]
        )

    def __init__(self, by_channel):
        self.by_channel = by_channel

    def manuiplate_motifs(self, mot):
        mot = torch.clone(mot)
        for c in range(mot.shape[1]):
            nonzero_mot = mot[:, c][mot[:, c] != 0].cpu().numpy()
            maxs, means = self.by_channel[c]
            bin_idxs = compute_bin(nonzero_mot, maxs)
            binned = means[bin_idxs]
            mot[:, c][mot[:, c] != 0] = torch.tensor(
                binned, device=mot.device, dtype=mot.dtype
            )
        return mot


class IdentityBinner:
    def __init__(self, mots):
        # ignore mots
        pass

    def manuiplate_motifs(self, mot):
        return mot


@permacache(
    "pixel_art/analysis/entropy_per_nonzero_activation/compute_binned_accuracy_2",
    key_function=dict(model=stable_hash),
)
def compute_binned_accuracy(
    model, binner_spec, calibration_samples, evaluation_samples
):
    from pixel_art.analysis.main_experiment import data_spec

    stamps = digit_stamps()
    actual = construct(domain_types(), data_spec["domain_spec"])
    dset = construct(DATA_TYPE_MAP, data_spec, seed=-3)

    _, _, mots, _ = evaluate_motifs(model, actual, stamps, samples=calibration_samples)
    binner = construct(
        dict(
            to_percentiles=ThresholdBinner.calibrate_to_percentiles,
            identity=IdentityBinner,
        ),
        binner_spec,
        mots=mots,
    )
    ys, yps = run_batched_fn(
        lambda xs: model.forward_test(
            torch.tensor(np.array(xs)).float().cuda(),
            dset.data_config["maximal_length"],
            manipulate_motifs=binner.manuiplate_motifs,
        ),
        dset,
        evaluation_samples,
        128,
        pbar=tqdm.tqdm,
    )
    yps = [strip_start_and_end_tokens(yp) for yp in yps]
    acc = compute_accuracy_from_sequences(ys, yps, accuracy_metric="edit-dist")
    return acc


@permacache(
    "pixel_art/analysis/entropy_per_nonzero_activation/compute_binned_accuracy_from_checkpoint_3",
    multiprocess_safe=True,
)
def compute_binned_accuracy_from_checkpoint(
    model_name,
    checkpoint_key,
    binner_spec,
    calibration_samples=10_000,
    evaluation_samples=10_000,
):
    print(model_name, checkpoint_key, binner_spec)
    m = load_with_remapping_pickle(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key),
        weights_only=False,
    ).eval()
    return compute_binned_accuracy(
        m, binner_spec, calibration_samples, evaluation_samples
    )


def compute_for_multiple_rounding_modes(
    models, num_bins_values, target_density, **kwargs
):
    checkpoint_key = f"{target_density:.6e}"
    binned_results = {}
    for name, (path, num_seeds) in tqdm.tqdm(list(models.items())):
        binned_results[name] = {}
        for seed in range(1, 1 + num_seeds):
            model_name = f"{path}_{seed}"
            if not os.path.exists(
                os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key)
            ):
                continue
            result = {
                None: compute_binned_accuracy_from_checkpoint(
                    model_name, checkpoint_key, dict(type="identity"), **kwargs
                )
            }
            result.update(
                {
                    num_bins: compute_binned_accuracy_from_checkpoint(
                        model_name,
                        checkpoint_key,
                        dict(type="to_percentiles", num_bins=num_bins),
                        **kwargs,
                    )
                    for num_bins in num_bins_values
                }
            )
            binned_results[name][seed] = result
    return binned_results

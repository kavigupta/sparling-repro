import copy
import os

import numpy as np
import torch
import tqdm.auto as tqdm
from permacache import permacache, stable_hash

from latex_decompiler.remapping_pickle import load_with_remapping_pickle
from pixel_art.analysis.evaluate_motifs import (
    categorize_relationships,
    confusion_from_results,
)
from pixel_art.analysis.gather_evaluation import compute_all_errors
from pixel_art.analysis.latex_experiment import all_results_generic
from pixel_art.analysis.main_experiment import SparsityBar
from pixel_art.audo_mnist.data.dataset import (
    AudioClipDomainSpectrogram,
    AudioMNISTDomainDataset,
)

SELECTED_CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "selected_checkpoints",
)

models = {
    "Audio-10c": ("aum-2g1", 1, 10),
    "Audio-10c [ms]": ("aum-2j1", 1, 10),
    "Audio-10c [ms]/noise=-10": ("aum-2ka1", 9, 10),
}

model_to_evaluate = "Audio-10c [ms]/noise=-10"


basic_ac_domain = dict(
    digits_per_speaker_limit=1,
    clip_length_seconds=15,
    length_range=(5, 10),
    speaker_set=[1],
    noise_amplitude=-100,
    operation_spec=dict(type="ListOffDigitsOperation"),
)

ac_multi_digit_domain = dict(
    digits_per_speaker_limit=None,
    clip_length_seconds=15,
    length_range=(5, 10),
    speaker_set=[1],
    noise_amplitude=-100,
    operation_spec=dict(type="ListOffDigitsOperation"),
)

ac_multi_speaker_domain_train = dict(
    digits_per_speaker_limit=None,
    clip_length_seconds=15,
    length_range=(5, 10),
    speaker_set=list(range(1, 1 + 51)),
    noise_amplitude=-100,
    operation_spec=dict(type="ListOffDigitsOperation"),
)

ac_multi_speaker_noisy_domain_train = dict(
    digits_per_speaker_limit=None,
    clip_length_seconds=15,
    length_range=(5, 10),
    speaker_set=list(range(1, 1 + 51)),
    noise_amplitude=-10,
    operation_spec=dict(type="ListOffDigitsOperation"),
)

ac_multi_speaker_domain_test = dict(
    digits_per_speaker_limit=None,
    clip_length_seconds=15,
    length_range=(5, 10),
    speaker_set=list(range(52, 1 + 60)),
    noise_amplitude=-100,
    operation_spec=dict(type="ListOffDigitsOperation"),
)

ac_multi_speaker_noisy_domain_test = dict(
    digits_per_speaker_limit=None,
    clip_length_seconds=15,
    length_range=(5, 10),
    speaker_set=list(range(52, 1 + 60)),
    noise_amplitude=-10,
    operation_spec=dict(type="ListOffDigitsOperation"),
)


def spectrogram_kwargs(**domain_kwargs):
    return dict(
        domain_spec=dict(type="AudioClipDomain", **domain_kwargs),
        n_mels=64,
    )


def get_data(**domain_kwargs):
    return AudioMNISTDomainDataset(**spectrogram_kwargs(**domain_kwargs), seed=1)


def get_domain(**domain_kwargs):
    return AudioClipDomainSpectrogram(
        **spectrogram_kwargs(**domain_kwargs), time_chunk=1.5
    )


def sparsity_bar(num_motifs):
    return SparsityBar.right_above_line(7.5 / (642 * num_motifs), num_motifs)


@permacache(
    "pixel_art/analysis/audio_mnist_experiment/data",
    key_function=dict(mod=stable_hash, batch_size=None),
)
def data(domain_kwargs, amount, seed):
    domain = get_domain(**domain_kwargs)
    rng = np.random.RandomState(seed % 2**32)
    xs, _, metas = [], [], []
    for _ in tqdm.trange(amount):
        x, _, meta = domain.sample_with_metadata(rng, stamps=None)
        xs.append(x)
        metas.append(meta["placed_stamps"])

    xs = np.array(xs)
    return xs, metas


def compute_motifs(mod, xs, batch_size=16):
    mots = []
    with torch.no_grad():
        for i in range(0, len(xs), batch_size):
            mots.append(
                mod.run_motifs_without_post_sparse(
                    torch.tensor(xs[i : i + batch_size]).cuda()
                )
                .cpu()
                .numpy()
            )
    mots = np.concatenate(mots)
    return mots


def categorize_all(mots, metas):
    motif_names = [f"#{i}" for i in range(mots[0].shape[0])]
    results = []
    for i in range(len(mots)):
        result = categorize_relationships(
            mots[i].copy(),
            placed_stamps=metas[i],
            handle_multi=False,
            motif_names=motif_names,
            include_stamp=True,
        )
        results += [((i, stamp), m, r) for (stamp, m, r) in result]
    return results


@permacache(
    "pixel_art/analysis/audio_mnist_experiment/precisely_evaluate_audio_mnist_motifs_tagged",
    key_function=dict(mod=stable_hash, batch_size=None),
)
def precisely_evaluate_audio_mnist_motifs_tagged(
    mod, domain_kwargs, *, amount, seed=-2, batch_size=16
):
    xs, metas = data(domain_kwargs, amount, seed)
    mots = compute_motifs(mod, xs, batch_size=batch_size)

    results = categorize_all(mots, metas)
    return results


@permacache(
    "pixel_art/analysis/audio_mnist_experiment/precisely_evaluate_audio_mnist_motifs_untagged",
    key_function=dict(mod=stable_hash, batch_size=None),
)
def precisely_evaluate_audio_mnist_motifs_untagged(
    mod, domain_kwargs, *, amount, seed=-2, batch_size=16
):
    results = precisely_evaluate_audio_mnist_motifs_tagged(
        mod, domain_kwargs, amount=amount, seed=seed, batch_size=batch_size
    )
    return [(m, r) for (_, m, r) in results]


@permacache(
    "pixel_art/analysis/audio_mnist_experiment/precisely_evaluate_audio_mnist_motifs_untagged_from_checkpoint_2",
)
def precisely_evaluate_audio_mnist_motifs_untagged_from_checkpoint(
    model_name, checkpoint_key, domain_kwargs, *, amount, seed=-2, batch_size=16
):
    print("Evaluating", model_name, checkpoint_key)
    mod = load_with_remapping_pickle(
        os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, checkpoint_key),
        weights_only=False,
    ).eval()
    return precisely_evaluate_audio_mnist_motifs_untagged(
        mod, domain_kwargs, amount=amount, seed=seed, batch_size=batch_size
    )


def all_results_audio_mnist(
    *,
    max_above_line,
    num_samples,
    models,
    monitoring_mode=False,
    domain_kwargs,
):
    fn = precisely_evaluate_audio_mnist_motifs_untagged_from_checkpoint
    kwargs = dict(amount=num_samples, domain_kwargs=domain_kwargs)

    return all_results_generic(
        fn,
        kwargs,
        models=models,
        max_above_line=max_above_line,
        monitoring_mode=monitoring_mode,
        compute_sparsity_bar=sparsity_bar,
    )


audio_mnist_data_spec_train = dict(
    type="AudioMNISTDomainDataset",
    **spectrogram_kwargs(**ac_multi_speaker_noisy_domain_train),
)

audio_mnist_data_spec_test = dict(
    type="AudioMNISTDomainDataset",
    **spectrogram_kwargs(**ac_multi_speaker_noisy_domain_test),
)


def all_audio_mnist_errors(max_num_above_line):
    return compute_all_errors(
        model_to_evaluate,
        models[model_to_evaluate],
        motif_results_fn=lambda **kwargs: all_results_audio_mnist(
            **kwargs,
            domain_kwargs=ac_multi_speaker_noisy_domain_test,
        ),
        dset_spec=audio_mnist_data_spec_test,
        sparsity_bar_fn=sparsity_bar,
        max_num_above_line=max_num_above_line,
        retrain_path="aum-3ka1",
        retrain_seeds=9,
        retrain_step=480_000,
    )


def digits_dataset_spec(dset_spec):
    dset_spec = copy.deepcopy(dset_spec)
    dset_spec["domain_spec"]["length_range"] = (1, 1)
    dset_spec["domain_spec"]["clip_length_seconds"] = 3
    dset_spec["domain_spec"]["type"] = "AudioMNISTSingleDigitDomain"
    dset_spec["type"] = "AudioMNISTSingleDigitDomainDataset"
    return dset_spec


@permacache("pixel_art/analysis/audio_mnist_experiment/digits_data_2")
def digits_data(dset_spec, seed, amount=10**4):
    assert dset_spec["type"] == "AudioMNISTDomainDataset"
    dset_spec = digits_dataset_spec(dset_spec)
    from .single_digit_motif_results import compute_digits_data

    return compute_digits_data(dset_spec, seed, amount)


def audio_mnist_confusion_matrix(num_samples=10_000):
    results = all_results_audio_mnist(
        max_above_line=0,
        num_samples=num_samples,
        models={
            model_to_evaluate: models[model_to_evaluate],
        },
        monitoring_mode=False,
        domain_kwargs=ac_multi_speaker_noisy_domain_test,
    )
    res = results[model_to_evaluate, 0][0]
    conf = confusion_from_results(res)
    conf.columns = [
        chr(ord("A") + int(x[1:])) if x != "none" else "none" for x in conf.columns
    ]
    return conf

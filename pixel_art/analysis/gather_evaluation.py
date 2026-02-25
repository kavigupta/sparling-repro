import pandas as pd
import tqdm.auto as tqdm

from latex_decompiler.evaluate import accuracies_at_density
from pixel_art.analysis.evaluate_latex_motifs import (
    confusion_error,
    false_negative_error,
    false_positive_error,
)
from pixel_art.analysis.evaluate_retrain import collect_retrain_results_for_step


def compute_statistic(statistic, models, results):
    max_above_line = max(i for _, i in results)
    return {
        model: [
            [statistic(v) for v in results[model, i]] for i in range(1 + max_above_line)
        ]
        for model in models
    }


def compute_motif_statistics(
    model_to_evaluate, model_spec, steps_above_line, results_fn
):
    models_to_eval = {k: model_spec for k in [model_to_evaluate]}
    results_test_noise = results_fn(
        max_above_line=steps_above_line,
        num_samples=10000,
        models=models_to_eval,
    )
    stats = {
        "ce": compute_statistic(confusion_error, models_to_eval, results_test_noise),
        "fpe": compute_statistic(
            false_positive_error, models_to_eval, results_test_noise
        ),
        "fne": compute_statistic(
            false_negative_error, models_to_eval, results_test_noise
        ),
    }
    stats = {k: stats[k][model_to_evaluate][steps_above_line] for k in stats}
    stats = {k: [x * 100 for x in xs] for k, xs in stats.items()}
    return stats


def compute_e2e_error(
    model_to_evaluate,
    model_spec,
    steps_above_line,
    *,
    dset_spec,
    sparsity_bar_fn,
    accuracy_metric,
):
    res = accuracies_at_density(
        {model_to_evaluate: model_spec[:2]},
        dset_spec,
        sparsity_bar_fn(model_spec[-1]).sparsity_bar / 0.75**steps_above_line,
        accuracy_metric=accuracy_metric,
    )
    return 100 - res.loc[model_to_evaluate]


def compute_all_errors_above_line(
    model_to_evaluate,
    model_spec,
    *,
    motif_results_fn,
    dset_spec,
    sparsity_bar_fn,
    steps_above_line,
):
    stats = compute_motif_statistics(
        model_to_evaluate,
        model_spec,
        steps_above_line,
        results_fn=motif_results_fn,
    )
    # e2e_exact = compute_e2e_error(
    #     model_to_evaluate,
    #     model_spec,
    #     steps_above_line,
    #     dset_spec=dset_spec,
    #     sparsity_bar_fn=sparsity_bar_fn,
    #     accuracy_metric="exact",
    # )
    e2e_edit = compute_e2e_error(
        model_to_evaluate,
        model_spec,
        steps_above_line,
        dset_spec=dset_spec,
        sparsity_bar_fn=sparsity_bar_fn,
        accuracy_metric="edit-dist",
    )
    # stats["e2e_exact"] = e2e_exact
    stats["e2e_edit"] = e2e_edit
    return {"MT": pd.DataFrame(stats)}


def compute_all_errors(
    model_to_evaluate,
    model_spec,
    *,
    motif_results_fn,
    dset_spec,
    sparsity_bar_fn,
    max_num_above_line,
    retrain_path,
    retrain_seeds,
    retrain_step,
):
    result = {
        steps_above_line: compute_all_errors_above_line(
            model_to_evaluate,
            model_spec,
            motif_results_fn=motif_results_fn,
            dset_spec=dset_spec,
            sparsity_bar_fn=sparsity_bar_fn,
            steps_above_line=steps_above_line,
        )
        for steps_above_line in range(1 + max_num_above_line)
    }
    result["not-sparse"] = {
        "MT": pd.DataFrame(
            dict(
                e2e_edit=100
                - accuracies_at_density(
                    {model_to_evaluate: model_spec[:2]},
                    dset_spec,
                    0.5,
                    accuracy_metric="edit-dist",
                ).loc[model_to_evaluate]
            )
        )
    }
    result["sparse-retrained"] = {
        "MT": pd.DataFrame(
            dict(
                e2e_edit=100
                - collect_retrain_results_for_step(
                    retrain_path,
                    retrain_seeds,
                    data_spec=dset_spec,
                    quiet=True,
                    step=retrain_step,
                    batch_size=16,
                )
            )
        )
    }
    return result


def get_domains(only_mt=False):
    from pixel_art.analysis.audio_mnist_experiment import all_audio_mnist_errors
    from pixel_art.analysis.latex_experiment import all_latex_errors
    from pixel_art.analysis.pixel_art_experiment import all_pixel_art_errors

    return [
        ("DigitCircle", all_pixel_art_errors, 1 if only_mt else 2),
        ("LaTeX-OCR", all_latex_errors, 1),
        ("AudioMNISTSequence", all_audio_mnist_errors, 1),
    ]


def get_minimum_sparsity_by_domain():
    from pixel_art.analysis.audio_mnist_experiment import (
        sparsity_bar as audio_load_sparsity_bars,
    )
    from pixel_art.analysis.latex_experiment import (
        sparsity_bar as latex_load_sparsity_bars,
    )
    from pixel_art.analysis.pixel_art_experiment import load_sparsity_bars

    return {
        "DigitCircle": load_sparsity_bars().right_above_line.sparsity_bar,
        "LaTeX-OCR": latex_load_sparsity_bars(32).sparsity_bar,
        "AudioMNISTSequence": audio_load_sparsity_bars(10).sparsity_bar,
    }


def compute_all_results(max_num_above_line):
    return {
        domain_name: fn(max_num_above_line)
        for domain_name, fn, _ in tqdm.tqdm(get_domains(), desc="domains")
    }


def main():
    for max_num_above_line in range(1, 1 + 10):
        print("max_num_above_line", max_num_above_line)
        print("*" * 80)
        compute_all_results(max_num_above_line)


if __name__ == "__main__":
    main()

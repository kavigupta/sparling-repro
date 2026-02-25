import os

import pandas as pd
import torch
from matplotlib import pyplot as plt
from permacache import permacache

from latex_decompiler.evaluate import accuracy_at_checkpoint
from latex_decompiler.remapping_pickle import load_with_remapping_pickle
from latex_decompiler.utils import construct, load_model, load_steps
from pixel_art.analysis.evaluate_motifs import errors_from_checkpoint, evaluate_motifs
from pixel_art.domain.domain import domain_types
from pixel_art.domain.stamp import digit_stamps
from pixel_art.utils.bootstrap import bootstrap_mean

SELECTED_CHECKPOINTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "selected_checkpoints",
)

l1_names = {
    0.1: "pae-8bia1_1",
    1: "pae-8bla1_1",
    2: "pae-8bma1_1",
    5: "pae-8bna1_1",
    10: "pae-8boa1_1",
}

metric_names = {
    "fpe": "FPE",
    "fne": "FNE",
    "ce": "CE",
    "e2e_edit": "E2EE",
}

l1_paths = {k: f"model/{path}" for k, path in l1_names.items()}

data_spec = dict(
    type="PixelArtDomainDataset",
    domain_spec=dict(
        type="StampCircleDomain",
        size=100,
        min_radius=20,
        random_shift=3,
        max_syms=6,
        pre_noise=0.5,
        post_noise=0.05,
    ),
    stamps_spec=dict(type="digit_stamps"),
)


def domain():
    actual = construct(domain_types(), data_spec["domain_spec"])
    stamps = digit_stamps()
    return actual, stamps


def get_step_for_model(model_name):
    """Read the step from the selected_checkpoints directory."""
    ckpt_dir = os.path.join(SELECTED_CHECKPOINTS_DIR, model_name)
    step_files = [f for f in os.listdir(ckpt_dir) if f.startswith("step_")]
    assert len(step_files) == 1, f"Expected 1 step file in {ckpt_dir}, got {step_files}"
    return int(step_files[0].split("_")[1])


@permacache("notebooks/latex-domain/pixel-art-l1/latest_density_v2")
def latest_density(model_name, step, samples=1000):
    actual, stamps = domain()
    path = os.path.join(SELECTED_CHECKPOINTS_DIR, model_name, f"step_{step}")
    updated = load_with_remapping_pickle(path, weights_only=False).eval()
    _, _, mots, _ = evaluate_motifs(updated, actual, stamps, samples=samples)
    return float((mots != 0).mean())


def compute_l1_results():
    actual, stamps = domain()
    steps = {}
    errs = {"fpe": {}, "fne": {}, "ce": {}, "e2e_edit": {}}
    for k, name in l1_names.items():
        steps[k] = get_step_for_model(name)
        short_path = name
        errs["e2e_edit"][k] = 100 - 100 * accuracy_at_checkpoint(
            short_path,
            f"step_{steps[k]}",
            data_spec,
            "edit-dist",
            batch_size=100,
        )
        motif_errs = errors_from_checkpoint(
            short_path, f"step_{steps[k]}", actual, stamps, samples=100
        )
        for i, metric in enumerate(("fne", "fpe", "ce")):
            errs[metric][k] = 100 * motif_errs[i]
    densities = {k: 100 * latest_density(l1_names[k], steps[k]) for k in l1_names}
    return errs, densities


def scatterplot(l1_errs, l1_densities, sparling_errs, sparling_density):
    plt.figure(dpi=200, facecolor="white", figsize=(6, 3), tight_layout=True)
    l1_values = sorted(l1_names)
    for k in l1_values:
        plt.scatter(
            [l1_densities[k]], [l1_errs["e2e_edit"][k]], label=f"$L_1$: $\\lambda$={k}"
        )
    lo, hi = bootstrap_mean(sparling_errs["e2e_edit"])
    plt.errorbar(
        sparling_density,
        (lo + hi) / 2,
        (hi - lo) / 2,
        capsize=2,
        color="black",
        label=f"Ours: {k}",
        linestyle=" ",
    )
    plt.xlabel("Density [%]")
    plt.ylabel(f"End-to-End error [%]")
    plt.ylim(0, plt.ylim()[1])
    plt.xscale("log")
    plt.axvline(100 * 4.492e-05, label="theoretical min density", color="blue")
    plt.legend()
    # plt.savefig("output/pixel_art_l1.png", bbox_inches="tight")
    plt.show()


def table(l1_errs, l1_densities, sparling_errs, sparling_density):
    table_names = []
    table_errors = {"fpe": [], "fne": [], "ce": [], "e2e_edit": []}
    table_densities = []
    for lam in sorted(l1_errs["e2e_edit"]):
        table_names.append(("$L_1$", rf"$\lambda = {lam}$"))

        for metric in table_errors:
            table_errors[metric].append(f"{l1_errs[metric][lam]:.2f}")
        table_densities.append(l1_densities[lam])
    # mu, (lo, hi) = elements["MT"]
    table_names.append((r"\textsc{Sparling}", "$\mathrm{MT}$"))
    for metric in table_errors:
        mu = sparling_errs[metric].mean()
        (lo, hi) = bootstrap_mean(sparling_errs[metric])
        table_errors[metric].append(f"{mu:.2f} [{lo:.2f}-{hi:.2f}]")
    table_densities.append(sparling_density)

    table_densities = [f"{x:.2g}" for x in table_densities]

    rows = {}
    for metric in table_errors:
        rows[metric_names[metric] + r" [\%]"] = table_errors[metric]
    rows[r"Density [\%]"] = table_densities
    return pd.DataFrame(
        rows, index=pd.MultiIndex.from_arrays(list(zip(*table_names)))
    ).T

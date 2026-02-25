"""
Select relevant checkpoints from model/ and save them to selected_checkpoints/.

For each model/seed/density combination used in the results notebooks, this script
finds the correct checkpoint using the same logic as evaluate.py, then saves just
the model state_dict to selected_checkpoints/<model_name>/<density_string>.

Usage:
    python select_checkpoints.py
"""

import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from latex_decompiler.evaluate import load_sparse_model_for_sparsity
from latex_decompiler.utils import load_model
from pixel_art.analysis.main_experiment import load_sparsity_bars
from pixel_art.analysis.latex_experiment import sparsity_bar as latex_sparsity_bar
from pixel_art.analysis.audio_mnist_experiment import sparsity_bar as audio_sparsity_bar

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "selected_checkpoints")


def save_checkpoint(model, model_name, key_string):
    out_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, key_string)
    if os.path.exists(out_path):
        print(f"  already exists: {out_path}")
        return
    torch.save(model, out_path)
    print(f"  saved: {out_path}")


def density_string(density):
    return f"{density:.6e}"


def step_string(step):
    return f"step_{step}"


def select_density_based(prefix, max_seed, bar, n_steps=11):
    """Select checkpoints for density-based models at bar / 0.75^n for n=0..n_steps-1, plus density=0.5."""
    densities = [bar / 0.75**n for n in range(n_steps)]
    densities.append(0.5)

    for seed in range(1, max_seed + 1):
        model_name = f"{prefix}_{seed}"
        print(f"Processing {model_name}...")
        for density in densities:
            key = density_string(density)
            out_path = os.path.join(OUTPUT_DIR, model_name, key)
            if os.path.exists(out_path):
                print(f"  already exists: {key}")
                continue
            try:
                model = load_sparse_model_for_sparsity(
                    model_name, target_density=density
                )
            except Exception as e:
                print(f"  SKIP {key}: {e}")
                continue
            if model is None:
                print(f"  SKIP {key}: no matching checkpoint")
                continue
            save_checkpoint(model, model_name, key)


def select_step_based(prefix, max_seed, step):
    """Select checkpoints for step-based models at a fixed training step."""
    for seed in range(1, max_seed + 1):
        model_name = f"{prefix}_{seed}"
        print(f"Processing {model_name}...")
        key = step_string(step)
        out_path = os.path.join(OUTPUT_DIR, model_name, key)
        if os.path.exists(out_path):
            print(f"  already exists: {key}")
            continue
        try:
            _, loaded = load_model(f"model/{model_name}", step=step)
            if loaded is None:
                print(f"  SKIP {key}: checkpoint not found")
                continue
            model = loaded["model"]
        except Exception as e:
            print(f"  SKIP {key}: {e}")
            continue
        save_checkpoint(model, model_name, key)


def select_step_based_with_filter(prefix, seed, max_step):
    """Select checkpoint for a model using step_filter <= max_step (latest step up to max_step)."""
    model_name = f"{prefix}_{seed}"
    print(f"Processing {model_name}...")
    try:
        step, loaded = load_model(
            f"model/{model_name}",
            key="model",
            step_filter=lambda x: x <= max_step,
        )
        if loaded is None:
            print(f"  SKIP: checkpoint not found")
            return
        model = loaded["model"]
    except Exception as e:
        print(f"  SKIP: {e}")
        return
    key = step_string(step)
    save_checkpoint(model, model_name, key)


def select_splicing(prefix, max_seed, target_density):
    """Select checkpoints for splicing models at a single density."""
    for seed in range(1, max_seed + 1):
        model_name = f"{prefix}_{seed}"
        print(f"Processing {model_name}...")
        key = density_string(target_density)
        out_path = os.path.join(OUTPUT_DIR, model_name, key)
        if os.path.exists(out_path):
            print(f"  already exists: {key}")
            continue
        try:
            model = load_sparse_model_for_sparsity(
                model_name, target_density=target_density
            )
        except Exception as e:
            print(f"  SKIP {key}: {e}")
            continue
        if model is None:
            print(f"  SKIP {key}: no matching checkpoint")
            continue
        save_checkpoint(model, model_name, key)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Density-based models ---

    pixel_art_bar = load_sparsity_bars().right_above_line.sparsity_bar
    latex_bar = latex_sparsity_bar(32).sparsity_bar
    audio_bar = audio_sparsity_bar(10).sparsity_bar

    print("=" * 60)
    print("DigitCircle ST no-batchnorm (pae-2b2), seeds 1-9")
    print("=" * 60)
    select_density_based("pae-2b2", 9, pixel_art_bar)

    print("=" * 60)
    print("DigitCircle MT no-batchnorm (pae-2ba2), seeds 1-9")
    print("=" * 60)
    select_density_based("pae-2ba2", 9, pixel_art_bar)

    print("=" * 60)
    print("DigitCircle ST (pae-6bb1), seeds 1-9")
    print("=" * 60)
    select_density_based("pae-6bb1", 9, pixel_art_bar)

    print("=" * 60)
    print("DigitCircle MT (pae-7bb1), seeds 1-9")
    print("=" * 60)
    select_density_based("pae-7bb1", 9, pixel_art_bar)

    print("=" * 60)
    print("LaTeX-OCR (ltx-2dc3), seeds 1-9")
    print("=" * 60)
    select_density_based("ltx-2dc3", 9, latex_bar)

    print("=" * 60)
    print("AudioMNIST (aum-2ka1), seeds 1-9")
    print("=" * 60)
    select_density_based("aum-2ka1", 9, audio_bar)

    # --- Splicing at single density ---

    splicing_density = 0.5 * 0.75**19
    print("=" * 60)
    print("Splicing (spl-1c2), seeds 1-4")
    print("=" * 60)
    select_splicing("spl-1c2", 4, splicing_density)

    # --- Step-based models ---

    print("=" * 60)
    print("Non-sparse baseline (pae-7bba1), seeds 1-9, step 600k")
    print("=" * 60)
    select_step_based("pae-7bba1", 9, 600_000)

    print("=" * 60)
    print("Retrain ST (pae-6bbb1), seeds 1-9, step 600k")
    print("=" * 60)
    select_step_based("pae-6bbb1", 9, 600_000)

    print("=" * 60)
    print("Retrain MT (pae-7bbb1), seeds 1-9, step 600k")
    print("=" * 60)
    select_step_based("pae-7bbb1", 9, 600_000)

    print("=" * 60)
    print("Retrain LaTeX (ltx-4dc3), seeds 1-9, step 60k")
    print("=" * 60)
    select_step_based("ltx-4dc3", 9, 60_000)

    print("=" * 60)
    print("Retrain AudioMNIST (aum-3ka1), seeds 1-9, step 480k")
    print("=" * 60)
    select_step_based("aum-3ka1", 9, 480_000)

    # --- KL study models (pae-9ba*1_1), seed 1, latest step <= 3M ---

    kl_names = [
        "pae-9baf1",
        "pae-9bai1",
        "pae-9bal1",
        "pae-9bao1",
        "pae-9bar1",
        "pae-9bau1",
        "pae-9bax1",
        "pae-9baza1",
    ]
    print("=" * 60)
    print("KL study models, seed 1, latest step <= 3M")
    print("=" * 60)
    for prefix in kl_names:
        select_step_based_with_filter(prefix, seed=1, max_step=3_000_000)

    # --- L1 study models (pae-8b*a1_1), seed 1, latest step ---

    l1_names = [
        "pae-8bia1",
        "pae-8bla1",
        "pae-8bma1",
        "pae-8bna1",
        "pae-8boa1",
    ]
    print("=" * 60)
    print("L1 study models, seed 1, latest step")
    print("=" * 60)
    for prefix in l1_names:
        select_step_based_with_filter(prefix, seed=1, max_step=float("inf"))

    # --- Non-adaptive sparsity models ---

    print("=" * 60)
    print("Non-adaptive sparsity (pae-7bbw2, pae-7bbx2), seed 1, step 9.02M")
    print("=" * 60)
    for prefix in ["pae-7bbw2", "pae-7bbx2"]:
        model_name = f"{prefix}_1"
        print(f"Processing {model_name}...")
        key = step_string(9_020_000)
        out_path = os.path.join(OUTPUT_DIR, model_name, key)
        if os.path.exists(out_path):
            print(f"  already exists: {key}")
            continue
        try:
            _, loaded = load_model(f"model/{model_name}", step=9_020_000)
            if loaded is None:
                print(f"  SKIP {key}: checkpoint not found")
                continue
            model = loaded["model"]
        except Exception as e:
            print(f"  SKIP {key}: {e}")
            continue
        save_checkpoint(model, model_name, key)

    # --- Direct evaluation models ---

    print("=" * 60)
    print("Direct DigitCircle (pae-11bb1), seeds 1-9, steps 100k-1M")
    print("=" * 60)
    for step in range(100_000, 1_000_001, 100_000):
        select_step_based("pae-11bb1", 9, step)

    print("=" * 60)
    print("Direct LaTeX (ltx-5dc3), seeds 1-9, steps 500k-3M")
    print("=" * 60)
    for step in range(500_000, 3_000_001, 500_000):
        select_step_based("ltx-5dc3", 9, step)

    print("=" * 60)
    print("Direct AudioMNIST (aum-4ka1), seeds 1-9, steps 500k-3M")
    print("=" * 60)
    for step in range(500_000, 3_000_001, 500_000):
        select_step_based("aum-4ka1", 9, step)

    print()
    print("Done!")


if __name__ == "__main__":
    main()

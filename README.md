
# Sparling Reproducibility

## Setup

### 1. Install dependencies

```sh
sudo apt install texlive-full texlive-latex-extra
conda install -c conda-forge poppler
pip install -r requirements.txt
```

### 2. Download data

Download spliceai data and AudioMNIST:

```sh
./setup_data.sh
```

### 3. Generate latex data

```sh
PYTHONPATH=. python -u pixel_art/scripts/generate_latex_data.py
PYTHONPATH=. python -u pixel_art/scripts/generate_latex_motif_data.py
```

## Training

Train models using:

```sh
PYTHONPATH=. python -u pixel_art/experiments/%name.py %seed
```

where `%name` is the name of the experiment and `%seed` is the seed (1-9).

Main experiments:

- `pae-7bb1`, `ltx-2dc3`, `aum-2ka1`: main experiments for each domain
- `ltx-4dc3`: retrained experiments
- `pae-11bb1`, `ltx-5dc3`, `aum-4ka1`: experiments training motifs directly

After training, select checkpoints for evaluation:

```sh
python select_checkpoints.py
```

Then run `all-results.ipynb` to generate the tables and figures.

Other experiments:

- `pae-6bb1`: ST experiment. See `pixel-art-results.ipynb`
- `pae-7bbw2`, `pae-7bbx2`: ablations without adaptive sparsity (1.5x and 1.1x starting sparsity). See `pixel-art-no-adaptive-sparsity.ipynb`
- `pae-2ba2`: ablation without batch normalization. See `pixel-art-results.ipynb`
- `pae-9bai1`, `pae-9bal1`, `pae-9bao1`, `pae-9bar1`, `pae-9bau1`, `pae-9bax1`, `pae-9baza1`: KL baselines with lambda values 0.1, 1, 10, 100, 1000, 10000, 100000. See `pixel-art-kl.ipynb`
- `pae-8bia1`, `pae-8bil1`, `pae-8bim1`, `pae-8bin1`, `pae-8bio1`: L1 baselines with lambda values 0.1, 1, 2, 5, 10. See `pixel-art-l1.ipynb`

## Using pre-trained checkpoints

As an alternative to training, you can download pre-trained checkpoints (~126 GB):

```sh
./download_checkpoints.sh
```

## Running tests

Tests require the pre-trained checkpoints (either trained locally or downloaded).

```sh
python -m pytest tests/ -v
```

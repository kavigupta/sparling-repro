
Install dependencies

```sh
sudo apt install texlive-full texlive-latex-extra
conda install -c conda-forge poppler
pip install -r requirements.txt
```

The latex data needs to be generated with

```sh
PYTHONPATH=. python -u pixel_art/scripts/generate_latex_data.py
```

And the latex motif data needs to be generated with

```sh
PYTHONPATH=. python -u pixel_art/scripts/generate_latex_motif_data.py
```

Train the models using the command

```sh
PYTHONPATH=. python -u pixel_art/experiments/%name.py %seed
```

where `%name` is the name of the experiment and `%seed` is the seed for the random number generator.
To run the experiments in the paper, use seeds=1 to 9 and

- `pae-7bb1`, `ltx-2dc3`, `aum-2ka1`: for the main experiments for each domain,
- `ltx-4dc3`: for the the Retrained experiments
- `pae-11bb1`, `ltx-5dc3`, `aum-4ka1`: for the experiments where we train motifs directly

then run `all-results.ipynb` to generate the tables and figures.

The models for the other experiments can be trained using the following names:

- `pae-6bb1` for the ST experiment. Use `pixel-art-results.ipynb` to see the results
- `pae-7bbw2` and `pae-7bbx2` for the ablations without adaptive sparsity, starting at 1.5x the sparsity target and 1.1x the sparsity target, respectively. Use `pixel-art-no-adaptive-sparsity.ipynb` to see the results
- `pae-2ba2` for the ablation without the batch normalization. Use `pixel-art-results.ipynb` to see the results
- `pae-9bai1`, `pae-9bal1`, `pae-9bao1`, `pae-9bar1`, `pae-9bau1`, `pae-9bax1`, `pae-9baza1` for the KL baselines with lambda
    values of 0.1, 1, 10, 100, 1000, 10_000, and 100_000 respectively. Use `pixel-art-kl.ipynb` to see the results
- `pae-8bia1`, `pae-8bil1`, `pae-8bim1`, `pae-8bin1`, `pae-8bio1` for the L1 baselines with lambda values of 0.1, 1, 2, 5, and 10 respectively. Use `pixel-art-l1.ipynb` to see the results
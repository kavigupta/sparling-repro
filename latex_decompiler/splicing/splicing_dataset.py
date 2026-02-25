import os

import h5py
import numpy as np

from .utils import SPLICEAI_DATA_DIR


class SplicingDataset:
    def __init__(self, is_training, seed):
        self.is_training = is_training
        self.seed = seed
        self.dataset = h5py.File(
            os.path.join(
                SPLICEAI_DATA_DIR,
                "dataset_train_all.h5" if is_training else "dataset_test_0.h5",
            ),
            "r",
        )
        num_xs = len([x for x in self.dataset if x.startswith("X")])
        self.x_keys = [f"X{i}" for i in range(num_xs)]
        rng = np.random.RandomState(seed % 2**32)
        rng.shuffle(self.x_keys)
        self.batch_order_each = {
            key: rng.permutation(len(self.dataset[key])) for key in self.x_keys
        }
        self._loaded = {None: None}
        self.index_to_chunk = np.concatenate(
            [np.full(len(self.dataset[key]), i) for i, key in enumerate(self.x_keys)]
        )
        self.index_to_within_chunk = np.concatenate(
            [np.arange(len(self.dataset[key])) for key in self.x_keys]
        )

    def load_chunk(self, chunk):
        if chunk not in self._loaded:
            x_key = self.x_keys[chunk]
            order = self.batch_order_each[x_key]
            y_key = x_key.replace("X", "Y")
            self._loaded = {
                chunk: {
                    "x": self.dataset[x_key][:][order],
                    "y": self.dataset[y_key][0][order],
                }
            }
        return self._loaded[chunk]

    def __getitem__(self, idx):
        idx = idx % len(self.index_to_chunk)
        chunk = self.index_to_chunk[idx]
        loaded = self.load_chunk(chunk)
        loaded = {k: v[self.index_to_within_chunk[idx]] for k, v in loaded.items()}
        return loaded["x"], loaded["y"]

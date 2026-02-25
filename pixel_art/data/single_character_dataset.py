import attr
import numpy as np
from permacache import permacache

from latex_decompiler.utils import compute_seed, construct
from pixel_art.domain.noise import noise
from pixel_art.domain.stamp import compute_background_pct, stamps_types
from pixel_art.domain.stamp_circle import stamp_image


@attr.s
class PixelArtSingleCharacterDataset:
    stamps_spec = attr.ib()
    config = attr.ib()
    seed = attr.ib()
    looping = attr.ib(default=None)

    def __getitem__(self, idx):
        result = generate_single_character_datapoint(
            stamps_spec=self.stamps_spec,
            **self.config,
            seed=compute_seed(self.seed, idx, self.looping),
        )
        return result["image"], result["stamps"][0]


@permacache(
    "pixel_art/data/single_character_dataset/generate_single_character_datapoint"
)
def generate_single_character_datapoint(*, stamps_spec, size, seed, post_noise):
    stamps = load_stamps(stamps_spec)
    rng = np.random.RandomState(seed)
    stamp = rng.choice(list(stamps))
    image = np.zeros((size, size), dtype=np.bool)
    x, y = size // 2, size // 2
    noise(image, compute_background_pct(stamps), rng)
    stamp_image(image, stamps[stamp], np.array([x, y]))
    noise(image, post_noise, rng)
    return dict(
        image=image, stamps=[dict(symbol=stamp, x=x, y=y, relative_size=1, target=True)]
    )


@permacache("pixel_art/data/single_character_dataset/load_stamps")
def load_stamps(stamp_spec):
    return construct(stamps_types(), stamp_spec)

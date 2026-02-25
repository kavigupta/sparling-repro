import numpy as np
from permacache import permacache, stable_hash

from latex_decompiler.utils import construct
from pixel_art.domain.stamp import stamps_types


def flippable(img, i, j):
    """
    Whether a pixel is flippable.

    We define a pixel to be flippable if it has at least 3 neighbors that are
    different from it.
    """
    return (
        sum(
            img[i + offi, j + offj] != img[i, j]
            for offi in [-1, 0, 1]
            for offj in [-1, 0, 1]
        )
        >= 3
    )


@permacache(
    "pixel_art/domain/stamp_flip/perturb_stamp",
    key_function=dict(img=stable_hash),
)
def perturb_stamp(img, iou_target):
    img = img.copy()
    img_original = img.copy()
    target = True
    rng = np.random.RandomState(int(stable_hash(img), 16) % 2**32)
    while True:
        iou = (img & img_original).sum() / (img | img_original).sum()
        if iou < iou_target:
            break
        i, j = rng.choice(img.shape[0] - 2) + 1, rng.choice(img.shape[1] - 2) + 1
        if img[i, j] != target:
            continue
        if not flippable(img, i, j):
            continue
        img[i, j] = ~img[i, j]
        target = not target
    return img


def perturbed_stamps(stamps_spec, iou_target):
    return {
        k: perturb_stamp(v, iou_target)
        for k, v in construct(stamps_types(), stamps_spec).items()
    }

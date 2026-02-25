import numpy as np

"""
# data/digit_circle_stamps.pkl Generated on a fresh install of Ubuntu 18.04 with Pillow==8.4.0

from PIL import Image, ImageFont, ImageDraw
import numpy as np
# import matplotlib.pyplot as plt

def character_stamp(char, w):
    h = int(w * 1.5)
    txt = Image.new("RGB", (w, h))
    fnt = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", h)
    d = ImageDraw.Draw(txt)
    d.text((0, 0), char, font=fnt, fill=(255, 255, 255))
    txt = np.array(txt)
    txt = 255 * (txt > 64)
    txt = txt.any(-1)
    xs, ys = np.where(txt)
    return txt[xs.min() - 1 : xs.max() + 2, ys.min() - 1 : ys.max() + 2]

def digit_stamps(w=10):
    return {str(x): character_stamp(str(x), w=w) for x in range(10)}

import pickle
with open("digit_circle_stamps.pkl", "wb") as f:
    pickle.dump(digit_stamps(), f)
"""


def digit_stamps(w=10):
    assert w == 10, "Only w=10 is supported"
    import pickle

    with open("data/digit_circle_stamps.pkl", "rb") as f:
        return pickle.load(f)


def compute_max_size(stamps):
    return np.max([v.shape for v in stamps.values()])


def compute_background_pct(stamps):
    return np.mean([np.mean(v) for v in stamps.values()])


def stamps_types():
    from .perturb_stamp import perturbed_stamps

    return dict(digit_stamps=digit_stamps, perturbed_stamps=perturbed_stamps)

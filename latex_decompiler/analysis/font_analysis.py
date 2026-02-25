import matplotlib
import numpy as np
import tqdm.auto as tqdm
from matplotlib import pyplot as plt
from permacache import permacache
from scipy.special import logsumexp
from sklearn.decomposition import PCA

from latex_decompiler.render import FONT_TO_HEADER, render


@permacache("latex_decompiler/analysis/font_analysis/font_list_6")
def font_list(letters, fonts=sorted(FONT_TO_HEADER), *, threshold=0.07):
    to_remove = redundant_fonts(fonts, letters, threshold=threshold)
    return [f for f in fonts if f not in to_remove]


@permacache("latex_decompiler/analysis/font_analysis/render_char")
def render_char(char, font):
    print(char, font)
    return render(char, font)


def remove_padding(img, pad_value):
    ys, xs = np.where(img != pad_value)
    return img[ys.min() : 1 + ys.max(), xs.min() : 1 + xs.max()]


def render_char_in_all_fonts(fonts, c):
    images = []
    for f in fonts:
        img = np.array(render_char(c, f)) < 128
        assert (img[:, :, [0]] == img).all()
        img = img[:, :, 0]
        img = remove_padding(img, 0)
        images.append(img)
    h, w = np.max([x.shape for x in images], 0)
    images_stack = np.zeros((len(images), h, w))
    for i, img in enumerate(images):
        pad_h = (h - img.shape[0]) // 2
        pad_w = (w - img.shape[1]) // 2
        images_stack[
            i, pad_h : pad_h + img.shape[0], pad_w : pad_w + img.shape[1]
        ] = img
    return images_stack


def make_into_grid(images_stack, zero=255):
    n, h, w = images_stack.shape
    hh = int(n**0.5)
    ww = int(np.ceil(n / hh))
    images_grid = np.zeros((hh * h, ww * w)) + zero
    for i in range(n):
        ii = i // ww
        jj = i % ww
        images_grid[ii * h : (ii + 1) * h, jj * w : (jj + 1) * w] = images_stack[i]
    return images_grid


def flat_vectors(fonts, letters):
    all_vectors = []
    for c in tqdm.tqdm(letters):
        for_c = render_char_in_all_fonts(fonts, c)
        for_c = for_c.reshape(for_c.shape[0], -1)
        all_vectors.append(for_c)
    return np.concatenate(all_vectors, axis=1)


def redundant_fonts(fonts, letters, *, threshold):
    vec = flat_vectors(fonts, letters)
    diff_matrix = np.abs(vec[:, None] - vec[None]).mean(-1)
    plt.hist(diff_matrix.flatten(), bins=100)
    plt.axvline(threshold, color="red")
    eq_a, eq_b = np.where(diff_matrix <= threshold)
    eq_a, eq_b = [x[eq_a < eq_b] for x in (eq_a, eq_b)]
    duplicates = []
    for a, b in zip(eq_a, eq_b):
        if b in duplicates:
            continue
        print(fonts[b], "is the same as", fonts[a])
        duplicates.append(b)
    return [fonts[b] for b in duplicates]


@permacache("latex_decompiler/analysis/font_analysis/pcaify_3")
def pcaify(fonts, letters):
    vec = flat_vectors(fonts, letters)
    pca = PCA(n_components=2, whiten=True)
    pca.fit(vec)
    return pca.transform(vec)


@permacache("latex_decompiler/analysis/font_analysis/font_groups_4")
def font_groups(fonts, letters):
    cs = pcaify(fonts, letters)
    mask = cs[:, 1] < 0
    return [x for x, m in zip(fonts, mask) if m], [x for x, m in zip(fonts, ~mask) if m]


def visualize_pca(ax, components, images, mask, h=0.2):
    # colormap from alpha=1 to black, linear like gray, but reversed
    cmap = matplotlib.cm.get_cmap("gray")
    cmap = cmap(np.linspace(0, 1, 256))
    cmap[:, 3] = np.linspace(1, 0, 256)
    cmap = cmap[::-1]
    cmap = matplotlib.colors.ListedColormap(cmap)

    for i, img in enumerate(images):
        # show the image at the given location, with height h
        w = h * img.shape[1] / img.shape[0]
        ax.imshow(
            img,
            extent=[
                components[i, 0],
                components[i, 0] + h,
                components[i, 1],
                components[i, 1] + w,
            ],
            cmap=cmap,
        )
    ax.scatter(components[:, 0], components[:, 1], c=mask, marker=".")


def psams_for_letter(fonts, group, c):
    stack = render_char_in_all_fonts(fonts, c)
    psams = stack[[x in group for x in fonts]]
    # nf = (psams**2).sum((1, 2)) ** 0.5
    # psams = psams / nf[:, None, None]
    return psams


def psams_for(fonts, letters):
    groups = font_groups(fonts, letters)
    return {c: [psams_for_letter(fonts, group, c) for group in groups] for c in letters}


def randomized_data(rng, stack, *, noise_level, n, is_control):
    if is_control:
        data = np.zeros((n, *stack.shape[1:])) + stack.mean()
    else:
        index = rng.choice(len(stack), n)
        data = stack[index]
    data = data.copy()
    assert 0 <= noise_level <= 1
    flip_mask = rng.uniform(size=data.shape) < noise_level * 0.5
    data[flip_mask] = 1 - data[flip_mask]
    return data


def apply_psams(psams, data):
    """
    psams: (K, H, W)
    data: (N, H, W)

    Both are bitvectors, so all values are 0 or 1.

    apply_psams(P, D): (N,)
        apply_psams(P, D)[n] = max_k -sum_{h,w} P[k,h,w] != D[n,h,w]
    """
    psams = psams.reshape(psams.shape[0], -1)
    data = data.reshape(data.shape[0], -1)
    data = data.T
    counts = (psams @ data + (1 - psams) @ (1 - data)) - psams.shape[-1]
    return np.max(counts, axis=0)


def line_histogram(ax, data, bin_quantum, **kwargs):
    mi, ma = data.min(), data.max()
    mi, ma = mi - mi % bin_quantum, ma + bin_quantum - ma % bin_quantum
    bins = np.arange(mi, ma + bin_quantum, bin_quantum)
    h = np.histogram(data, bins=bins)[0]
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.plot(bin_centers, h, **kwargs)


def compute_distribution(noise_level, psams, stack, is_control):
    data = randomized_data(
        np.random.RandomState(0),
        stack,
        noise_level=noise_level,
        n=10000,
        is_control=is_control,
    )
    return apply_psams(psams, data)


def fnr_for_fpr(noise_level, psam, stack, delta):
    ab = compute_distribution(noise_level, psam, stack, False)
    ar = compute_distribution(noise_level, psam, stack, True)
    thresh = np.quantile(ar, 1 - delta)
    return (ab < thresh).mean()

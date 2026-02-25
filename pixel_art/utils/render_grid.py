from matplotlib import pyplot as plt


def render_grid(grid_size, img_size, datapoint_fn, grid_height=None, axs=None):
    grid_height = grid_size if grid_height is None else grid_height
    if axs is None:
        _, axs = plt.subplots(
            grid_size if grid_height is None else grid_height,
            grid_size,
            figsize=(img_size, img_size * grid_height / grid_size),
            facecolor="white",
            tight_layout=True,
            dpi=10 / img_size * 200,
        )
    for i, ax in enumerate(axs.flatten()):
        img, sym = datapoint_fn(i)
        ax.imshow(img, interpolation="none", cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("".join(sym))

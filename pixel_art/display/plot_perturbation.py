import tempfile
from typing import List

import numpy as np
import torch
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image

from latex_decompiler.latex_cfg import Token


def plot_perturbation(
    x: np.ndarray,
    y_before: List[Token],
    y_after: List[Token],
    mots_before: np.ndarray,
    mots_after: np.ndarray,
    idx_locations: np.ndarray,
    symbol_before: str,  # single character
    symbols_after: List[str],  # list of characters
    color_before: str,
    color_after: List[str],
):
    # Enable LaTeX rendering
    plt.rcParams["text.usetex"] = True

    # Create a figure with 2 columns: Before and After
    fig = plt.figure(figsize=(12, 8), dpi=200)

    # Use gridspec to ensure equal-sized subplots
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], hspace=0, wspace=0.1)

    # Create 2 subplots for the 2 columns with equal sizes
    ax_before = fig.add_subplot(gs[0, 0])
    ax_after = fig.add_subplot(gs[0, 1])

    # Set up coordinate system for each column: x and y from 0 to 1
    # Add left margin for labels, extend bottom for output text
    for ax in [ax_before, ax_after]:
        ax.set_xlim(-0.08, 1)
        ax.set_ylim(
            0.01, 1
        )  # Extended lower to accommodate output text moved down further
        ax.axis("off")

    # Convert mots_before and mots_after to numpy if needed
    if isinstance(mots_before, torch.Tensor):
        mots_before_np = mots_before.cpu().numpy()
    else:
        mots_before_np = mots_before

    if isinstance(mots_after, torch.Tensor):
        mots_after_np = mots_after.cpu().numpy()
    else:
        mots_after_np = mots_after

    # Calculate image dimensions and position in top third
    img_height, img_width = x.shape[:2]
    img_aspect = img_width / img_height

    # Position image in top third (y from ~0.67 to 1.0)
    top_y = 1.0
    bottom_y = 0.67
    image_height = top_y - bottom_y
    image_width = image_height * img_aspect

    # Center horizontally
    left_x = 0.5 - image_width / 2
    right_x = 0.5 + image_width / 2

    # Ensure image fits within bounds
    if image_width > 1.0:
        image_width = 1.0
        image_height = image_width / img_aspect
        left_x = 0.0
        right_x = 1.0
        bottom_y = top_y - image_height
    else:
        actual_image_height = image_width / img_aspect
        if actual_image_height < (top_y - bottom_y):
            bottom_y = top_y - actual_image_height

    # Helper function to convert image coordinates to plot coordinates
    def img_to_plot_coords(
        x_img, y_img, left_x, right_x, bottom_y, top_y, img_width, img_height
    ):
        x_plot = left_x + (x_img / (img_width - 1)) * (right_x - left_x)
        y_plot = bottom_y + (1 - y_img / (img_height - 1)) * (top_y - bottom_y)
        return x_plot, y_plot

    # Helper function to plot a column's content
    # Returns: (vector_positions, output_token_positions) for drawing arrows
    def plot_column(
        ax,
        mots_data,
        show_motifs,
        show_vectors,
        show_output,
        column_type,  # 'before', 'after', or 'comparison'
        max_components=None,  # Maximum number of components across all vectors
        fixed_vector_height=None,  # Fixed height for each vector (ensures consistency)
        fixed_cell_height=None,  # Fixed height for each cell (ensures consistency)
    ):
        vector_positions = []  # List of (center_x, center_y) for each vector
        output_token_positions = (
            {}
        )  # Maps token index to (center_x, y, color) for changed tokens
        # Plot the image for 'before' column only (top left), black and white (not greyed out)
        if column_type == "before":
            ax.imshow(
                x,
                cmap="gray",
                aspect="equal",
                alpha=1.0,  # Not greyed out
                extent=[left_x, right_x, bottom_y, top_y],
                origin="upper",
            )

        # Calculate bottom image coordinates for both columns (for motif scatterplot)
        image_height_bottom = top_y - bottom_y  # Same height as top image
        bottom_image_top = 0.50  # Position higher up, below vectors
        bottom_image_bottom = bottom_image_top - image_height_bottom

        # Find all motif locations
        batch_idx = 0
        motif_mask = mots_data[batch_idx].any(axis=0)  # (height, width)
        y_all, x_all = np.where(motif_mask)

        # Convert all motif locations to plot coordinates
        # Use bottom image coordinates for both columns
        x_all_plot = []
        y_all_plot = []
        for x_img, y_img in zip(x_all, y_all):
            if bottom_image_bottom is not None:
                # Use bottom image coordinates
                x_plot, y_plot = img_to_plot_coords(
                    x_img,
                    y_img,
                    left_x,
                    right_x,
                    bottom_image_bottom,
                    bottom_image_top,
                    img_width,
                    img_height,
                )
            else:
                # Fallback to top image coordinates
                x_plot, y_plot = img_to_plot_coords(
                    x_img,
                    y_img,
                    left_x,
                    right_x,
                    bottom_y,
                    top_y,
                    img_width,
                    img_height,
                )
            x_all_plot.append(x_plot)
            y_all_plot.append(y_plot)

        # Plot all motif locations for both columns
        if show_motifs:
            ax.scatter(
                x_all_plot,
                y_all_plot,
                c="gray",
                s=25,
                marker="o",
                alpha=0.7,
                edgecolors="darkgray",
                linewidths=1,
                zorder=5,
                label="All motifs",
            )

        # Highlight relevant locations from idx_locations for both columns
        x_coords_img = idx_locations[3, :]
        y_coords_img = idx_locations[2, :]

        # Convert image coordinates to plot coordinates
        # Use bottom image coordinates for both columns
        x_coords_plot = []
        y_coords_plot = []
        for x_img, y_img in zip(x_coords_img, y_coords_img):
            if bottom_image_bottom is not None:
                # Use bottom image coordinates
                x_plot, y_plot = img_to_plot_coords(
                    x_img,
                    y_img,
                    left_x,
                    right_x,
                    bottom_image_bottom,
                    bottom_image_top,
                    img_width,
                    img_height,
                )
            else:
                # Fallback to top image coordinates
                x_plot, y_plot = img_to_plot_coords(
                    x_img,
                    y_img,
                    left_x,
                    right_x,
                    bottom_y,
                    top_y,
                    img_width,
                    img_height,
                )
            x_coords_plot.append(x_plot)
            y_coords_plot.append(y_plot)

        # Draw circles and dots around relevant locations for both columns
        # Use color_before for 'before' column, color_after[i] for 'after' column
        for i, (x_plot, y_plot) in enumerate(zip(x_coords_plot, y_coords_plot)):
            if column_type == "before":
                highlight_color = color_before
            else:
                # Use output color for this location
                highlight_color = color_after[i] if i < len(color_after) else "black"

            ax.scatter(
                x_plot,
                y_plot,
                color=highlight_color,
                s=25,
                marker="o",
                alpha=0.7,
                edgecolors=highlight_color,
                linewidths=1,
                zorder=9,
            )

            circle = Circle(
                (x_plot, y_plot),
                0.02,
                fill=False,
                color=highlight_color,
                linewidth=1.0,
                zorder=10,
            )
            ax.add_patch(circle)

        # Display column vectors
        if show_vectors:
            num_locations = len(x_coords_img)

            # Use passed fixed heights, or calculate if not provided
            if fixed_vector_height is None:
                if max_components is not None and max_components > 0:
                    # Calculate a reasonable default height
                    fixed_vector_height = 0.15
                else:
                    fixed_vector_height = 0.15

            if fixed_cell_height is None:
                if max_components is not None and max_components > 0:
                    fixed_cell_height = fixed_vector_height / max_components
                else:
                    fixed_cell_height = fixed_vector_height / 10  # arbitrary default

            for i in range(num_locations):
                batch_idx = int(idx_locations[0, i])
                x_coord = int(idx_locations[3, i])
                y_coord = int(idx_locations[2, i])

                # Get column vector
                if column_type == "before":
                    column_vec = mots_before_np[batch_idx, :, y_coord, x_coord]
                    vec_color = color_before
                else:
                    column_vec = mots_after_np[batch_idx, :, y_coord, x_coord]
                    vec_color = color_after[i] if i < len(color_after) else "black"

                # Position vector adjacent to its corresponding circle
                # Place vector to the right of the circle
                circle_x = x_coords_plot[i]
                circle_y = y_coords_plot[i]

                # Vector dimensions
                num_components = (
                    max_components if max_components is not None else len(column_vec)
                )
                cell_height = fixed_cell_height
                cell_width = cell_height * 3
                vec_width = cell_width
                vec_height = fixed_vector_height

                # Position vector to the right of the circle with a small gap
                gap = 0.03
                vec_left = circle_x + gap
                vec_right = vec_left + vec_width

                # Center vector vertically on the circle
                vec_center_y = circle_y
                vec_bottom = vec_center_y - vec_height / 2
                vec_top = vec_center_y + vec_height / 2

                # Plot vector cells using fixed sizing
                vec_center_x = (vec_left + vec_right) / 2

                # Plot cells for the actual vector, but use max_components for sizing
                for j in range(num_components):
                    # Get value if within vector bounds, otherwise 0
                    val = column_vec[j] if j < len(column_vec) else 0.0
                    cell_left = vec_center_x - cell_width / 2
                    cell_right = vec_center_x + cell_width / 2
                    cell_bottom = vec_top - (j + 1) * cell_height
                    cell_top = cell_bottom + cell_height

                    if abs(val) > 1e-6:
                        fill_color = vec_color
                        edge_color = vec_color
                        alpha = 0.7
                    else:
                        fill_color = "white"
                        edge_color = "gray"
                        alpha = 1.0

                    cell_rect = Rectangle(
                        (cell_left, cell_bottom),
                        cell_width,
                        cell_height,
                        fill=True,
                        facecolor=fill_color,
                        edgecolor=edge_color,
                        linewidth=0.5,
                        alpha=alpha,
                        zorder=6,
                    )
                    ax.add_patch(cell_rect)

                # Annotate symbol
                if column_type == "before":
                    symbol = symbol_before
                else:
                    symbol = (
                        symbols_after[i]
                        if symbols_after and i < len(symbols_after)
                        else None
                    )

                if symbol is not None:
                    ax.text(
                        (vec_left + vec_right) / 2,
                        vec_top + 0.01,
                        str(symbol),
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color=vec_color,
                        fontweight="bold",
                        usetex=False,
                        zorder=8,
                    )

                # Store vector top position for arrow drawing (aligned with top of vector)
                vector_positions.append((vec_center_x, vec_top))

        # Add white image with border below the vectors for both columns to preserve aspect ratio
        # (Coordinates already calculated earlier for motif scatterplot)
        if bottom_image_bottom is not None:
            # Create a white image with the same shape as x
            white_image = np.ones(x.shape[:2]) * 255  # White image
            ax.imshow(
                white_image,
                cmap="gray",
                aspect="equal",
                alpha=1.0,
                vmin=0,
                extent=[left_x, right_x, bottom_image_bottom, bottom_image_top],
                origin="upper",
            )
            # Add a border rectangle on top
            rect = Rectangle(
                (left_x, bottom_image_bottom),
                right_x - left_x,
                bottom_image_top - bottom_image_bottom,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )
            ax.add_patch(rect)

        # Render output
        if show_output:

            def tokens_to_string(tokens):
                if hasattr(tokens, "__iter__") and not isinstance(tokens, str):
                    try:
                        return "".join(
                            [
                                tok.code
                                if hasattr(tok, "code")
                                else (tok.name if hasattr(tok, "name") else str(tok))
                                for tok in tokens
                            ]
                        )
                    except:
                        return str(tokens)
                return str(tokens)

            if column_type == "before":
                output_tokens = y_before
            else:
                output_tokens = y_after

            if output_tokens is not None:
                output_str = tokens_to_string(output_tokens)
                max_len = len(output_str)

                # Position output text below the bottom image
                # Bottom image bottom is at approximately 0.17 (0.50 - 0.33)
                # Place output text below with some gap, moved down further
                output_y = 0.01
                output_x_start = 0.1
                output_width = 0.8
                char_width = output_width / max_len if max_len > 0 else 0.01

                # Find changed positions and map tokens to locations
                changed_positions = set()
                token_to_location_map = {}  # Maps token index to location index

                if y_before is not None and y_after is not None:
                    if len(y_before) == len(y_after):
                        for i, (tok_before, tok_after) in enumerate(
                            zip(y_before, y_after)
                        ):
                            before_name = (
                                tok_before.name
                                if hasattr(tok_before, "name")
                                else str(tok_before)
                            )
                            after_name = (
                                tok_after.name
                                if hasattr(tok_after, "name")
                                else str(tok_after)
                            )
                            if before_name != after_name:
                                changed_positions.add(i)
                                # Map this change to a location by matching symbols
                                before_symbol = str(before_name) if before_name else ""
                                after_symbol = str(after_name) if after_name else ""

                                # Match if before_symbol matches symbol_before and after_symbol matches one in symbols_after
                                if before_symbol == str(symbol_before):
                                    # Find which location this corresponds to by matching after_symbol
                                    location_idx = None
                                    for loc_idx, sym_after in enumerate(symbols_after):
                                        if str(sym_after) == after_symbol:
                                            # Check if this location hasn't been used yet
                                            if (
                                                loc_idx
                                                not in token_to_location_map.values()
                                            ):
                                                location_idx = loc_idx
                                                break

                                    if location_idx is not None:
                                        token_to_location_map[i] = location_idx

                # Render output text
                char_idx = 0
                for i, tok in enumerate(output_tokens):
                    tok_str = (
                        tok.code
                        if hasattr(tok, "code")
                        else (tok.name if hasattr(tok, "name") else str(tok))
                    )
                    is_changed = i in changed_positions

                    if column_type == "before":
                        text_color = color_before if is_changed else "black"
                    else:
                        if is_changed:
                            # Get color for this location using the mapping
                            if i in token_to_location_map:
                                location_idx = token_to_location_map[i]
                                if location_idx < len(color_after):
                                    text_color = color_after[location_idx]
                                else:
                                    text_color = "black"
                            else:
                                text_color = "black"
                        else:
                            text_color = "black"

                    # Store token start position
                    token_start_x = output_x_start + char_idx * char_width

                    for char in tok_str:
                        char_x = output_x_start + char_idx * char_width
                        ax.text(
                            char_x,
                            output_y,
                            char,
                            ha="left",
                            va="center",  # Center vertically to align with label
                            fontsize=10,
                            fontfamily="monospace",
                            color=text_color,
                            fontweight="bold" if is_changed else "normal",
                            usetex=False,
                            zorder=8,
                        )
                        char_idx += 1

                    # Store position for changed tokens
                    if is_changed:
                        token_end_x = output_x_start + char_idx * char_width
                        token_center_x = (token_start_x + token_end_x) / 2
                        output_token_positions[i] = (
                            token_center_x,
                            output_y,
                            text_color,
                        )

        # Add labels
        if column_type == "before":
            title = "Original"
        else:  # column_type == 'after'
            title = "Perturbed"

        ax.text(
            0.5,
            1.02,
            title,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            zorder=11,
        )

        # Add section labels on the left side for 'before' column, rotated 90 degrees
        if column_type == "before":
            # Input label (top image section)
            input_center_y = (bottom_y + top_y) / 2  # Center of top image
            ax.text(
                -0.05,
                input_center_y,
                "Input",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                rotation=90,
                zorder=11,
            )

            # Motifs label (bottom image/motif scatterplot section)
            # bottom_image_top = 0.50, bottom_image_bottom calculated earlier
            if bottom_image_bottom is not None:
                motifs_center_y = (bottom_image_bottom + bottom_image_top) / 2
                ax.text(
                    -0.05,
                    motifs_center_y,
                    "Motifs",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    rotation=90,
                    zorder=11,
                )

            # Output label (output text section)
            output_y_pos = 0.01  # Output text y position (moved down further)
            ax.text(
                -0.05,
                output_y_pos,
                "Output",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                rotation=90,
                zorder=11,
            )

        return vector_positions, output_token_positions

    # Calculate maximum number of components across all vectors (before and after)
    # to ensure consistent sizing
    max_components = 0
    num_locations = idx_locations.shape[1]
    for i in range(num_locations):
        batch_idx = int(idx_locations[0, i])
        x_coord = int(idx_locations[3, i])
        y_coord = int(idx_locations[2, i])
        vec_before = mots_before_np[batch_idx, :, y_coord, x_coord]
        vec_after = mots_after_np[batch_idx, :, y_coord, x_coord]
        max_components = max(max_components, len(vec_before), len(vec_after))

    # Calculate fixed vector and cell heights once to ensure both columns use the same sizes
    vector_area_center = 0.5
    vector_height_total = 0.25
    vector_area_top = vector_area_center + vector_height_total / 2
    vector_area_bottom = vector_area_center - vector_height_total / 2
    vector_height = vector_area_top - vector_area_bottom
    cols = min(4, num_locations)
    rows = (num_locations + cols - 1) // cols
    vector_height_per_location = vector_height / rows
    fixed_vector_height = vector_height_per_location * 0.8
    fixed_cell_height = (
        fixed_vector_height / max_components
        if max_components > 0
        else fixed_vector_height / 10
    )

    # Plot Before column
    before_vectors, before_output_tokens = plot_column(
        ax_before,
        mots_before_np,
        True,
        True,
        True,
        "before",
        max_components,
        fixed_vector_height,
        fixed_cell_height,
    )

    # Plot After column
    after_vectors, after_output_tokens = plot_column(
        ax_after,
        mots_after_np,
        True,
        True,
        True,
        "after",
        max_components,
        fixed_vector_height,
        fixed_cell_height,
    )

    # Draw arrows from Before to After (left to right)
    # Use FancyArrowPatch to draw arrows between subplots with arcs to avoid intersections
    for i, (before_vec_pos, after_vec_pos) in enumerate(
        zip(before_vectors, after_vectors)
    ):
        # Get color for this arrow
        arrow_color = color_after[i] if i < len(color_after) else "black"

        # Convert coordinates from axes to figure
        posA = ax_before.transData.transform((before_vec_pos[0], before_vec_pos[1]))
        posB = ax_after.transData.transform((after_vec_pos[0], after_vec_pos[1]))
        posA = fig.transFigure.inverted().transform(posA)
        posB = fig.transFigure.inverted().transform(posB)

        # Use an upward arc for all vector arrows
        # Negative radius creates an upward arc
        rad = -0.3

        # Draw arrow using FancyArrowPatch with upward arced connection
        arrow = FancyArrowPatch(
            posA,
            posB,
            transform=fig.transFigure,
            patchA=None,
            patchB=None,
            arrowstyle="->",
            mutation_scale=20,
            linewidth=1.5,
            color=arrow_color,
            alpha=0.7,
            zorder=15,
            connectionstyle=f"arc3,rad={rad}",
        )
        fig.patches.append(arrow)

    # Draw arrows from Input to Motifs and Motifs to Output in the 'before' column
    # Calculate bottom image coordinates (same as used in plot_column)
    # Use the same coordinate values as in plot_column
    top_y = 1.0
    bottom_y = 0.67
    image_height_bottom = top_y - bottom_y  # Same height as top image
    bottom_image_top = 0.50  # Position higher up, below vectors
    bottom_image_bottom = bottom_image_top - image_height_bottom

    # Arrow from Input (below bottom of top image) to Motifs (top of bottom image area)
    # Use center horizontally, straight down
    arrow_x = 0.5
    # Start slightly below the bottom edge of the image
    arrow_start_y = bottom_y - 0.06
    arrow_end_y = bottom_image_top
    posA = ax_before.transData.transform((arrow_x, arrow_start_y))
    posB = ax_before.transData.transform((arrow_x, arrow_end_y))
    posA = fig.transFigure.inverted().transform(posA)
    posB = fig.transFigure.inverted().transform(posB)

    # No connectionstyle - straight arrow
    arrow = FancyArrowPatch(
        posA,
        posB,
        transform=fig.transFigure,
        patchA=None,
        patchB=None,
        arrowstyle="->",
        mutation_scale=20,
        linewidth=1.5,
        color="black",
        alpha=0.7,
        zorder=15,
    )
    fig.patches.append(arrow)

    # Add label $\hat g$ to the Input to Motifs arrow
    # Use the exact same y-coordinates as the arrow, then move up by ~0.5 inch
    arrow_mid_y = (arrow_start_y + arrow_end_y) / 2
    # Figure height is 8 inches, y-axis spans ~0.99 units, so 0.5 inch ≈ 0.062 units
    label_y_hat_g = arrow_mid_y + 0.03  # Move up
    label_x = 0.4  # Position to the right of the arrow
    ax_before.text(
        label_x,
        label_y_hat_g,
        r"$\hat g$",
        ha="left",
        va="center",
        fontsize=24,
        fontweight="bold",
        zorder=16,
    )

    # Arrow from Motifs (bottom of bottom image area) to above Output (output text) - Before column
    # End above the output text, not at it
    output_y = 0.04
    output_arrow_end_y = output_y + 0.05  # End above the output text
    motifs_output_arrow_start_y = bottom_image_bottom + 0.11
    motifs_output_arrow_end_y = bottom_image_bottom + 0.02
    posA = ax_before.transData.transform((arrow_x, motifs_output_arrow_start_y))
    posB = ax_before.transData.transform((arrow_x, motifs_output_arrow_end_y))
    posA = fig.transFigure.inverted().transform(posA)
    posB = fig.transFigure.inverted().transform(posB)

    # No connectionstyle - straight arrow
    arrow = FancyArrowPatch(
        posA,
        posB,
        transform=fig.transFigure,
        patchA=None,
        patchB=None,
        arrowstyle="->",
        mutation_scale=20,
        linewidth=1.5,
        color="black",
        alpha=0.7,
        zorder=15,
    )
    fig.patches.append(arrow)

    # Add label $\hat h$ to the Motifs to Output arrow
    # Use the exact same y-coordinates as the arrow, then move down by ~1 inch
    arrow_mid_y = (motifs_output_arrow_start_y + motifs_output_arrow_end_y) / 2
    # Figure height is 8 inches, y-axis spans ~0.99 units, so 1 inch ≈ 0.124 units
    label_y_hat_h = arrow_mid_y - 0.15  # Move down
    label_x = 0.4  # Position to the right of the arrow
    ax_before.text(
        label_x,
        label_y_hat_h,
        r"$\hat h$",
        ha="left",
        va="center",
        fontsize=24,
        fontweight="bold",
        zorder=16,
    )

    # Arrow from Motifs (bottom of bottom image area) to above Output (output text) - After column
    motifs_output_arrow_start_y_after = bottom_image_bottom + 0.11
    motifs_output_arrow_end_y_after = bottom_image_bottom + 0.02
    posA = ax_after.transData.transform((arrow_x, motifs_output_arrow_start_y_after))
    posB = ax_after.transData.transform((arrow_x, motifs_output_arrow_end_y_after))
    posA = fig.transFigure.inverted().transform(posA)
    posB = fig.transFigure.inverted().transform(posB)

    # No connectionstyle - straight arrow
    arrow = FancyArrowPatch(
        posA,
        posB,
        transform=fig.transFigure,
        patchA=None,
        patchB=None,
        arrowstyle="->",
        mutation_scale=20,
        linewidth=1.5,
        color="black",
        alpha=0.7,
        zorder=15,
    )
    fig.patches.append(arrow)

    # Add label $\hat h$ to the Motifs to Output arrow - After column
    # Use the exact same y-coordinates as the arrow, then move down by ~1 inch
    arrow_mid_y_after = (
        motifs_output_arrow_start_y_after + motifs_output_arrow_end_y_after
    ) / 2
    # Figure height is 8 inches, y-axis spans ~0.99 units, so 1 inch ≈ 0.124 units
    label_y_hat_h_after = arrow_mid_y_after - 0.15  # Move down (same as before column)
    label_x = 0.4  # Position to the right of the arrow (same as before column)
    ax_after.text(
        label_x,
        label_y_hat_h_after,
        r"$\hat h$",
        ha="left",
        va="center",
        fontsize=24,
        fontweight="bold",
        zorder=16,
    )

    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix=".png") as tmpfile:
        path_out = tmpfile.name
        plt.savefig(path_out)
        im = Image.open(path_out)
    # clip out whitespace before and after
    arr = np.array(im)
    # Find non-white rows and columns
    pad = 20
    non_white_rows = np.where(np.min(arr[:, :, :3], axis=(1, 2)) < 250)[0]
    non_white_cols = np.where(np.min(arr[:, :, :3], axis=(0, 2)) < 250)[0]
    if non_white_rows.size > 0 and non_white_cols.size > 0:
        top, bottom = non_white_rows[0], non_white_rows[-1] + 1
        left, right = non_white_cols[0], non_white_cols[-1] + 1
        im = im.crop((left - pad, top - pad, right + pad, bottom + pad))
    plt.close(fig)
    return im

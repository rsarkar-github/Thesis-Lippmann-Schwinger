import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid


def plot_image_xy(
        image, x0, xn, y0, yn, scale=None, vmin=None, vmax=None, sfac=1.0, clip=1.0,
        grid="off", aspect="auto", cmap="Greys", colorbar=True, cbar_size="1%", cbar_pad=0.15,
        xlabel=None, ylabel=None, fontname="Times New Roman", fontsize=15,
        nxticks=5, nyticks=5, xticklabels_fmt="{:4.1f}", yticklabels_fmt="{:4.1f}",
        draw_line_coords=None, linewidth=1, linestyle="-", linecolor="red",
        draw_line_coords_grp1=None, linewidth_grp1=1, linestyle_grp1="-", linecolor_grp1="red",
        marker_coords=None, markersize=2, markerstyle="X", markercolor="red",
        savefig_fname=None
):

    extent = [1e-3 * x0, 1e-3 * xn, 1e-3 * yn, 1e-3 * y0]
    xticks = np.arange(1e-3 * x0, 1e-3 * xn, 1e-3 * (xn - x0) / nxticks)
    xticklabels = [xticklabels_fmt.format(item) for item in xticks]
    yticks = np.arange(1e-3 * y0, 1e-3 * yn, 1e-3 * (yn - y0) / nyticks)
    yticklabels = [yticklabels_fmt.format(item) for item in yticks]

    if scale is None:
        scale = np.max(np.abs(image)) * sfac
    if vmin is None:
        vmin = -scale
    if vmax is None:
        vmax = scale

    plot = plt.imshow(image, aspect=aspect, vmin=clip * vmin, vmax=clip * vmax, cmap=cmap, extent=extent)

    if grid == "on":
        plt.grid()

    if xlabel is None:
        plt.xlabel('X position (km)', fontname=fontname, fontsize=fontsize)
    else:
        plt.xlabel(xlabel, fontname=fontname, fontsize=fontsize)

    if ylabel is None:
        plt.ylabel('Time (s)', fontname=fontname, fontsize=fontsize)
    else:
        plt.ylabel(ylabel, fontname=fontname, fontsize=fontsize)

    ax = plt.gca()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontname=fontname, fontsize=fontsize)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontname=fontname, fontsize=fontsize)

    # Optionally draw line (Group 0)
    if draw_line_coords is not None:
        for item in draw_line_coords:
            plt.plot(item[0], item[1], color=linecolor, linewidth=linewidth, linestyle=linestyle)

    # Optionally draw line (Group 1)
    if draw_line_coords_grp1 is not None:
        for item in draw_line_coords_grp1:
            plt.plot(item[0], item[1], color=linecolor_grp1, linewidth=linewidth_grp1, linestyle=linestyle_grp1)

    # Optionally draw points
    if marker_coords is not None:
        for item in marker_coords:
            plt.plot(item[0], item[1], color=markercolor, markersize=markersize, marker=markerstyle)

    # Create aligned colorbar on the right
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=cbar_size, pad=cbar_pad)
        plt.colorbar(plot, cax=cax)
        for i in cax.yaxis.get_ticklabels():
            i.set_family(fontname)
            i.set_size(fontsize)

    # Save the figure
    if savefig_fname is not None:
        plt.savefig(savefig_fname, format="pdf", bbox_inches="tight", pad_inches=0.01)

    plt.show()
    if mpl.get_backend() in ["QtAgg", "Qt4Agg"]:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    plt.close()


def plot_images_grid_xy(
        image_grid, image_titles, x0, xn, y0, yn, figsize=(30, 10), axes_pad=0.35,
        scale=None, vmin=None, vmax=None, sfac=1.0, clip=1.0,
        grid="off", aspect="auto", cmap="Greys", colorbar=True, cbar_size="1%", cbar_pad=0.15,
        xlabel=None, ylabel=None, fontname="Times New Roman", fontsize=15,
        nxticks=5, nyticks=5, xticklabels_fmt="{:4.1f}", yticklabels_fmt="{:4.1f}",
        savefig_fname=None
):
    # Get number of rows & cols
    nrows, ncols, _, _ = image_grid.shape

    # Get axis limits and labels
    extent = [1e-3 * x0, 1e-3 * xn, 1e-3 * yn, 1e-3 * y0]
    xticks = np.arange(1e-3 * x0, 1e-3 * xn, 1e-3 * (xn - x0) / nxticks)
    xticklabels = [xticklabels_fmt.format(item) for item in xticks]
    yticks = np.arange(1e-3 * y0, 1e-3 * yn, 1e-3 * (yn - y0) / nyticks)
    yticklabels = [yticklabels_fmt.format(item) for item in yticks]

    if scale is None:
        scale = np.max(np.abs(image_grid[0, 0, :, :])) * sfac
    if vmin is None:
        vmin = -scale
    if vmax is None:
        vmax = scale

    if xlabel is None:
        xlabel = "X position [km]"
    if ylabel is None:
        ylabel = "Z position [km]"

    # Create figure and image grid
    fig = plt.figure(figsize=figsize)
    img_grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(nrows, ncols),
        axes_pad=axes_pad,
        share_all=True,
        aspect=False,
        cbar_location="right",
        cbar_mode="single",
        cbar_size=cbar_size,
        cbar_pad=cbar_pad,
    )

    # Plot images
    for i in range(nrows):
        for j in range(ncols):

            idx = i * ncols + j
            ax = img_grid[idx]
            im = ax.imshow(
                np.squeeze(image_grid[i, j, :, :]),
                aspect=aspect,
                cmap=cmap,
                vmin=clip * vmin,
                vmax=clip * vmax,
                extent=extent
            )

            if grid == "on":
                ax.grid(True, color="white", linestyle="-", linewidth=0.5)

            ax.set_xticks(xticks)
            if i == nrows - 1:
                ax.set_xticklabels(xticklabels, fontname=fontname, fontsize=fontsize)
                ax.set_xlabel(xlabel, fontname=fontname, fontsize=fontsize)

            ax.set_yticks(yticks)
            if j == 0:
                ax.set_yticklabels(yticklabels, fontname=fontname, fontsize=fontsize)
                ax.set_ylabel(ylabel, fontname=fontname, fontsize=fontsize)

            ax.set_aspect(aspect)
            ax.set_title(image_titles[i][j], fontname=fontname, fontsize=fontsize)

            # Create aligned colorbar on the right
            if colorbar and idx == nrows * ncols - 1:
                ax.cax.colorbar(im)
                for i1 in ax.cax.yaxis.get_ticklabels():
                    i1.set_family(fontname)
                    i1.set_size(fontsize)

    # Save the figure
    if savefig_fname is not None:
        fig.savefig(savefig_fname, format="pdf", bbox_inches="tight", pad_inches=0.01)

    plt.show()
    if mpl.get_backend() in ["QtAgg", "Qt4Agg"]:
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    plt.close()

import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # ----------------------------------------------
    # Check arguments and read in parameters
    # ----------------------------------------------
    if len(sys.argv) < 2:
        raise ValueError("Program missing command line arguments.")
    model_mode = int(sys.argv[1])

    if model_mode == 0:
        filepath = "Lippmann-Schwinger/Data/p04a-sigsbee-new-vz-2d.npz"
    elif model_mode == 1:
        filepath = "Lippmann-Schwinger/Data/p04b-marmousi-new-vz-2d.npz"
    elif model_mode == 2:
        filepath = "Lippmann-Schwinger/Data/p04c-seiscope-new-vz-2d.npz"
    else:
        print("model mode = ", model_mode, " is not supported. Must be 0, 1, or 2.")

    # ----------------------------------------------
    # Load vz
    # ----------------------------------------------
    with np.load(filepath) as data:
        vel = data["arr_0"]

    # ----------------------------------------------
    # Set parameters
    # ----------------------------------------------
    n_ = 351
    nz_ = 251
    dx = 15.0       # something dummy (does not matter)
    dz = 15.0       # something dummy (does not matter)
    lenx = dx * (n_ - 1)
    lenz = dz * (nz_ - 1)

    # ----------------------------------------------
    # Create sources
    # ----------------------------------------------
    def create_arc_source(sigma=15):

        src = np.zeros(shape=(nz_, n_), dtype=np.float32)

        coord_x0 = lenx * 0.125
        coord_z0 = lenz * 0.875
        r0 = 500.0

        for i in range(nz_):
            for j in range(n_):

                coord_z = i * dz
                coord_x = j * dx
                r = ((coord_z - coord_z0) ** 2.0 + (coord_x - coord_x0) ** 2.0) ** 0.5
                fac = 0.5 * (((r - r0) / sigma) ** 2.0)

                if coord_x > coord_x0 and coord_z < coord_z0:
                    src[i, j] = np.exp(-fac)

        return src

    def create_gaussian_point_source(sigma=15):

        src = np.zeros(shape=(nz_, n_), dtype=np.float32)
        coord_x0 = lenx / 2.0
        coord_z0 = lenz * 0.05

        for i in range(nz_):
            for j in range(n_):

                coord_z = i * dz
                coord_x = j * dx

                r = ((coord_z - coord_z0) ** 2.0 + (coord_x - coord_x0) ** 2.0) ** 0.5
                fac = 0.5 * ((r / sigma) ** 2.0)

                src[i, j] = np.exp(-fac)

        return src

    def create_line_source():

        src = np.zeros(shape=(nz_, n_), dtype=np.float32)
        src[125:150, 25] = 1.0
        # src[25, 150:200] = 1.0
        src = gaussian_filter(src, sigma=1)

        return src

    if model_mode == 0:
        sou = create_arc_source()
        np.savez("Lippmann-Schwinger/Data/p06-sigsbee-source.npz", sou)
        savefig_fname = "Lippmann-Schwinger/Fig/p06-sigsbee-source.pdf"

    if model_mode == 1:
        sou = create_gaussian_point_source()
        np.savez("Lippmann-Schwinger/Data/p06-marmousi-source.npz", sou)
        savefig_fname = "Lippmann-Schwinger/Fig/p06-marmousi-source.pdf"

    if model_mode == 2:
        sou = create_line_source()
        np.savez("Lippmann-Schwinger/Data/p06-seiscope-source.npz", sou)
        savefig_fname = "Lippmann-Schwinger/Fig/p06-seiscope-source.pdf"


    # ----------------------------------------------
    # Overlay plots
    # ----------------------------------------------
    def plot(src, vel, fname):

        masked_src = np.ma.masked_where(src < 0.2, src)

        extent = [0, lenx / 1000.0, lenz / 1000.0, 0]
        plt.figure(figsize=(12, 3))  # define figure size
        plt.imshow(vel, cmap="jet", interpolation='nearest', extent=extent, vmin=1.5, vmax=4.5)
        cbar = plt.colorbar(aspect=10, pad=0.02)
        cbar.set_label('Vp [km/s]', labelpad=10)
        plt.imshow(masked_src, cmap="seismic", interpolation="none", extent=extent, vmin=-1, vmax=1)
        plt.xlabel(r'$x_1$ [km]')
        plt.ylabel(r'$z$ [km]')

        plt.savefig(fname, format="pdf", bbox_inches="tight", pad_inches=0.01)
        plt.show()

    plot(src=sou, vel=vel, fname=savefig_fname)

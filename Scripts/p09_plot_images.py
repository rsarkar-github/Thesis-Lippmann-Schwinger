import sys
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # ----------------------------------------------
    # Check arguments and read in parameters
    # ----------------------------------------------
    if len(sys.argv) < 3:
        raise ValueError("Program missing command line arguments.")

    model_mode = int(sys.argv[1])
    freq_mode = int(sys.argv[2])

    if model_mode == 0:
        filepath = "Lippmann-Schwinger/Data/p04a-sigsbee-new-vz-2d.npz"
        filepath1 = "Lippmann-Schwinger/Data/p04a-sigsbee-new-2d.npz"
        filepath3_ = "Lippmann-Schwinger/Data/p07-sigsbee-"
        savefig_fname = "Lippmann-Schwinger/Fig/p09-sigsbee-overlay-rhs.pdf"
        savefig_fname_sol = "Lippmann-Schwinger/Fig/p09-sigsbee-overlay-sol.pdf"
    elif model_mode == 1:
        filepath = "Lippmann-Schwinger/Data/p04b-marmousi-new-vz-2d.npz"
        filepath1 = "Lippmann-Schwinger/Data/p04b-marmousi-new-2d.npz"
        filepath3_ = "Lippmann-Schwinger/Data/p07-marmousi-"
        savefig_fname = "Lippmann-Schwinger/Fig/p09-marmousi-overlay-rhs.pdf"
        savefig_fname_sol = "Lippmann-Schwinger/Fig/p09-marmousi-overlay-sol.pdf"
    elif model_mode == 2:
        filepath = "Lippmann-Schwinger/Data/p04c-seiscope-new-vz-2d.npz"
        filepath1 = "Lippmann-Schwinger/Data/p04c-seiscope-new-2d.npz"
        filepath3_ = "Lippmann-Schwinger/Data/p07-seiscope-"
        savefig_fname = "Lippmann-Schwinger/Fig/p09-seiscope-overlay-rhs.pdf"
        savefig_fname_sol = "Lippmann-Schwinger/Fig/p09-seiscope-overlay-sol.pdf"
    else:
        raise ValueError("model mode = ", model_mode, " is not supported. Must be 0, 1, or 2.")


    if freq_mode == 0:
        freq = 5.0   # in Hz
    elif freq_mode == 1:
        freq = 7.5   # in Hz
    elif freq_mode == 2:
        freq = 10.0  # in Hz
    elif freq_mode == 3:
        freq = 15.0  # in Hz
    else:
        raise ValueError("freq mode = ", freq_mode, " is not supported. Must be 0, 1, 2, or 3.")

    # ----------------------------------------------
    # Load velocities
    # ----------------------------------------------
    with np.load(filepath) as data:
        vz = data["arr_0"]

    with np.load(filepath1) as data:
        vel = data["arr_0"]

    # ----------------------------------------------
    # Load wavefields
    # ----------------------------------------------
    with np.load(filepath3_ + "rhs-" + "{:4.2f}".format(freq) + ".npz") as data:
        rhs = data["arr_0"]

    with np.load(filepath3_ + "sol-" + "gmres" + "-" + "{:4.2f}".format(freq) + ".npz") as data:
        data_sol = data["arr_0"]

    # ----------------------------------------------
    # Plot overlays
    # ----------------------------------------------
    n_ = 351
    nz_ = 251
    dx = 15.0  # something dummy (does not matter)
    dz = 15.0  # something dummy (does not matter)
    lenx = dx * (n_ - 1)
    lenz = dz * (nz_ - 1)

    def plot(data, vel, fname):

        extent = [0, lenx / 1000.0, lenz / 1000.0, 0]
        plt.figure(figsize=(12, 3))  # define figure size
        plt.imshow(vel, cmap="jet", interpolation='nearest', extent=extent, vmin=1.5, vmax=4.5)
        cbar = plt.colorbar(aspect=10, pad=0.02)
        cbar.set_label('Vp [km/s]', labelpad=10)

        plt.imshow(data, cmap="Greys", interpolation="none", extent=extent, vmin=-5e-6, vmax=5e-6, alpha=0.5)

        plt.xlabel(r'$x_1$ [km]')
        plt.ylabel(r'$z$ [km]')

        plt.savefig(fname, format="pdf", bbox_inches="tight", pad_inches=0.01)
        plt.show()


    plot(data=np.real(rhs), vel=vz, fname=savefig_fname)
    plot(data=np.real(data_sol), vel=vel, fname=savefig_fname_sol)

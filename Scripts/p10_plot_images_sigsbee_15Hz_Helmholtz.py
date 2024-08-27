import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # ----------------------------------------------
    # Load wavefields
    # ----------------------------------------------
    filepath3_ = "Lippmann-Schwinger/Data/p08-sigsbee-"
    freq = 15

    with np.load(filepath3_ + "sol-" + "gmres" + "-" + "{:4.2f}".format(freq) + ".npz") as data:
        data_sol1 = data["arr_0"]
    with np.load(filepath3_ + "sol-" + "lsqr" + "-" + "{:4.2f}".format(freq) + ".npz") as data:
        data_sol2 = data["arr_0"]
    with np.load(filepath3_ + "sol-" + "lsmr" + "-" + "{:4.2f}".format(freq) + ".npz") as data:
        data_sol3 = data["arr_0"]

    # ----------------------------------------------
    # Plot overlays
    # ----------------------------------------------
    n_ = 351
    nz_ = 251
    dx = 15.0  # something dummy (does not matter)
    dz = 15.0  # something dummy (does not matter)
    lenx = dx * (n_ - 1)
    lenz = dz * (nz_ - 1)

    def plot(data, savefig_name):

        extent = [0, lenx / 1000.0, lenz / 1000.0, 0]
        plt.imshow(data, cmap="Greys", interpolation="none", extent=extent, vmin=-5e-6, vmax=5e-6)
        plt.xlabel(r'$x_1$ [km]')
        plt.ylabel(r'$z$ [km]')

        plt.savefig(savefig_name, format="pdf", bbox_inches="tight", pad_inches=0.01)
        plt.show()

    plot(data=np.real(data_sol1), savefig_name="Lippmann-Schwinger/Fig/p10-sigsbee-sol-gmres-15.00.pdf")
    plot(data=np.real(data_sol2), savefig_name="Lippmann-Schwinger/Fig/p10-sigsbee-sol-lsqr-15.00.pdf")
    plot(data=np.real(data_sol3), savefig_name="Lippmann-Schwinger/Fig/p10-sigsbee-sol-lsmr-15.00.pdf")

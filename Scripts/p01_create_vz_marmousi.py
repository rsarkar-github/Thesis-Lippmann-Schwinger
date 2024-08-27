import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Load Marmousi
    with np.load("Lippmann-Schwinger/Data/marmousi-vp.npz") as data:
        vp = data["arr_0"]

    # Get shape (500, 174)
    nx, nz = vp.shape

    # ------------------------
    # Plot Marmousi-2 vp-model
    # ------------------------

    dx = 20
    dz = 20

    # Define xmax, zmax and model extension
    xmax = nx * dx
    zmax = nz * dz
    extent = [0, xmax, zmax, 0]

    fig = plt.figure(figsize=(12, 3))  # define figure size
    image = plt.imshow(vp.T / 1000, cmap="jet", interpolation='nearest', extent=extent)

    cbar = plt.colorbar(aspect=10, pad=0.02)
    cbar.set_label('Vp [km/s]', labelpad=10)
    plt.title('Marmousi-2 model')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.show()

    # ------------------------
    # Perform mean operation
    # ------------------------

    vp_mean = np.mean(vp, axis=0)
    vp_mean_2d = vp * 1.0
    for i in range(nx):
        vp_mean_2d[i, :] = vp_mean

    # Save Marmousi v(z)
    np.savez("Lippmann-Schwinger/Data/marmousi-vp-vz.npz", vp_mean)
    np.savez("Lippmann-Schwinger/Data/marmousi-vp-vz-2d.npz", vp_mean_2d)

    # ---------------------------
    # Plot Marmousi-2 vp-vz-model
    # ---------------------------

    with np.load("Lippmann-Schwinger/Data/marmousi-vp-vz-2d.npz") as data:
        vp_mean_2d = data["arr_0"]

    fig = plt.figure(figsize=(12, 3))  # define figure size
    image = plt.imshow(vp_mean_2d.T / 1000, cmap="jet", interpolation='nearest', extent=extent)

    cbar = plt.colorbar(aspect=10, pad=0.02)
    cbar.set_label('Vp [km/s]', labelpad=10)
    plt.title('Marmousi-2 v(z) model')
    plt.xlabel('x$_1$ [m]')
    plt.ylabel('z [m]')

    savefig_fname = "Lippmann-Schwinger/Fig/marmousi-vp-vz-2d.pdf"
    plt.savefig(savefig_fname, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.show()

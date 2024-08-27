import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from ..Utilities.Utils import cosine_taper_2d


def estimate_vz(vel_slice_, cutoff):

    # Flatten array, and count all values greater than cutoff
    v = vel_slice_.flatten()
    n = v.shape[0]

    cnt = 0
    for ii in range(n):
        if v[ii] >= cutoff:
            v[ii] = 0.
            cnt += 1

    cnt_valid = n - cnt

    return np.sum(v) / cnt_valid


if __name__ == "__main__":

    # ----------------------
    # Load Sigsbee
    # ----------------------
    with np.load("Lippmann-Schwinger/Data/sigsbee.npz") as data:
        vp = data["arr_0"]
    vp = 0.3048 * vp / 1000.0

    # ----------------------
    # Perform cropping
    # ----------------------
    vp_new = vp[166:1067:2, 0:581:2]
    vp_new = vp_new.T

    # ---------------------------------------------------
    # Interpolator
    # ---------------------------------------------------
    dz = 15
    dx = 15
    nz_new = 251
    nx_new = 351
    zmax = (nz_new - 1) * dz
    xmax = (nx_new - 1) * dx
    extent = [0, xmax / 1000.0, zmax / 1000.0, 0]

    def func_interp(vel, nz_new, nx_new):
        """
        Vel must have shape (nz, nx).

        :param vel: Velocity to interpolate.
        :param nz_new: New number of nz gridpoints.
        :param nx_new: New number of nx gridpoints.

        :return: Interpolated velocity on 15m x 15m grid.
        """

        nz_vel = vel.shape[0]
        nx_vel = vel.shape[1]

        zgrid_input = np.linspace(start=0, stop=zmax, num=nz_vel, endpoint=True).astype(np.float64)
        xgrid_input = np.linspace(start=0, stop=xmax, num=nx_vel, endpoint=True).astype(np.float64)
        interp = RegularGridInterpolator((zgrid_input, xgrid_input), vel.astype(np.float64))

        vel_interp = np.zeros(shape=(nz_new, nx_new), dtype=np.float64)

        for i1 in range(nz_new):
            for i2 in range(nx_new):
                point = np.array([i1 * dz, i2 * dx])
                vel_interp[i1, i2] = interp(point)

        return vel_interp

    print("Creating Sigsbee model...")
    vp_interp = func_interp(vp_new, nz_new, nx_new)
    print("Interp shape: ", vp_interp.shape)

    fig = plt.figure(figsize=(12, 3))  # define figure size
    image = plt.imshow(vp_interp, cmap="jet", interpolation='nearest', extent=extent, vmin=1.5, vmax=4.5)
    cbar = plt.colorbar(aspect=10, pad=0.02)
    cbar.set_label('Vp [km/s]', labelpad=10)
    plt.title('Sigsbee model')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.show()

    # Turn this on to save intermediate Sigsbee model
    # np.savez("Lippmann-Schwinger/Data/sigsbee-new.npz", vp_interp)

    # --------------------------------------------------------
    # Estimating v(z)
    # --------------------------------------------------------

    vp_mean = np.zeros(shape=(nz_new,), dtype=np.float32)
    for i in range(nz_new):
        vp_mean[i] = estimate_vz(vel_slice_=vp_interp[i, :], cutoff=3.4)

    vp_mean_2d = vp_interp * 1.0
    for i in range(nx_new):
        vp_mean_2d[:, i] = vp_mean
    print(vp_mean_2d.shape)

    fig = plt.figure(figsize=(12, 3))  # define figure size
    image = plt.imshow(vp_mean_2d, cmap="jet", interpolation='nearest', extent=extent, vmin=1.5, vmax=4.5)
    cbar = plt.colorbar(aspect=10, pad=0.02)
    cbar.set_label('Vp [km/s]', labelpad=10)
    plt.xlabel(r'$x_1$ [km]')
    plt.ylabel(r'$z$ [km]')

    np.savez("Lippmann-Schwinger/Data/p04a-sigsbee-new-vz-2d.npz", vp_mean_2d)
    savefig_fname = "Lippmann-Schwinger/Fig/p04a-sigsbee-new-vz-2d.pdf"
    plt.savefig(savefig_fname, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.show()

    # --------------------------------------------------------
    # Create perturbation
    # --------------------------------------------------------

    # Create perturbation and apply taper
    skip = 5
    vp_diff = vp_interp - vp_mean_2d
    vp_diff1 = vp_diff[skip:nz_new - skip, skip:nx_new - skip] * 1.0
    cosine_taper_2d(array2d=vp_diff1, ncells_pad_x=20, ncells_pad_z=20)
    vp_diff1 = vp_diff1.astype(np.float32)
    vp_diff *= 0
    vp_diff[skip:nz_new - skip, skip:nx_new - skip] += vp_diff1
    vp_total = vp_diff + vp_mean_2d

    fig = plt.figure(figsize=(12, 3))  # define figure size
    image = plt.imshow(vp_total, cmap="jet", interpolation='nearest', extent=extent, vmin=1.5, vmax=4.5)
    cbar = plt.colorbar(aspect=10, pad=0.02)
    cbar.set_label('Vp [km/s]', labelpad=10)
    plt.xlabel(r'$x_1$ [km]')
    plt.ylabel(r'$z$ [km]')

    np.savez("Lippmann-Schwinger/Data/p04a-sigsbee-new-2d.npz", vp_total)
    savefig_fname = "Lippmann-Schwinger/Fig/p04a-sigsbee-new-2d.pdf"
    plt.savefig(savefig_fname, format="pdf", bbox_inches="tight", pad_inches=0.01)
    plt.show()

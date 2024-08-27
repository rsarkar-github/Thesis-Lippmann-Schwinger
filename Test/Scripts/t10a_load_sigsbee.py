import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Load Sigsbee
    with np.load("Lippmann-Schwinger/Data/sigsbee.npz") as data:
        vp = data["arr_0"]
    vp = 0.3048 * vp / 1000.0

    # Get shape (1067, 601)
    nx, nz = vp.shape
    print("Original shape: ", vp.shape)

    # ------------------------
    # Plot Sigsbee vp-model
    # ------------------------

    dx = 20
    dz = 20

    # Define xmax, zmax and model extension
    xmax = nx * dx
    zmax = nz * dz
    extent = [0, xmax, zmax, 0]

    fig = plt.figure(figsize=(12, 3))  # define figure size
    image = plt.imshow(vp.T, cmap="jet", interpolation='nearest', extent=extent)

    cbar = plt.colorbar(aspect=10, pad=0.02)
    cbar.set_label('Vp [km/s]', labelpad=10)
    plt.title('Sigsbee model')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.show()

    # ----------------------
    # Perform cropping
    # ----------------------
    vp_new = vp[166:1067:2, 0:581:2]
    print("New shape: ", vp_new.shape)

    # Define xmax, zmax and model extension
    dx = 15
    dz = 15
    nx, nz = vp_new.shape
    xmax = nx * dx
    zmax = nz * dz
    extent = [0, xmax, zmax, 0]

    fig = plt.figure(figsize=(12, 3))  # define figure size
    image = plt.imshow(vp_new.T, cmap="jet", interpolation='nearest', extent=extent)
    cbar = plt.colorbar(aspect=10, pad=0.02)
    cbar.set_label('Vp [km/s]', labelpad=10)
    plt.title('Sigsbee model')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.show()

    # ---------------------------------------------------
    # Interpolator
    # ---------------------------------------------------
    vp_new = vp_new.T
    dz = 15
    dx = 15
    nz_new = 301
    nx_new = 401
    zmax = (nz_new - 1) * dz
    xmax = (nx_new - 1) * dx
    extent = [0, xmax, zmax, 0]

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

    vp_interp = func_interp(vp_new, nz_new, nx_new)
    print("Interp shape: ", vp_interp.shape)

    fig = plt.figure(figsize=(12, 3))  # define figure size
    image = plt.imshow(vp_interp, cmap="jet", interpolation='nearest', extent=extent)
    cbar = plt.colorbar(aspect=10, pad=0.02)
    cbar.set_label('Vp [km/s]', labelpad=10)
    plt.title('Sigsbee model')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.show()

    np.savez("Lippmann-Schwinger/Data/sigsbee_new.npz", vp_interp)

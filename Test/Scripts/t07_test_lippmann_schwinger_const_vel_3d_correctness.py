import numpy as np
import time
from scipy import special
from matplotlib import pyplot as plt
from ...Solver.ScatteringIntegralConstantVelStorageOptimized import TruncatedKernelConstantVel3d


if __name__ == "__main__":

    sigma = 0.05
    n = 201
    d = 1.0 / (n - 1)
    precision = np.complex64
    f = 10.0
    v0 = 1.0
    k = 2 * np.pi * f / v0

    # Compute analytical solution
    # Correct expression (lap + k^2)u = f
    def analytical():

        sol_true = np.zeros(shape=(n, n, n), dtype=precision)

        j = complex(0, 1)
        for i1 in range(n):
            for i2 in range(n):
                for i3 in range(n):

                    coord_x = -0.5 + i1 * d
                    coord_y = -0.5 + i2 * d
                    coord_z = -0.5 + i3 * d
                    r = (coord_x ** 2 + coord_y ** 2 + coord_z ** 2) ** 0.5

                    if r <= 1e-6:
                        sol_true[i1, i2, i3] = 1.0

                    else:
                        t = ((sigma ** 2) * k * j - r) / (sigma * np.sqrt(2))
                        t = special.erf(t)
                        t = t * np.exp(-j * k * r)
                        t = np.real(t) - j * np.sin(k * r)
                        t = t * np.exp(-0.5 * k * k * sigma * sigma) / (4 * np.pi * r)
                        sol_true[i1, i2, i3] = t

        return sol_true

    sol_true = analytical()
    scale = 1e-3
    plt.imshow(np.real(sol_true[:, :, int(n / 2)]), cmap="Greys", vmin=-scale, vmax=scale)
    plt.colorbar()
    plt.show()

    # Compute Lippmann-Schwinger solver
    def lippmann_schwinger():
        op = TruncatedKernelConstantVel3d(n=n, k=k, precision=precision)

        grid = np.linspace(start=-0.5, stop=0.5, num=n, endpoint=True)
        x1, y1, z1 = np.meshgrid(grid, grid, grid, indexing="ij")
        u = np.exp(-1.0 * (x1 ** 2 + y1 ** 2 + z1 ** 2) / (2 * sigma * sigma))
        u *= (1.0 / (sigma ** 3)) / ((2 * np.pi) ** 1.5)
        u = u.astype(precision)
        sol1 = u * 0

        start_t = time.time()
        op.convolve_kernel(u=u, output=sol1)
        end_t = time.time()
        print("Total time to execute convolution: ", "{:4.2f}".format(end_t - start_t), " s \n")

        return sol1

    sol1 = lippmann_schwinger()
    sol1[int(n / 2), int(n / 2), int(n / 2)] = 1.0  # Singular point
    plt.imshow(np.real(sol1[:, :, int(n/2)]), cmap="Greys", vmin=-scale, vmax=scale)
    plt.colorbar()
    plt.show()

    print("Relative error = ", np.linalg.norm(sol1 - sol_true) / np.linalg.norm(sol1))

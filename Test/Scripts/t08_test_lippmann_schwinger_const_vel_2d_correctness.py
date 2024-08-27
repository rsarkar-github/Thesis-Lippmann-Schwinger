import numpy as np
import time
from scipy.sparse.linalg import splu
from matplotlib import pyplot as plt
from ...Solver.HelmholtzOperators import create_helmholtz2d_matrix
from ...Solver.ScatteringIntegralConstantVelStorageOptimized import TruncatedKernelConstantVel2d


if __name__ == "__main__":

    sigma = 0.03
    n = 201
    d = 1.0 / (n - 1)
    precision = np.complex64
    f = 15.0
    v0 = 1.0
    k = 2 * np.pi * f / v0

    # Compute Lippmann-Schwinger solver
    def lippmann_schwinger():
        op = TruncatedKernelConstantVel2d(n=n, k=k, precision=precision)

        grid = np.linspace(start=-0.5, stop=0.5, num=n, endpoint=True)
        x1, z1 = np.meshgrid(grid, grid, indexing="ij")
        u = np.exp(-1.0 * (x1 ** 2 + z1 ** 2) / (2 * sigma * sigma))
        u = u.astype(precision)
        sol1 = u * 0

        start_t = time.time()
        op.convolve_kernel(u=u, output=sol1)
        end_t = time.time()
        print("Total time to execute convolution: ", "{:4.2f}".format(end_t - start_t), " s \n")

        return sol1, u

    sol1, u = lippmann_schwinger()
    scale = 1e-4
    plt.imshow(np.real(sol1), cmap="Greys", vmin=-scale, vmax=scale)
    plt.show()

    def helmholtz():

        pml_cells = 10
        n1 = 221
        vel = np.zeros(shape=(n1, n1), dtype=np.float32) + v0
        omega = 2 * np.pi * f
        a1 = d * (n1 - 1)

        mat = create_helmholtz2d_matrix(
            a1=a1,
            a2=a1,
            pad1=pml_cells,
            pad2=pml_cells,
            omega=omega,
            precision=precision,
            vel=vel,
            pml_damping=50.0,
            adj=False,
            warnings=True
        )
        u1 = np.zeros(shape=(n1, n1), dtype=precision)
        u1[10:211, 10:211] = u

        mat_lu = splu(mat)
        sol = mat_lu.solve(np.reshape(u1, newshape=(n1 * n1, 1)))
        sol2 = np.reshape(sol, newshape=(n1, n1))[10:211, 10:211]

        return sol2

    sol2 = helmholtz()
    plt.imshow(np.real(sol2), cmap="Greys", vmin=-scale, vmax=scale)
    plt.show()

    print("Relative error = ", np.linalg.norm(sol1 - sol2) / np.linalg.norm(sol1))

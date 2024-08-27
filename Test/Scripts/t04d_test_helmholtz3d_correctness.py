# Estimated runtime: 5-6hrs
# Performs about 3017 GMRES iterations

import numpy as np
import time
from scipy.sparse.linalg import lsqr, gmres
from matplotlib import pyplot as plt
from ...Solver.HelmholtzOperators import create_helmholtz3d_matrix
from ...Solver.ScatteringIntegralConstantVelStorageOptimized import TruncatedKernelConstantVel3d


if __name__ == "__main__":

    # Lippmann-Schwinger solver
    n = 401
    d = 1.0 / (n - 1)
    precision = np.complex64
    f = 10.0
    v0 = 1.0
    k = 2 * np.pi * f / v0
    sigma = 0.03

    def create_source():

        grid = np.linspace(start=-0.5, stop=0.5, num=n, endpoint=True)
        x1, y1, z1 = np.meshgrid(grid, grid, grid, indexing="ij")
        u = np.exp(-1.0 * (x1 ** 2 + y1 ** 2 + z1 ** 2) / (2 * sigma * sigma))
        u = u.astype(precision)

        return u

    u = create_source()

    def lippmann_schwinger():

        op = TruncatedKernelConstantVel3d(n=n, k=k, precision=precision)
        sol = u * 0

        start_t = time.time()
        op.convolve_kernel(u=u, output=sol)
        end_t = time.time()
        print("Total time to execute convolution: ", "{:4.2f}".format(end_t - start_t), " s \n")

        return sol

    sol1 = lippmann_schwinger()
    sol1_plot = sol1[:, :, int(n/2)]

    scale = 1e-4
    plt.imshow(np.real(sol1_plot), cmap="Greys", vmin=-scale, vmax=scale)
    plt.show()


    def make_callback():
        closure_variables = dict(counter=0, residuals=[])

        def callback(residuals):
            closure_variables["counter"] += 1
            closure_variables["residuals"].append(residuals)
            print(closure_variables["counter"], residuals)

        return callback

    def helmholtz3d():

        pml_cells = 10
        n1 = n + 2 * pml_cells
        n2 = n + 2 * pml_cells
        n3 = n + 2 * pml_cells
        vel_array_3d = np.zeros(shape=(n1, n2, n3), dtype=np.float32) + v0
        omega = 2 * np.pi * f
        a1 = d * (n1 - 1)
        a2 = d * (n2 - 1)
        a3 = d * (n3 - 1)

        mat_3d = create_helmholtz3d_matrix(
            a1=a1,
            a2=a2,
            a3=a3,
            pad1=pml_cells,
            pad2=pml_cells,
            pad3=pml_cells,
            omega=omega,
            precision=precision,
            vel=vel_array_3d,
            pml_damping=50.0,
            adj=False,
            warnings=False
        )

        u1 = np.zeros(shape=(n1, n2, n3), dtype=precision)
        u1[pml_cells:pml_cells+n, pml_cells:pml_cells+n, pml_cells:pml_cells+n] += u

        tol = 1e-5

        # GMRES
        sol, exitcode = gmres(
            mat_3d,
            np.reshape(u1, newshape=(n1 * n2 * n3, 1)),
            maxiter=10000,
            restart=20,
            callback=make_callback(),
            tol=tol
        )
        print("\nExitcode ", exitcode)

        sol = np.reshape(sol, newshape=(n1, n2, n3))

        return sol[pml_cells:pml_cells+n, pml_cells:pml_cells+n, pml_cells:pml_cells+n]

    sol2 = helmholtz3d()
    sol2_plot = sol2[:, :, int(n/2)]
    plt.imshow(np.real(sol2_plot), cmap="Greys", vmin=-scale, vmax=scale)
    plt.show()

    print("Relative error = ", np.linalg.norm(sol1 - sol2) / np.linalg.norm(sol1))

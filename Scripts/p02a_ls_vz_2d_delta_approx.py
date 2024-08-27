import numpy as np
import time
from scipy.sparse.linalg import splu
from matplotlib import pyplot as plt
from ..Solver.HelmholtzOperators import create_helmholtz2d_matrix
from ..Solver.ScatteringIntegralGeneralVz import TruncatedKernelGeneralVz2d


if __name__ == "__main__":

    # Define parameters
    sigma = 0.03
    n = 201
    d = 1.0 / (n - 1)
    precision = np.complex64
    f = 15.0
    v0 = 1.0
    omega = 2 * np.pi * f

    # Compute source
    grid = np.linspace(start=-0.5, stop=0.5, num=n, endpoint=True)
    x1, z1 = np.meshgrid(grid, grid, indexing="ij")
    u = np.exp(-1.0 * (x1 ** 2 + z1 ** 2) / (2 * sigma * sigma))
    u = u.astype(precision)

    # Compute Helmholtz solution and plot
    def helmholtz():

        pml_cells = 10
        n1 = 221
        vel = np.zeros(shape=(n1, n1), dtype=np.float32) + v0
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

    extent = [0, 1.0, 1.0, 0]
    scale = 1e-5
    plt.imshow(np.real(sol2), cmap="Greys", vmin=-scale, vmax=scale, extent=extent)
    plt.title('Helmholtz')
    plt.xlabel('x')
    plt.ylabel('z')
    savefig_fname = "Lippmann-Schwinger/Fig/p02a_Helmholtz.pdf"
    plt.savefig(savefig_fname, format="pdf", bbox_inches="tight", pad_inches=0.01)

    # Compute Lippmann-Schwinger solver
    a = 0.
    b = a + (1.0 / (n - 1)) * (n - 1)
    m = 5
    vz = np.zeros(shape=(n, 1), dtype=np.float32) + v0
    sigma_list = [sigma, 3 * d/m, 2 * d/m, d/m, (d/m)/2, (d/m)/5]
    rel_error = []

    for item in sigma_list:

        # Compute LS solution for each sigma in sigma_list
        def lippmann_schwinger():
            op = TruncatedKernelGeneralVz2d(
                n=n,
                nz=n,
                a=a,
                b=b,
                k=omega,
                vz=vz,
                m=m,
                sigma=item,
                precision=precision,
                green_func_dir="Lippmann-Schwinger/Test/Data/t09",
                num_threads=8,
                verbose=False,
                light_mode=False
            )

            sol1 = u * 0
            start_t = time.time()
            op.apply_kernel(u=u, output=sol1)
            end_t = time.time()
            print("Total time to execute convolution: ", "{:4.2f}".format(end_t - start_t), " s \n")

            return sol1

        sol1 = lippmann_schwinger()

        plt.imshow(np.real(sol1), cmap="Greys", vmin=-scale, vmax=scale, extent=extent)
        plt.title('LS, ' + r'$\sigma = $' + "{:6.4f}".format(item))
        plt.xlabel('x')
        plt.ylabel('z')
        savefig_fname = "Lippmann-Schwinger/Fig/p02a_ls_sigma_" + "{:6.6f}".format(item) + ".pdf"
        plt.savefig(savefig_fname, format="pdf", bbox_inches="tight", pad_inches=0.01)

        rel_error.append(np.linalg.norm(sol1 - sol2) / np.linalg.norm(sol1))

    print(rel_error)

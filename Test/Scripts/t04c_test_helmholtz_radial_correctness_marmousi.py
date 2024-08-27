import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import gmres
from matplotlib import pyplot as plt
from ...Solver.HelmholtzOperators import create_helmholtz3d_matrix
from ...Solver.HelmholtzOperators import create_helmholtz2d_matrix_radial
from ...Utilities.Utils import make_velocity_from_trace, extend_vel_trace_1d, make_velocity3d_from_trace


if __name__ == "__main__":

    # Load Marmousi velocity trace
    with np.load("Lippmann-Schwinger/Data/marmousi-vp-vz.npz") as data:
        vel_trace = data["arr_0"]
    vel_trace /= 1000.0
    n1_vel_trace = vel_trace.shape[0]
    vel_trace = np.reshape(vel_trace, newshape=(n1_vel_trace, 1))

    # Lippmann-Schwinger solver
    n = 401
    d = 1.0 / (n - 1)
    precision = np.complex64
    f = 20.0
    sigma = 0.005

    def create_source():

        grid = np.linspace(start=-0.5, stop=0.5, num=n, endpoint=True)
        x1, y1, z1 = np.meshgrid(grid, grid, grid, indexing="ij")
        u = np.exp(-1.0 * (x1 ** 2 + y1 ** 2 + z1 ** 2) / (2 * sigma * sigma))
        u *= (1.0 / (sigma ** 3)) / ((2 * np.pi) ** 1.5)
        u = u.astype(precision)

        return u

    u = create_source()

    def helmholtz():

        pml_cells = 10
        n1 = n + 2 * pml_cells
        n2 = int(n / 2) + 1 + pml_cells

        vel_trace_mod = make_velocity_from_trace(vel_trace_=vel_trace, n1_=n, n2_=1)
        vel_trace_mod = extend_vel_trace_1d(vel_trace_=vel_trace_mod, pad_cells_=pml_cells)
        vel = make_velocity_from_trace(vel_trace_=vel_trace_mod, n1_=n + 2 * pml_cells, n2_=int(n / 2) + 1 + pml_cells)

        omega = 2 * np.pi * f
        a1 = d * (n1 - 1)
        a2 = d * (n2 - 1)

        mat = create_helmholtz2d_matrix_radial(
            a1=a1,
            a2=a2,
            pad1=pml_cells,
            pad2=pml_cells,
            omega=omega,
            precision=precision,
            vel=vel,
            pml_damping=50.0,
            adj=False,
            warnings=True
        )
        u1 = np.zeros(shape=(n1, n2), dtype=precision)
        u1[pml_cells:(n1 - pml_cells), 0:(n2 - pml_cells)] = u[:, int(n/2):, int(n/2)]

        plt.imshow(np.real(u1), cmap="Greys")
        plt.show()

        mat_lu = splu(mat)
        sol = mat_lu.solve(np.reshape(u1, newshape=(n1 * n2, 1)))
        sol2 = np.reshape(sol, newshape=(n1, n2))[pml_cells:(n1 - pml_cells), 0:(n2 - pml_cells)]

        return sol2

    sol = helmholtz()
    scale = 1e-1
    plt.imshow(np.real(sol), cmap="Greys", vmin=-scale, vmax=scale)
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

        omega = 2 * np.pi * f
        a1 = d * (n1 - 1)
        a2 = d * (n2 - 1)
        a3 = d * (n3 - 1)

        vel_trace_mod = make_velocity_from_trace(vel_trace_=vel_trace, n1_=n, n2_=1)
        vel_trace_mod = extend_vel_trace_1d(vel_trace_=vel_trace_mod, pad_cells_=pml_cells)
        vel_array_3d = make_velocity3d_from_trace(
            vel_trace_=vel_trace_mod,
            n1_=n + 2 * pml_cells,
            n2_=n + 2 * pml_cells,
            n3_=n + 2 * pml_cells
        )

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
        print("Matrix created")

        u1 = np.zeros(shape=(n1, n2, n3), dtype=precision)
        u1[pml_cells:pml_cells + n, pml_cells:pml_cells + n, pml_cells:pml_cells + n] += u

        # GMRES
        print("------------------------------------------------------------")
        print("Starting GMRES...")
        tol = 1e-5
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

        return sol[pml_cells:pml_cells + n, pml_cells:pml_cells + n, pml_cells:pml_cells + n]


    sol1 = helmholtz3d()
    sol1_crop = sol1[:, int(n / 2):, int(n/2)]

    plt.imshow(np.real(sol1_crop), cmap="Greys", vmin=-scale, vmax=scale)
    plt.show()

    print("Relative error = ", np.linalg.norm(sol1_crop - sol) / np.linalg.norm(sol1_crop))

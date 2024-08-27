import sys
import numpy as np
import time
import json
from scipy.sparse.linalg import lsqr, lsmr
from ..Solver.HelmholtzOperators import create_helmholtz2d_matrix


if __name__ == "__main__":

    # ----------------------------------------------
    # Check arguments and read in parameters
    # ----------------------------------------------
    if len(sys.argv) < 4:
        raise ValueError("Program missing command line arguments.")

    model_mode = int(sys.argv[1])
    freq_mode = int(sys.argv[2])
    solver_mode = int(sys.argv[3])

    if model_mode == 0:
        filepath1 = "Lippmann-Schwinger/Data/p04a-sigsbee-new-2d.npz"
        filepath2 = "Lippmann-Schwinger/Data/p06-sigsbee-source.npz"
        filepath3_ = "Lippmann-Schwinger/Data/p08a-sigsbee-"
    elif model_mode == 1:
        filepath1 = "Lippmann-Schwinger/Data/p04b-marmousi-new-2d.npz"
        filepath2 = "Lippmann-Schwinger/Data/p06-marmousi-source.npz"
        filepath3_ = "Lippmann-Schwinger/Data/p08a-marmousi-"
    elif model_mode == 2:
        filepath1 = "Lippmann-Schwinger/Data/p04c-seiscope-new-2d.npz"
        filepath2 = "Lippmann-Schwinger/Data/p06-seiscope-source.npz"
        filepath3_ = "Lippmann-Schwinger/Data/p08a-seiscope-"
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


    if solver_mode == 1:
        solver_name = "lsqr"
    elif solver_mode == 2:
        solver_name = "lsmr"
    else:
        raise ValueError("solver mode = ", solver_mode, " is not supported. Must be 1, or 2.")

    # ----------------------------------------------
    # Load vel
    # ----------------------------------------------
    with np.load(filepath1) as data:
        vel = data["arr_0"]

    # ----------------------------------------------
    # Set parameters
    # ----------------------------------------------
    n_ = 351
    nz_ = 251
    dx = 1.0 / (n_ - 1)
    dz = dx
    freq_ = freq * 5.25
    omega_ = 2 * np.pi * freq_
    precision_ = np.complex64

    # ----------------------------------------------
    # Initialize helmholtz matrix
    # ----------------------------------------------

    pml_cells = int((np.max(vel) / freq_) / dx)

    n_helmholtz_ = n_ + 2 * pml_cells
    nz_helmholtz_ = nz_ + 2 * pml_cells
    vel_helmholtz = np.zeros(shape=(nz_helmholtz_, n_helmholtz_), dtype=np.float32)
    vel_helmholtz[pml_cells: pml_cells + nz_, pml_cells: pml_cells + n_] += vel

    vel_helmholtz[:, 0:pml_cells] += np.reshape(vel_helmholtz[:, pml_cells], newshape=(nz_helmholtz_, 1))
    vel_helmholtz[:, pml_cells + n_:] += np.reshape(vel_helmholtz[:, pml_cells + n_ - 1], newshape=(nz_helmholtz_, 1))
    vel_helmholtz[0:pml_cells, :] += vel_helmholtz[pml_cells, :]
    vel_helmholtz[pml_cells + nz_:, :] += vel_helmholtz[pml_cells + nz_ - 1, :]

    mat = create_helmholtz2d_matrix(
        a1=dz * nz_helmholtz_,
        a2=dx * n_helmholtz_,
        pad1=pml_cells,
        pad2=pml_cells,
        omega=omega_,
        precision=precision_,
        vel=vel_helmholtz,
        pml_damping=50.0,
        adj=False,
        warnings=True
    )

    # ----------------------------------------------
    # Load source
    # ----------------------------------------------
    with np.load(filepath2) as data:
        sou_ = data["arr_0"]

    sou_ = sou_.astype(precision_)
    sou_helmholtz_ = np.zeros(shape=(nz_helmholtz_, n_helmholtz_), dtype=precision_)
    sou_helmholtz_[pml_cells: pml_cells + nz_, pml_cells: pml_cells + n_] += sou_

    rhs_norm = np.linalg.norm(sou_helmholtz_)
    rhs_ = sou_helmholtz_ / rhs_norm

    # ----------------------------------------------
    # Run solver iterations
    # ----------------------------------------------

    if solver_name == "lsqr":

        print("----------------------------------------------")
        print("Solver: LSQR \n")

        tol_ = 1e-5

        start_t = time.time()
        sol_, istop, itn_, r1norm = lsqr(
            mat,
            np.reshape(rhs_, newshape=(nz_helmholtz_ * n_helmholtz_, 1)),
            atol=0,
            btol=tol_,
            show=True,
            iter_lim=200000
        )[:4]
        sol_ = np.reshape(sol_, newshape=(nz_helmholtz_, n_helmholtz_))
        end_t = time.time()
        print("Total iterations: ", itn_)
        print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

        total_iter = itn_
        tsolve = end_t - start_t

    if solver_name == "lsmr":

        print("----------------------------------------------")
        print("Solver: LSMR \n")

        tol_ = 1e-5

        start_t = time.time()
        sol_, istop, itn_, r1norm = lsmr(
            mat,
            np.reshape(rhs_, newshape=(nz_helmholtz_ * n_helmholtz_, 1)),
            atol=0,
            btol=tol_,
            show=True,
            maxiter=200000
        )[:4]
        sol_ = np.reshape(sol_, newshape=(nz_helmholtz_, n_helmholtz_))
        end_t = time.time()
        print("Total iterations: ", itn_)
        print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

        total_iter = itn_
        tsolve = end_t - start_t

    sol_ = sol_ * rhs_norm
    sol_ = sol_[pml_cells: pml_cells + nz_, pml_cells: pml_cells + n_]

    # ----------------------------------------------
    # Save files
    # ----------------------------------------------

    np.savez(filepath3_ + "sol-" + solver_name + "-" + "{:4.2f}".format(freq) + ".npz", sol_)

    file_data = {}
    file_data["niter"] = total_iter
    file_data["tsolve"] = "{:4.2f}".format(tsolve)

    with open(filepath3_ + "stats-" + solver_name + "-" + "{:4.2f}".format(freq) + ".json", "w") as file:
        json.dump(file_data, file, indent=4)

import sys
import numpy as np
import time
import json
from scipy.sparse.linalg import LinearOperator, lsqr, lsmr
from ..Solver.ScatteringIntegralGeneralVz import TruncatedKernelGeneralVz2d
from ..Utilities.LinearSolvers import gmres_counter


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
        filepath = "Lippmann-Schwinger/Data/p04a-sigsbee-new-vz-2d.npz"
        filepath1 = "Lippmann-Schwinger/Data/p04a-sigsbee-new-2d.npz"
        filepath2 = "Lippmann-Schwinger/Data/p06-sigsbee-source.npz"
        filepath3_ = "Lippmann-Schwinger/Data/p07a-sigsbee-"
    elif model_mode == 1:
        filepath = "Lippmann-Schwinger/Data/p04b-marmousi-new-vz-2d.npz"
        filepath1 = "Lippmann-Schwinger/Data/p04b-marmousi-new-2d.npz"
        filepath2 = "Lippmann-Schwinger/Data/p06-marmousi-source.npz"
        filepath3_ = "Lippmann-Schwinger/Data/p07a-marmousi-"
    elif model_mode == 2:
        filepath = "Lippmann-Schwinger/Data/p04c-seiscope-new-vz-2d.npz"
        filepath1 = "Lippmann-Schwinger/Data/p04c-seiscope-new-2d.npz"
        filepath2 = "Lippmann-Schwinger/Data/p06-seiscope-source.npz"
        filepath3_ = "Lippmann-Schwinger/Data/p07a-seiscope-"
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
        raise ValueError("solver mode = ", solver_mode, " is not supported. Must be 1 or 2.")

    # ----------------------------------------------
    # Load vz and calculate psi
    # ----------------------------------------------
    with np.load(filepath) as data:
        vel = data["arr_0"]
    vel_trace = vel[:, 0]
    n1_vel_trace = vel_trace.shape[0]
    vel_trace = np.reshape(vel_trace, newshape=(n1_vel_trace, 1)).astype(np.float32)

    with np.load(filepath1) as data:
        vel1 = data["arr_0"]

    psi_ = (1.0 / vel) ** 2.0 - (1.0 / vel1) ** 2.0

    # ----------------------------------------------
    # Set parameters
    # ----------------------------------------------
    n_ = 351
    nz_ = 251
    a_ = 0.
    b_ = a_ + (1.0 / (n_ - 1)) * (nz_ - 1)
    freq_ = freq * 5.25
    omega_ = 2 * np.pi * freq_
    m_ = 4
    sigma_ = 0.0015
    precision_ = np.complex64
    green_func_dir_ = "Lippmann-Schwinger/Data/p05-green-func-" + str(model_mode) + "-" + str(freq_mode)
    num_threads_ = 4
    vz_ = np.zeros(shape=(nz_, 1), dtype=np.float32) + vel_trace

    psi_ = psi_.astype(precision_)

    # ----------------------------------------------
    # Load source
    # ----------------------------------------------
    with np.load(filepath2) as data:
        sou_ = data["arr_0"]

    sou_ = sou_.astype(precision_)

    # ----------------------------------------------
    # Initialize operator
    # ----------------------------------------------
    op = TruncatedKernelGeneralVz2d(
        n=n_,
        nz=nz_,
        a=a_,
        b=b_,
        k=omega_,
        vz=vz_,
        m=m_,
        sigma=sigma_,
        precision=precision_,
        green_func_dir=green_func_dir_,
        num_threads=num_threads_,
        verbose=False,
        light_mode=True
    )
    op.set_parameters(
        n=n_,
        nz=nz_,
        a=a_,
        b=b_,
        k=omega_,
        vz=vz_,
        m=m_,
        sigma=sigma_,
        precision=precision_,
        green_func_dir=green_func_dir_,
        num_threads=num_threads_,
        verbose=False
    )

    rhs_ = np.zeros(shape=(nz_, n_), dtype=precision_)
    t1 = time.time()
    op.apply_kernel(u=sou_, output=rhs_)
    t2 = time.time()
    print("Operator application time = ", "{:6.2f}".format(t2 - t1), " s")

    np.savez(filepath3_ + "rhs-" + "{:4.2f}".format(freq) + ".npz", rhs_)

    rhs_norm = np.linalg.norm(rhs_)
    rhs_ = rhs_ / rhs_norm

    # ----------------------------------------------
    # Initialize linear operator objects
    # ----------------------------------------------
    def func_matvec(v):
        v = np.reshape(v, newshape=(nz_, n_))
        u = v * 0
        op.apply_kernel(u=v * psi_, output=u, adj=False, add=False)
        return np.reshape(v - (omega_ ** 2) * u, newshape=(nz_ * n_, 1))


    def func_matvec_adj(v):
        v = np.reshape(v, newshape=(nz_, n_))
        u = v * 0
        op.apply_kernel(u=v, output=u, adj=True, add=False)
        return np.reshape(v - (omega_ ** 2) * u * psi_, newshape=(nz_ * n_, 1))


    linop_lse = LinearOperator(
        shape=(nz_ * n_, nz_ * n_),
        matvec=func_matvec,
        rmatvec=func_matvec_adj,
        dtype=precision_
    )

    # ----------------------------------------------
    # Run solver iterations
    # ----------------------------------------------

    if solver_name == "lsqr":

        print("----------------------------------------------")
        print("Solver: LSQR \n")

        tol_ = 1e-5

        start_t = time.time()
        sol_, istop, itn_, r1norm = lsqr(
            linop_lse,
            np.reshape(rhs_, newshape=(nz_ * n_, 1)),
            atol=0,
            btol=tol_,
            show=True,
            iter_lim=50000
        )[:4]
        sol_ = np.reshape(sol_, newshape=(nz_, n_))
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
            linop_lse,
            np.reshape(rhs_, newshape=(nz_ * n_, 1)),
            atol=0,
            btol=tol_,
            show=True,
            maxiter=50000
        )[:4]
        sol_ = np.reshape(sol_, newshape=(nz_, n_))
        end_t = time.time()
        print("Total iterations: ", itn_)
        print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

        total_iter = itn_
        tsolve = end_t - start_t

    sol_ = sol_ * rhs_norm

    # ----------------------------------------------
    # Save files
    # ----------------------------------------------

    np.savez(filepath3_ + "sol-" + solver_name + "-" + "{:4.2f}".format(freq) + ".npz", sol_)

    file_data = {}
    file_data["niter"] = total_iter
    file_data["tsolve"] = "{:4.2f}".format(tsolve)

    with open(filepath3_ + "stats-" + solver_name + "-" + "{:4.2f}".format(freq) + ".json", "w") as file:
        json.dump(file_data, file, indent=4)

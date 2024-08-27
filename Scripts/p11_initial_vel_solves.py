import sys
import numpy as np
import time
import json
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, gmres, lsqr, lsmr
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
        filepath3_ = "Lippmann-Schwinger/Data/p07-sigsbee-"
        filepath4_ = "Lippmann-Schwinger/Data/p11-sigsbee-"
    elif model_mode == 1:
        filepath = "Lippmann-Schwinger/Data/p04b-marmousi-new-vz-2d.npz"
        filepath1 = "Lippmann-Schwinger/Data/p04b-marmousi-new-2d.npz"
        filepath2 = "Lippmann-Schwinger/Data/p06-marmousi-source.npz"
        filepath3_ = "Lippmann-Schwinger/Data/p07-marmousi-"
        filepath4_ = "Lippmann-Schwinger/Data/p11-marmousi-"
    elif model_mode == 2:
        filepath = "Lippmann-Schwinger/Data/p04c-seiscope-new-vz-2d.npz"
        filepath1 = "Lippmann-Schwinger/Data/p04c-seiscope-new-2d.npz"
        filepath2 = "Lippmann-Schwinger/Data/p06-seiscope-source.npz"
        filepath3_ = "Lippmann-Schwinger/Data/p07-seiscope-"
        filepath4_ = "Lippmann-Schwinger/Data/p11-seiscope-"
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


    if solver_mode == 0:
        solver_name = "gmres"
    elif solver_mode == 1:
        solver_name = "lsqr"
    elif solver_mode == 2:
        solver_name = "lsmr"
    else:
        raise ValueError("freq mode = ", freq_mode, " is not supported. Must be 0, 1, 2, or 3.")

    # ----------------------------------------------
    # Load vz and calculate psi
    # Load initial solution
    # ----------------------------------------------
    with np.load(filepath) as data:
        vel = data["arr_0"]
    vel_trace = vel[:, 0]
    n1_vel_trace = vel_trace.shape[0]
    vel_trace = np.reshape(vel_trace, newshape=(n1_vel_trace, 1)).astype(np.float32)

    with np.load(filepath1) as data:
        vel1 = data["arr_0"]

    vel_diff = vel1 - vel

    with np.load(filepath3_ + "sol-" + "gmres" + "-" + "{:4.2f}".format(freq) + ".npz") as data:
        intitial_sol = data["arr_0"]

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
    green_func_dir_ = filepath = "Lippmann-Schwinger/Data/p05-green-func-" + str(model_mode) + "-" + str(freq_mode)
    num_threads_ = 4
    vz_ = np.zeros(shape=(nz_, 1), dtype=np.float32) + vel_trace

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

    # ----------------------------------------------
    # Run tests
    # ----------------------------------------------
    # scalars = [1.01, 1.03, 1.05]
    scalars = [1.01]
    file_data = {}

    for fac in scalars:

        file_data[str(fac)] = {}

        vel_diff1 = vel_diff * fac
        psi1 = (1.0 / vel) ** 2.0 - (1.0 / (vel + vel_diff1)) ** 2.0
        psi1 = psi1.astype(precision_)

        rhs_norm = np.linalg.norm(rhs_)
        rhs1 = rhs_ / rhs_norm

        # ----------------------------------------------
        # Initialize linear operator objects
        # ----------------------------------------------
        def func_matvec(v):
            v = np.reshape(v, newshape=(nz_, n_))
            u = v * 0
            op.apply_kernel(u=v * psi1, output=u, adj=False, add=False)
            return np.reshape(v - (omega_ ** 2) * u, newshape=(nz_ * n_, 1))


        def func_matvec_adj(v):
            v = np.reshape(v, newshape=(nz_, n_))
            u = v * 0
            op.apply_kernel(u=v, output=u, adj=True, add=False)
            return np.reshape(v - (omega_ ** 2) * u * psi1, newshape=(nz_ * n_, 1))


        linop_lse = LinearOperator(
            shape=(nz_ * n_, nz_ * n_),
            matvec=func_matvec,
            rmatvec=func_matvec_adj,
            dtype=precision_
        )

        # ----------------------------------------------
        # Run solver iterations
        # ----------------------------------------------

        if solver_name == "gmres":

            print("----------------------------------------------")
            print("Solver: GMRES \n")

            tol_ = 1e-5
            counter = gmres_counter()

            start_t = time.time()
            sol_, exitcode = gmres(
                linop_lse,
                np.reshape(rhs1, newshape=(nz_ * n_, 1)),
                maxiter=5000,
                restart=5000,
                atol=0,
                tol=tol_,
                callback=counter
            )
            sol_ = np.reshape(sol_, newshape=(nz_, n_))
            end_t = time.time()
            print("Exitcode= ", exitcode)
            print("Total iterations= ", counter.niter)
            print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

            total_iter = counter.niter
            tsolve = end_t - start_t

        if solver_name == "lsqr":

            print("----------------------------------------------")
            print("Solver: LSQR \n")

            tol_ = 1e-5

            start_t = time.time()
            sol_, istop, itn_, r1norm = lsqr(
                linop_lse,
                np.reshape(rhs1, newshape=(nz_ * n_, 1)),
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
                np.reshape(rhs1, newshape=(nz_ * n_, 1)),
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

        file_data[str(fac)]["niter"] = total_iter
        file_data[str(fac)]["tsolve"] = "{:4.2f}".format(tsolve)

        # ----------------------------------------------
        # Run solver iterations with initial solution
        # ----------------------------------------------

        rhs1 = rhs_ - np.reshape(func_matvec(np.reshape(intitial_sol, newshape=(nz_ * n_, 1))), newshape=(nz_, n_))
        rhs1_norm = np.linalg.norm(rhs1)
        rhs1 = rhs1 / rhs1_norm

        if solver_name == "gmres":
            print("----------------------------------------------")
            print("Solver: GMRES \n")

            tol_ = 1e-5 * rhs_norm / rhs1_norm
            counter = gmres_counter()

            start_t = time.time()
            sol_, exitcode = gmres(
                linop_lse,
                np.reshape(rhs1, newshape=(nz_ * n_, 1)),
                maxiter=50000,
                restart=5000,
                atol=0,
                tol=tol_,
                callback=counter
            )
            sol_ = np.reshape(sol_, newshape=(nz_, n_))
            end_t = time.time()
            print("Exitcode= ", exitcode)
            print("Total iterations= ", counter.niter)
            print("Total time to solve: ", "{:4.2f}".format(end_t - start_t), " s \n")

            total_iter = counter.niter
            tsolve = end_t - start_t

        if solver_name == "lsqr":
            print("----------------------------------------------")
            print("Solver: LSQR \n")

            tol_ = 1e-5 * rhs_norm / rhs1_norm

            start_t = time.time()
            sol_, istop, itn_, r1norm = lsqr(
                linop_lse,
                np.reshape(rhs1, newshape=(nz_ * n_, 1)),
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

            tol_ = 1e-5 * rhs_norm / rhs1_norm

            start_t = time.time()
            sol_, istop, itn_, r1norm = lsmr(
                linop_lse,
                np.reshape(rhs1, newshape=(nz_ * n_, 1)),
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

        sol_ = sol_ * rhs1_norm + intitial_sol
        file_data[str(fac)]["niter_u0"] = total_iter
        file_data[str(fac)]["tsolve_u0"] = "{:4.2f}".format(tsolve)

    # ----------------------------------------------
    # Save files
    # ----------------------------------------------
    with open(filepath4_ + "stats-" + solver_name + "-" + "{:4.2f}".format(freq) + ".json", "w") as file:
        json.dump(file_data, file, indent=4)

import sys
import numpy as np
from ..Solver.ScatteringIntegralGeneralVz import TruncatedKernelGeneralVz2d


if __name__ == "__main__":

    # ----------------------------------------------
    # Check arguments and read in parameters
    # ----------------------------------------------
    if len(sys.argv) < 3:
        raise ValueError("Program missing command line arguments.")

    model_mode = int(sys.argv[1])
    freq_mode = int(sys.argv[2])

    if model_mode == 0:
        filepath = "Lippmann-Schwinger/Data/p04a-sigsbee-new-vz-2d.npz"
    elif model_mode == 1:
        filepath = "Lippmann-Schwinger/Data/p04b-marmousi-new-vz-2d.npz"
    elif model_mode == 2:
        filepath = "Lippmann-Schwinger/Data/p04c-seiscope-new-vz-2d.npz"
    else:
        print("model mode = ", model_mode, " is not supported. Must be 0, 1, or 2.")

    if freq_mode == 0:
        freq = 5.0
    elif freq_mode == 1:
        freq = 7.5
    elif freq_mode == 2:
        freq = 10.0
    elif freq_mode == 3:
        freq = 15.0
    else:
        print("freq mode = ", freq_mode, " is not supported. Must be 0, 1, 2, or 3.")

    # ----------------------------------------------
    # Load vz
    # ----------------------------------------------
    with np.load(filepath) as data:
        vel = data["arr_0"]
    vel_trace = vel[:, 0]
    n1_vel_trace = vel_trace.shape[0]
    vel_trace = np.reshape(vel_trace, newshape=(n1_vel_trace, 1)).astype(np.float32)

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
    print("Green's function directory: ", green_func_dir_)

    # ----------------------------------------------
    # Calculate Green's function and save to disk
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
        light_mode=False
    )

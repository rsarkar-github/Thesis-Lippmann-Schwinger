import time
import numpy as np
import matplotlib.pyplot as plt
from ...Solver.ScatteringIntegralGeneralVz import TruncatedKernelGeneralVz2d


if __name__ == "__main__":

    # ----------------------------------------------
    # 2d Test

    n_ = 351
    nz_ = 251
    a_ = 0.
    b_ = a_ + (1.0 / (n_ - 1)) * (nz_ - 1)
    freq_ = 3.0 * 5.25
    omega_ = 2 * np.pi * freq_
    m_ = 4
    sigma_ = 0.004
    precision_ = np.complex64
    green_func_dir_ = "Lippmann-Schwinger/Test/Data/t11/"
    num_threads_ = 4
    vz_ = np.zeros(shape=(nz_, 1), dtype=np.float32) + 3.0

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

    u_ = np.zeros(shape=(nz_, n_), dtype=precision_)
    u_[int(nz_ / 2), int(n_ / 2)] = 1.0
    output_ = u_ * 0

    t1 = time.time()
    op.apply_kernel(u=u_, output=output_)
    t2 = time.time()
    print("Operator application time = ", "{:6.2f}".format(t2 - t1), " s")

    plt.imshow(np.imag(output_), cmap="Greys")
    plt.show()

import numpy as np
import time
from ...Solver.ScatteringIntegralGeneralVz import TruncatedKernelGeneralVz3d



if __name__ == "__main__":

    sigma = 0.03
    n = 251
    d = 1.0 / (n - 1)
    precision = np.complex64
    f = 10.0
    v0 = 1.0
    omega = 2 * np.pi * f

    # Compute Lippmann-Schwinger solver
    a = 0.
    b = a + (1.0 / (n - 1)) * (n - 1)
    m = 5
    vz = np.zeros(shape=(n, 1), dtype=np.float32) + v0

    op = TruncatedKernelGeneralVz3d(
        n=n,
        nz=n,
        a=a,
        b=b,
        k=omega,
        vz=vz,
        m=m,
        sigma=3 * d / m,
        precision=precision,
        green_func_dir="Lippmann-Schwinger/Test/Data/t13",
        num_threads=8,
        verbose=False,
        light_mode=True
    )

    t1 = time.time()
    op.set_parameters(
        n=n,
        nz=n,
        a=a,
        b=b,
        k=omega,
        vz=vz,
        m=m,
        sigma=3 * d / m,
        precision=precision,
        green_func_dir="Lippmann-Schwinger/Test/Data/t13",
        num_threads=100,
        verbose=False
    )
    t2 = time.time()

    print("Time to read Green's function = ", "{:4.2f}".format(t2-t1), " s.")

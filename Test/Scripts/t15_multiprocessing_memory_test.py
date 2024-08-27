import time
import numpy as np
from numpy import ndarray
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from ...Solver.ScatteringIntegralGeneralVz import TruncatedKernelGeneralVz2d


def func(params):

    op = params[0]
    rhs = params[1]
    sou = params[2]
    green_func_shape = params[3]
    precision = params[4]
    sm_name = params[5]
    job_id = params[6]

    # Attach to shared memory
    sm = SharedMemory(sm_name)
    greens_func = ndarray(shape=green_func_shape, dtype=precision, buffer=sm.buf)
    op.greens_func = greens_func

    # Perform convolution
    t1 = time.time()
    op.apply_kernel(u=sou, output=rhs)
    t2 = time.time()
    print(
        "Job id = ", str(job_id),
        ", Operator application time = ", "{:6.2f}".format(t2 - t1), " s",
        ", Norm of result = ", np.linalg.norm(rhs)
    )

    # Close shared memory
    sm.close()


if __name__ == "__main__":

    # Load files
    filepath = "Lippmann-Schwinger/Data/p04c-seiscope-new-vz-2d.npz"
    filepath1 = "Lippmann-Schwinger/Data/p04c-seiscope-new-2d.npz"
    filepath2 = "Lippmann-Schwinger/Data/p06-seiscope-source.npz"

    # Freq and solver
    freq = 5.0   # in Hz
    solver_name = "gmres"

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
    green_func_dir_ = filepath = "Lippmann-Schwinger/Data/p05-green-func-2-0"
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
    # Get Green's function
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

    green_func = op.greens_func
    num_bytes = op.green_func_bytes
    green_func_shape = op.green_func_shape

    # ----------------------------------------------
    # Setup multiprocessing workflow
    # ----------------------------------------------

    num_jobs = 500
    rhs_ = np.zeros(shape=(nz_, n_), dtype=precision_)

    # Read in multiprocessing mode
    with SharedMemoryManager() as smm:

        # Create shared memory
        sm = smm.SharedMemory(size=num_bytes)
        green_func_shared = ndarray(
            shape=op.greens_func.shape,
            dtype=precision_,
            buffer=sm.buf
        )
        green_func_shared *= 0
        green_func_shared += green_func

        op1 = TruncatedKernelGeneralVz2d(
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
        op1.set_parameters(
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
            green_func_set=False,
            num_threads=num_threads_,
            verbose=False
        )

        param_tuple_list = [
            (
                op1,
                rhs_,
                sou_,
                green_func_shape,
                precision_,
                sm.name,
                ii
            ) for ii in range(num_jobs)
        ]

        print("\nRunning multiprocessing jobs...")
        print("Num CPUs = ", mp.cpu_count())

        with Pool(min(len(param_tuple_list), mp.cpu_count(), 100)) as pool:
            max_ = len(param_tuple_list)

            with tqdm(total=max_) as pbar:
                for _ in pool.imap_unordered(func, param_tuple_list):
                    pbar.update()

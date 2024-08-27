import os
import sys
import time
import numpy as np
from numpy import ndarray
import scipy.fft as scfft
from scipy import interpolate
from scipy.sparse.linalg import splu
import numba
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from ..Solver.HelmholtzOperators import create_helmholtz2d_matrix_radial
from ..Solver.HelmholtzOperators import create_helmholtz2d_matrix_even
from ..Utilities import TypeChecker, FourierTools
import matplotlib.pyplot as plt


def func_helper2D(params):
    """
    Helper function to solve Helmholtz equation in 3D.
    Compute Fourier transform of truncated kernel.

    :param params:
        params[0]: (2D numpy.ndarray)
            Solved Green's functions for the depth slice
        params[1]: (np.complex64 or np.complex128)
            precision
        params[2]: (int)
            Index for starting extraction in z direction
        params[3]: (int)
            Index for ending extraction in z direction
        params[4]: (int)
            self._m (decimation factor)
        params[5]: (np.ndarray)
            self._cutoff_func_coarse_grid (cutoff function)
        params[6]: (float)
            fine grid spacing along x
        params[7]: (int)
            depth slice number
        params[8]: (str)
            shared memory object name holding the Green's function
        params[9]: (bool)
            verbose flag
    """

    t_start = time.time()

    # ------------------------------------------------------------
    # Read in parameters

    green_func = params[0]
    precision = params[1]
    start_index_z = params[2]
    end_index_z = params[3]
    m = params[4]
    cutoff_func = params[5]
    dx = params[6]
    num_slice = params[7]
    sm_name = params[8]
    verbose = params[9]

    # ------------------------------------------------------------
    # Extract solution

    t1 = time.time()

    green_func_slice = green_func[start_index_z:end_index_z + 1:m, ::m]

    nz_new = green_func_slice.shape[0]
    nx_new = cutoff_func.shape[0]

    green_func_slice_truncated = np.zeros(shape=(nz_new, nx_new), dtype=precision)
    green_func_slice_truncated[:, 0:green_func_slice.shape[1]] = green_func_slice

    t2 = time.time()
    if verbose:
        print(
            "Thread num: ", num_slice, ", Extraction done  in",
            "{:6.2f}".format(t2 - t1), " s"
        )

    # ------------------------------------------------------------
    # Calculate Fourier transform of truncated kernel

    t1 = time.time()

    green_func_slice_truncated_ft = FourierTools.hfft_truncated_gfunc_even_2d(
        greens_func=green_func_slice_truncated,
        smooth_cutoff=cutoff_func,
        dx=dx * m
    )

    t2 = time.time()
    if verbose:
        print(
            "Thread num: ", num_slice, ", Fourier transform of truncated kernel done in",
            "{:6.2f}".format(t2 - t1), " s"
        )

    # ------------------------------------------------------------
    # Copy result to shared memory

    sm = SharedMemory(sm_name)
    data = ndarray(shape=(nz_new, nz_new, nx_new), dtype=precision, buffer=sm.buf)
    data[num_slice, :, :] += green_func_slice_truncated_ft
    sm.close()

    t_end = time.time()
    if verbose:
        print("Thread num", num_slice, "finished in", "{:6.2f}".format(t_end - t_start), "s")


def func_helper2D_no_mpi(
    green_func,
    precision,
    start_index_z,
    end_index_z,
    m,
    cutoff_func,
    dx,
    num_slice,
    green_func_truncated_ft,
    verbose
):
    """
    Helper function to solve Helmholtz equation in 3D.
    Compute Fourier transform of truncated kernel.

    :param green_func:(2D numpy.ndarray)
        Solved Green's functions for the depth slice
    :param precision: (np.complex64 or np.complex128)
        precision
    :param start_index_z: (int)
        Index for starting extraction in z direction
    :param end_index_z: (int)
        Index for ending extraction in z direction
    :params m: (int)
        self._m (decimation factor)
    :param cutoff_func: (np.ndarray)
        self._cutoff_func_coarse_grid (cutoff function)
    :param dx: (float)
        fine grid spacing along x
    :param num_slice: (int)
        depth slice number
    :param green_func_truncated_ft: (str)
        Numpy ndarray containing the truncated Green's function
    :param verbose: (bool)
        verbose flag
    """

    t_start = time.time()

    # ------------------------------------------------------------
    # Extract solution

    t1 = time.time()

    green_func_slice = green_func[start_index_z:end_index_z + 1:m, ::m]

    nz_new = green_func_slice.shape[0]
    nx_new = cutoff_func.shape[0]

    green_func_slice_truncated = np.zeros(shape=(nz_new, nx_new), dtype=precision)
    green_func_slice_truncated[:, 0:green_func_slice.shape[1]] = green_func_slice

    t2 = time.time()
    if verbose:
        print(
            "Thread num: ", num_slice, ", Extraction done  in",
            "{:6.2f}".format(t2 - t1), " s"
        )

    # ------------------------------------------------------------
    # Calculate Fourier transform of truncated kernel

    t1 = time.time()

    green_func_slice_truncated_ft = FourierTools.hfft_truncated_gfunc_even_2d(
        greens_func=green_func_slice_truncated,
        smooth_cutoff=cutoff_func,
        dx=dx * m
    )

    t2 = time.time()
    if verbose:
        print(
            "Thread num: ", num_slice, ", Fourier transform of truncated kernel done in",
            "{:6.2f}".format(t2 - t1), " s"
        )

    # ------------------------------------------------------------
    # Copy result to shared memory
    green_func_truncated_ft[num_slice, :, :] += green_func_slice_truncated_ft

    t_end = time.time()
    if verbose:
        print("Thread num", num_slice, "finished in", "{:6.2f}".format(t_end - t_start), "s")


def func_helper3D(params):
    """
    Helper function to solve Helmholtz equation in 3D.
    Compute Fourier transform of truncated kernel.

    :param params:
        params[0]: (2D numpy.ndarray)
            Solved Green's functions for the depth slice
        params[1]: (np.complex64 or np.complex128)
            precision
        params[2]: (int)
            Index for starting extraction in z direction
        params[3]: (int)
            Index for ending extraction in z direction
        params[4]: (int)
            self._m (decimation factor)
        params[5]: (int)
            self._num_bins_non_neg
        params[6]: (np.ndarray)
            self._cutoff_func_fine_grid (cutoff function on fine grid)
        params[7]: (float)
            fine grid spacing along x
        params[8]: (int)
            depth slice number
        params[9]: (str)
            shared memory object name holding the Green's function
        params[10]: (bool)
            verbose flag
    """

    t_start = time.time()

    # ------------------------------------------------------------
    # Read in parameters

    green_func = params[0]
    precision = params[1]
    start_index_z = params[2]
    end_index_z = params[3]
    m = params[4]
    num_bins_non_neg = params[5]
    cutoff_func_fine = params[6]
    dx = params[7]
    num_slice = params[8]
    sm_name = params[9]
    verbose = params[10]

    # ------------------------------------------------------------
    # Extract solution

    t1 = time.time()

    green_func_slice = green_func[start_index_z:end_index_z + 1:m, :]

    nz_new = green_func_slice.shape[0]
    nx_new = cutoff_func_fine.shape[0]

    green_func_slice_truncated = np.zeros(shape=(nz_new, nx_new), dtype=precision)
    green_func_slice_truncated[:, 0:green_func_slice.shape[1]] = green_func_slice

    t2 = time.time()
    if verbose:
        print(
            "Thread num: ", num_slice, ", Extraction done  in",
            "{:6.2f}".format(t2 - t1), " s"
        )

    # ------------------------------------------------------------
    # Calculate Fourier transform of truncated kernel

    t1 = time.time()

    green_func_slice_truncated_ft = FourierTools.hfft_truncated_gfunc_radial_3d(
        greens_func=green_func_slice_truncated,
        smooth_cutoff=cutoff_func_fine,
        dx=dx,
        fac=m
    )

    t2 = time.time()
    if verbose:
        print(
            "Thread num: ", num_slice, ", Fourier transform of truncated kernel done in",
            "{:6.2f}".format(t2 - t1), " s"
        )

    # ------------------------------------------------------------
    # Copy result to shared memory

    sm = SharedMemory(sm_name)
    data = ndarray(shape=(nz_new, nz_new, num_bins_non_neg, num_bins_non_neg), dtype=precision, buffer=sm.buf)
    data[num_slice, :, :, :] += green_func_slice_truncated_ft
    sm.close()

    t_end = time.time()
    if verbose:
        print("Thread num", num_slice, "finished in", "{:6.2f}".format(t_end - t_start), "s")


def func_write3D(params):
    """
    Helper function to write Green's function to disk in 3D.
    Green's func has shape (nz, nz, 2n - 1, 2n -1)

    :param params: (str, int, int, int, str)
        params[0]: Green's func directory
        params[1]: Slice number in nz
        params[2]: nz
        params[3]: n
        params[4]: precision
        params[5]: shared memory object name holding the Green's function
    """

    green_func_dir = params[0]
    nz_slice = params[1]
    nz = params[2]
    n = params[3]
    precision = params[4]
    sm_name = params[5]

    # Attach to shared memory
    sm = SharedMemory(sm_name)
    data = ndarray(shape=(nz, nz, 2 * n - 1, 2 * n - 1), dtype=precision, buffer=sm.buf)

    # Write slice to disk using default numpy compression
    file_name = os.path.join(green_func_dir, "green_func_slice_" + str(nz_slice) + ".npz")
    np.savez_compressed(file_name, data[nz_slice, :, :, :])

    # Close shared memory
    sm.close()


def func_read3D(params):
    """
    Helper function to read Green's function from disk in 3D.
    Green's func has shape (nz, nz, 2n - 1, 2n -1)

    :param params: (str, int, int, int, str)
        params[0]: Green's func directory
        params[1]: Slice number in nz
        params[2]: nz
        params[3]: n
        params[4]: precision
        params[5]: shared memory object name holding the Green's function
    """

    green_func_dir = params[0]
    nz_slice = params[1]
    nz = params[2]
    n = params[3]
    precision = params[4]
    sm_name = params[5]

    # Attach to shared memory
    sm = SharedMemory(sm_name)
    data = ndarray(shape=(nz, nz, 2 * n - 1, 2 * n - 1), dtype=precision, buffer=sm.buf)

    # Read slice from disk
    file_name = os.path.join(green_func_dir, "green_func_slice_" + str(nz_slice) + ".npz")
    with np.load(file_name) as f:
        data_slice = f["arr_0"]
    data[nz_slice, :, :, :] += data_slice

    # Close shared memory
    sm.close()


class TruncatedKernelGeneralVz3d:

    def __init__(self, n, nz, a, b, k, vz, m, sigma, precision, green_func_dir, num_threads, no_mpi=False, verbose=False, light_mode=False):
        """
        The Helmholtz equation reads (lap + k^2 / vz^2)u = f, on the domain [a,b] x [-0.5, 0.5]^2.

        :param n: The field will have shape n x n x nz, with n odd. This means that the cube
        [-0.5, 0.5]^2 is gridded into n^2 points. The points have coordinates {-0.5 + k/(n-1) : 0 <= k <= n-1}.
        :param nz: The field will have shape n x n x nz, with n odd. This means that the interval [a, b] is gridded
        into nz points. The points have coordinates {a + (b-a)k/(nz-1) : 0 <= k <= nz-1}.
        :param a: The domain in z direction is [a, b] with 0 < a < b.
        :param b: The domain in z direction is [a, b] with 0 < a < b.
        :param k: The frequency k.
        :param vz: The 1D vz velocity profile. Must be ndarray of shape (nz, 1).
        :param m: Decimation factor for calculating Green's function on a fine grid.
        :param sigma: The standard deviation of delta to inject for Green's function calculation.
        :param precision: np.complex64 or np.complex128
        :param green_func_dir: Name of directory where to read / write Green's function from.
        :param num_threads: Maximum number of threads to use
        :param no_mpi: bool (if False do not use multiprocessing)
        :param verbose: bool (if True print messages during Green's function calculation).
        :param light_mode: bool (if True an empty class is initialized)
        """

        TypeChecker.check(x=light_mode, expected_type=(bool,))
        self._initialized_flag = not light_mode
        self._green_func_set_flag = not light_mode

        if not light_mode:

            print("\n\nInitializing the class")

            TypeChecker.check(x=n, expected_type=(int,))
            if n % 2 != 1 or n < 3:
                raise ValueError("n must be an odd integer >= 3")

            TypeChecker.check(x=nz, expected_type=(int,))
            if nz < 2:
                raise ValueError("n must be an integer >= 2")

            TypeChecker.check(x=a, expected_type=(float,))
            TypeChecker.check_float_strict_lower_bound(x=b, lb=a)
            TypeChecker.check_float_positive(k)

            # Minimum velocity of 0.1 km/s
            TypeChecker.check_ndarray(x=vz, shape=(nz, 1), dtypes=(np.float32,), lb=0.1)
            TypeChecker.check_int_positive(m)

            TypeChecker.check_float_positive(x=sigma)

            if precision not in [np.complex64, np.complex128]:
                raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

            TypeChecker.check(x=green_func_dir, expected_type=(str,))

            if not os.path.exists(green_func_dir):
                os.makedirs(green_func_dir)

            TypeChecker.check_int_positive(x=num_threads)
            TypeChecker.check(x=no_mpi, expected_type=(bool,))
            TypeChecker.check(x=verbose, expected_type=(bool,))

            self._n = n
            self._nz = nz
            self._k = k
            self._a = a
            self._b = b
            self._vz = vz
            self._m = m
            self._sigma = sigma
            self._precision = precision
            self._green_func_dir = green_func_dir
            self._num_threads = num_threads
            self._no_mpi = no_mpi
            self._verbose = verbose

            self._cutoff1 = np.sqrt(2.0)
            self._cutoff2 = 1.5

            # Run class initializer
            self.__initialize_class()

    def set_parameters(self, n, nz, a, b, k, vz, m, sigma, precision, green_func_dir, num_threads, green_func_set=True, no_mpi=False, verbose=False):
        """
        The Helmholtz equation reads (lap + k^2 / vz^2)u = f, on the domain [a,b] x [-0.5, 0.5]^2.

        :param n: The field will have shape n x n x nz, with n odd. This means that the cube
        [-0.5, 0.5]^2 is gridded into n^2 points. The points have coordinates {-0.5 + k/(n-1) : 0 <= k <= n-1}.
        :param nz: The field will have shape n x n x nz, with n odd. This means that the interval [a, b] is gridded
        into nz points. The points have coordinates {a + (b-a)k/(nz-1) : 0 <= k <= nz-1}.
        :param a: The domain in z direction is [a, b] with 0 < a < b.
        :param b: The domain in z direction is [a, b] with 0 < a < b.
        :param k: The frequency k.
        :param vz: The 1D vz velocity profile. Must be ndarray of shape (nz, 1).
        :param m: Decimation factor for calculating Green's function on a fine grid.
        :param sigma: The standard deviation of delta to inject for Green's function calculation.
        :param precision: np.complex64 or np.complex128
        :param green_func_dir: Name of directory where to read / write Green's function from.
        :param num_threads: Maximum number of threads to use
        :param green_func_set: bool (if False do not set Green's function)
        :param no_mpi: bool (if False do not use multiprocessing)
        :param verbose: bool (if True print messages during Green's function calculation).
        """

        print("\n\nInitializing the class")

        TypeChecker.check(x=n, expected_type=(int,))
        if n % 2 != 1 or n < 3:
            raise ValueError("n must be an odd integer >= 3")

        TypeChecker.check(x=nz, expected_type=(int,))
        if nz < 2:
            raise ValueError("n must be an integer >= 2")

        TypeChecker.check(x=a, expected_type=(float,))
        TypeChecker.check_float_strict_lower_bound(x=b, lb=a)
        TypeChecker.check_float_positive(k)

        # Minimum velocity of 0.1 km/s
        TypeChecker.check_ndarray(x=vz, shape=(nz, 1), dtypes=(np.float32,), lb=0.1)
        TypeChecker.check_int_positive(m)

        TypeChecker.check_float_positive(x=sigma)

        if precision not in [np.complex64, np.complex128]:
            raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

        TypeChecker.check(x=green_func_dir, expected_type=(str,))
        if not os.path.exists(green_func_dir):
            print("\n")
            print("Green's function directory does not exist or is empty.")
            print("Class set_parameters method failed. Exiting without changes to class.")
            print("\n")
            return

        for ii in range(nz):
            file_name = os.path.join(green_func_dir, "green_func_slice_" + str(ii) + ".npz")
            if not os.path.exists(file_name):
                print("\n")
                print("Green's function directory does not have all the needed files.")
                print("Class set_parameters method failed. Exiting without changes to class.")
                print("\n")

        TypeChecker.check_int_positive(x=num_threads)
        TypeChecker.check(x=green_func_set, expected_type=(bool,))
        TypeChecker.check(x=no_mpi, expected_type=(bool,))
        TypeChecker.check(x=verbose, expected_type=(bool,))

        self._n = n
        self._nz = nz
        self._k = k
        self._a = a
        self._b = b
        self._vz = vz
        self._m = m
        self._sigma = sigma
        self._precision = precision
        self._green_func_dir = green_func_dir
        self._num_threads = num_threads
        self._no_mpi = no_mpi
        self._verbose = verbose

        self._cutoff1 = np.sqrt(2.0)
        self._cutoff2 = 1.5

        self.__initialize_class(green_func_flag=False)
        if green_func_set:
            self.__read_green_func()
        self._initialized_flag = True
        self._green_func_set_flag = green_func_set

    def apply_kernel(self, u, output, adj=False, add=False):
        """
        :param u: 3d numpy array (must be nz x n x n dimensions with n odd).
        :param output: 3d numpy array (same dimension as u).
        :param adj: Boolean flag (forward or adjoint operator)
        :param add: Boolean flag (whether to add result to output)
        """

        if not self._green_func_set_flag:
            raise ValueError("Class initialized in light mode or Green's function not set, cannot perform operation.")

        # Check types of input
        if u.dtype != self._precision or output.dtype != self._precision:
            raise TypeError("Types of 'u' and 'output' must match that of class: ", self._precision)

        # Check dimensions
        if u.shape != (self._nz, self._n, self._n) or output.shape != (self._nz, self._n, self._n):
            raise ValueError("Shapes of 'u' and 'output' must be (nz, n, n) with nz = ", self._nz, " and n = ", self._n)

        # Count of non-negative wave numbers
        count = self._num_bins_non_neg - 1

        # Copy u into temparray and compute Fourier transform along first axis
        temparray = np.zeros(shape=(self._nz, self._num_bins, self._num_bins), dtype=self._precision)
        temparray[:, self._start_index:(self._end_index + 1), self._start_index:(self._end_index + 1)] = u
        temparray = scfft.fftn(scfft.fftshift(temparray, axes=(1, 2)), axes=(1, 2))
        if not adj:
            temparray *= self._mu
        else:
            temparray = np.conjugate(temparray)

        # Split temparray into negative and positive wave numbers (4 quadrants)
        # 1. Non-negative wave numbers (both kx and ky), quadrant 1: temparray1
        # 2. Negative wave numbers (kx), non-negative wave numbers (ky), quadrant 2: temparray2
        # 3. Negative wave numbers (kx), negative wave numbers (ky), quadrant 3: temparray3
        # 4. Non-negative wave numbers (kx), negative wave numbers (ky), quadrant 4: temparray (after reassignment)
        temparray1 = temparray[:, 0:count, 0:count]
        temparray2 = temparray[:, self._num_bins - 1:count - 1:-1, 0:count]
        temparray3 = temparray[:, self._num_bins - 1:count - 1:-1, self._num_bins - 1:count - 1:-1]
        temparray = temparray[:, 0:count, self._num_bins - 1:count - 1:-1]

        # Allocate temporary array
        temparray4 = np.zeros(shape=(self._nz, self._num_bins, self._num_bins), dtype=self._precision)

        # Forward mode
        # Do the following for each z slice
        # 1. Multiply with Fourier transform of Truncated Kernel Green's function slice for that z
        # 2. Compute weighted sum along z with trapezoidal integration weights
        if not adj:

            for j in range(self._nz):
                temparray5 = temparray1 * self._green_func[j, :, 0:count, 0:count]
                temparray6 = temparray2 * self._green_func[j, :, 1:count + 1, 0:count]
                temparray7 = temparray3 * self._green_func[j, :, 1:count + 1, 1:count + 1]
                temparray8 = temparray * self._green_func[j, :, 0:count, 1:count + 1]

                temparray4[j, 0:count, 0:count] = temparray5.sum(axis=0)
                temparray4[j, count:self._num_bins, 0:count] = temparray6.sum(axis=0)[::-1, :]
                temparray4[j, count:self._num_bins, count:self._num_bins] = temparray7.sum(axis=0)[::-1, ::-1]
                temparray4[j, 0:count, count:self._num_bins] = temparray8.sum(axis=0)[:, ::-1]

            # Compute Inverse Fourier transform along first 2 axis
            temparray4 = scfft.fftshift(scfft.ifftn(temparray4, axes=(1, 2)), axes=(1, 2))

            # Copy into output appropriately
            if not add:
                output *= 0
            output += self._dz * temparray4[
                                 :,
                                 self._start_index:(self._end_index + 1),
                                 self._start_index:(self._end_index + 1)
                                 ]

        # Adjoint mode
        # Do the following for each z slice
        # 1. Multiply with Fourier transform of Truncated Kernel Green's function slice for that z (complex conjugate)
        # 2. Compute sum along z
        # 3. Multiply with integration weight for the z slice
        if adj:

            for j in range(self._nz):
                temparray5 = temparray1 * self._green_func[j, :, 0:count, 0:count]
                temparray6 = temparray2 * self._green_func[j, :, 1:count + 1, 0:count]
                temparray7 = temparray3 * self._green_func[j, :, 1:count + 1, 1:count + 1]
                temparray8 = temparray * self._green_func[j, :, 0:count, 1:count + 1]

                temparray4[j, 0:count, 0:count] = self._mu[j, 0] * temparray5.sum(axis=0)
                temparray4[j, count:self._num_bins, 0:count] = self._mu[j, 0] * temparray6.sum(axis=0)[::-1, :]
                temparray4[j, count:self._num_bins, count:self._num_bins] = \
                    self._mu[j, 0] * temparray7.sum(axis=0)[::-1, ::-1]
                temparray4[j, 0:count, count:self._num_bins] = self._mu[j, 0] * temparray8.sum(axis=0)[:, ::-1]

            # Compute Inverse Fourier transform along first 2 axis
            temparray4 = scfft.fftshift(scfft.ifftn(np.conjugate(temparray4), axes=(1, 2)), axes=(1, 2))

            # Copy into output appropriately
            if not add:
                output *= 0
            output += self._dz * temparray4[
                                 :,
                                 self._start_index:(self._end_index + 1),
                                 self._start_index:(self._end_index + 1)
                                 ]

    def write_green_func(self, green_func_dir=None):
        """
        Write operation parallelized using multiprocessing.

        :param green_func_dir: Directory to save green's function to. If None, use value from class.
        """
        if not self._green_func_set_flag:
            raise ValueError("Class initialized in light mode or Green's function not set, cannot perform operation.")

        else:
            file_dir = self._green_func_dir
            if green_func_dir is not None:
                TypeChecker.check(x=green_func_dir, expected_type=(str,))
                if not os.path.exists(green_func_dir):
                    os.makedirs(green_func_dir)
                file_dir = green_func_dir

            # Calculate number of bytes
            num_bytes = self._nz * self._nz * self._num_bins_non_neg * self._num_bins_non_neg
            if self._precision == np.complex64:
                num_bytes *= 8
            if self._precision == np.complex128:
                num_bytes *= 16

            t1 = time.time()

            # Write in multiprocessing mode
            with SharedMemoryManager() as smm:

                # Create shared memory
                sm = smm.SharedMemory(size=num_bytes)
                data = ndarray(shape=self._green_func.shape, dtype=self._precision, buffer=sm.buf)
                data *= 0
                data += self._green_func

                param_tuple_list = [
                    (
                        file_dir,
                        ii,
                        self._nz,
                        self._n,
                        self._precision,
                        sm.name
                    ) for ii in range(self._nz)
                ]

                print("\nWriting Green's function...")

                with Pool(min(len(param_tuple_list), mp.cpu_count(), self._num_threads)) as pool:
                    max_ = len(param_tuple_list)

                    with tqdm(total=max_) as pbar:
                        for _ in pool.imap_unordered(func_write3D, param_tuple_list):
                            pbar.update()

            t2 = time.time()
            print("\n")
            print("Time needed to write Green's function to disk = ", "{:6.2f}".format(t2 - t1), " s")
            print("\n")

    @property
    def greens_func(self):
        if not self._green_func_set_flag:
            raise ValueError("Class initialized in light mode or Green's function not set, cannot perform operation.")
        else:
            return self._green_func

    @greens_func.setter
    def greens_func(self, green_func):

        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")

        TypeChecker.check_ndarray(
            x=green_func,
            dtypes=(self._precision,),
            shape=(self._nz, self._nz, self._num_bins_non_neg, self._num_bins_non_neg)
        )

        self._green_func = green_func
        self._green_func_set_flag = True

    @property
    def green_func_bytes(self):

        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")

        # Calculate number of bytes
        num_bytes = self._nz * self._nz * self._num_bins_non_neg * self._num_bins_non_neg
        if self._precision == np.complex64:
            num_bytes *= 8
        if self._precision == np.complex128:
            num_bytes *= 16

        return num_bytes

    @property
    def green_func_shape(self):

        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")

        return (self._nz, self._nz, self._num_bins_non_neg, self._num_bins_non_neg)

    @property
    def n(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._n

    @property
    def nz(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._nz

    @property
    def k(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._k

    @property
    def a(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._a

    @property
    def b(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._b

    @property
    def vz(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._vz

    @property
    def m(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._m

    @property
    def sigma(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._sigma

    @property
    def precision(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._precision

    @property
    def green_func_dir(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._green_func_dir

    def __calculate_green_func(self):

        t11 = time.time()
        print("\nStarting Green's Function calculation ")

        # Extra cells to deal with Helmholtz solver issues on top & bottom
        num_extra_cells_z = 10
        num_extra_cells_x = max(10, self._m)

        # Calculate parameters for 2D radial Helmholtz solver
        d = self._d / self._m
        dz = self._dz / self._m
        pml_width = 2 * np.pi * np.max(self._vz) / self._k
        pml_cells_x = int(np.ceil(pml_width / d))
        pml_cells_z = int(np.ceil(pml_width / dz))

        num_cells_x = int(np.ceil(self._cutoff2 / d)) + num_extra_cells_x + pml_cells_x
        num_cells_z = self._m * (self._nz - 1) + 2 * (num_extra_cells_z + pml_cells_z)

        extent_x = d * num_cells_x
        extent_z = dz * num_cells_z

        # Prepare velocity
        vz_extend = self.__make_velocity_from_trace(
            vz=self._vz,
            nz=self._m * (self._nz - 1) + 1,
            nx=1
        )

        vz_extend = self.__extend_vel_trace_1d(
            vz=vz_extend,
            pad_cells=num_extra_cells_z + pml_cells_z
        )

        vel = self.__make_velocity_from_trace(
            vz=vz_extend,
            nz=num_cells_z + 1,
            nx=num_cells_x + 1
        )

        # ----------------------------------------------------------------------
        # Solve Helmholtz equation for Green's function for all depth levels

        # Create Helmholtz matrix
        print("---------------------------------------------------")
        print("Creating helmholtz matrix and factorizing it...")
        t1 = time.time()
        mat = create_helmholtz2d_matrix_radial(
            a1=extent_z,
            a2=extent_x,
            pad1=pml_cells_z,
            pad2=pml_cells_x,
            omega=self._k,
            precision=self._precision,
            vel=vel,
            pml_damping=50.0,
            adj=False,
            warnings=True
        )
        mat_lu = splu(mat)
        t2 = time.time()
        print("Helmholtz matrix created and factorized in ", "{:6.2f}".format(t2 - t1), " s")
        print("\n")

        # Source creation in parallel using numba
        print("---------------------------------------------------")
        print("Creating sources for Helmholtz solves...")
        t1 = time.time()

        grid_z = np.linspace(start=0, stop=extent_z, num=num_cells_z + 1, endpoint=True)
        grid_x = np.linspace(start=0, stop=extent_x, num=num_cells_x + 1, endpoint=True)
        z1, x1 = np.meshgrid(grid_z, grid_x, indexing="ij")
        sources = np.zeros(shape=(self._nz, num_cells_z + 1, num_cells_x + 1), dtype=np.float32)

        self.__create_sources_for_helmholtz(
            sources,
            z1=z1,
            x1=x1,
            num_slices=self._nz,
            start_index_z=num_extra_cells_z + pml_cells_z,
            dz=dz,
            m=self._m,
            sigma=self._sigma
        )
        sources = sources.astype(self._precision)

        t2 = time.time()
        print("\nSources created in ", "{:6.2f}".format(t2 - t1), " s")
        print("\n")

        # Solve the Helmholtz equation for all rhs
        print("---------------------------------------------------")
        print("Starting Helmholtz solves...")
        t1 = time.time()

        sol = sources * 0.0
        for i in range(self._nz):
            print("Solving for depth slice ", i)
            sol1 = mat_lu.solve(
                np.reshape(
                    sources[i, :, :], newshape=((num_cells_z + 1) * (num_cells_x + 1), 1)
                )
            )
            sol[i, :, :] += np.reshape(sol1, newshape=(num_cells_z + 1, num_cells_x + 1))

        t2 = time.time()
        print("\nHelmholtz solves completed in ", "{:6.2f}".format(t2 - t1), " s")
        print("\n")

        # ----------------------------------------------------------------------
        # Compute truncated kernel

        # Parallelize using multiprocessing
        if not self._no_mpi:

            # Calculate number of bytes
            num_bytes = self._nz * self._nz * self._num_bins_non_neg * self._num_bins_non_neg
            if self._precision == np.complex64:
                num_bytes *= 8
            if self._precision == np.complex128:
                num_bytes *= 16

            # Read in multiprocessing mode
            with SharedMemoryManager() as smm:

                # Create shared memory
                sm = smm.SharedMemory(size=num_bytes)
                temp = ndarray(
                    shape=(self._nz, self._nz, self._num_bins_non_neg, self._num_bins_non_neg),
                    dtype=self._precision,
                    buffer=sm.buf
                )
                temp *= 0

                param_tuple_list = [
                    (
                        sol[ii, :, :],
                        self._precision,
                        num_extra_cells_z + pml_cells_z,
                        self._m * (self._nz - 1) + num_extra_cells_z + pml_cells_z,
                        self._m,
                        self._num_bins_non_neg,
                        self._cutoff_func_fine_grid,
                        d,
                        ii,
                        sm.name,
                        self._verbose
                    ) for ii in range(self._nz)
                ]

                with Pool(min(len(param_tuple_list), mp.cpu_count(), self._num_threads)) as pool:
                    max_ = len(param_tuple_list)

                    with tqdm(total=max_) as pbar:
                        for _ in pool.imap_unordered(func_helper3D, param_tuple_list):
                            pbar.update()

                self._green_func = temp * 1.0

            # Write Green's function to disk
            self.write_green_func()

        t22 = time.time()
        print("\nComputing 3d Green's Function took ", "{:6.2f}".format(t22 - t11), " s\n")
        print("\nGreen's Function size in memory (Gb) : ", "{:6.2f}".format(sys.getsizeof(self._green_func) / 1e9))
        print("\n")

    @staticmethod
    @numba.njit(parallel=True)
    def __create_sources_for_helmholtz(sources, z1, x1, num_slices, start_index_z, dz, m, sigma):

        for num_slice in numba.prange(num_slices):
            print("Computing source number ", num_slice)

            gaussian_center_z = (start_index_z + num_slice * m) * dz
            sources[num_slice, :, :] += np.exp(-0.5 * ((z1 - gaussian_center_z) ** 2.0 + x1 ** 2.0) / (sigma ** 2.0))

        sources /= (2 * np.pi * sigma * sigma) ** 1.5

    @staticmethod
    @numba.njit
    def __extend_vel_trace_1d(vz, pad_cells):
        """
        :param vz: velocity values as a 1D numpy array of shape [nz_in, 1], assumed dtype = np.float32
        :param pad_cells: number of cells to pad along each end
        """

        nz_in, _ = vz.shape
        nz_out = nz_in + 2 * pad_cells

        vz_out = np.zeros(shape=(nz_out, 1), dtype=np.float32)
        vz_out[pad_cells: pad_cells + nz_in, 0] = vz.flatten()
        vz_out[0: pad_cells, 0] = vz_out[pad_cells, 0]
        vz_out[pad_cells + nz_in: nz_out, 0] = vz_out[pad_cells + nz_in - 1, 0]

        return vz_out

    @staticmethod
    def __make_velocity_from_trace(vz, nz, nx):
        """
        :param vz: velocity values as a 2D numpy array of shape [N, 1], assumed dtype = np.float32
        :param nz: points along z direction
        :param nx: points along x direction

        The trace is first interpolated to a grid with nz points in z direction (same vertical extent).
        Then it is copied nx times in x direction.
        Result is returned as a numpy array or shape (nz, nx).
        """

        nz_in, _ = vz.shape

        z_start = 0.0
        z_end = nz_in

        coord_in = np.linspace(start=z_start, stop=z_end, num=nz_in, endpoint=True)
        val_in = np.reshape(vz, newshape=(nz_in,))

        f = interpolate.interp1d(coord_in, val_in, kind="linear")

        coord_out = np.linspace(start=z_start, stop=z_end, num=nz, endpoint=True)
        val_out = np.reshape(f(coord_out), newshape=(nz, 1))

        vel = np.zeros(shape=(nz, nx), dtype=np.float32)

        for ii in range(nz):
            vel[ii, :] = val_out[ii, 0]

        return vel

    def __check_grid_size(self):

        vmin = np.min(self._vz)
        lambda_min = 2 * np.pi * vmin / self._k
        grid_min = min(self._d, self._dz)

        if grid_min >= lambda_min / 5.0:
            raise ValueError("Minimum grid spacing condition violated")

        if grid_min >= lambda_min / 10.0:
            print("\nWarning: Recommended minimum grid spacing is 10 times smallest wave length.\n")
            print("Recommended minimum grid spacing", lambda_min / 10.0)
            print("Grid spacing detected", grid_min)

    def __read_green_func(self):

        # Calculate number of bytes
        num_bytes = self._nz * self._nz * self._num_bins_non_neg * self._num_bins_non_neg
        if self._precision == np.complex64:
            num_bytes *= 8
        if self._precision == np.complex128:
            num_bytes *= 16

        t1 = time.time()

        # Read in multiprocessing mode
        with SharedMemoryManager() as smm:

            # Create shared memory
            sm = smm.SharedMemory(size=num_bytes)
            self._green_func = ndarray(
                shape=(self._nz, self._nz, self._num_bins_non_neg, self._num_bins_non_neg),
                dtype=self._precision,
                buffer=sm.buf
            )
            self._green_func *= 0

            param_tuple_list = [
                (
                    self._green_func_dir,
                    ii,
                    self._nz,
                    self._n,
                    self._precision,
                    sm.name
                ) for ii in range(self._nz)
            ]

            print("\nReading Green's function...")

            with Pool(min(len(param_tuple_list), mp.cpu_count(), self._num_threads)) as pool:
                max_ = len(param_tuple_list)

                with tqdm(total=max_) as pbar:
                    for _ in pool.imap_unordered(func_read3D, param_tuple_list):
                        pbar.update()

        t2 = time.time()
        print("\n")
        print("Time needed to read Green's function from disk = ", "{:6.2f}".format(t2 - t1), " s")
        print("\n")

    def __calculate_cutoff_func(self):
        """
        Calculate cutoff function for use in Green's func calculation.
        Calculate for both fine and coarse grid, which starts at x=0.0, stops at x=2.0.
        The grid spacing for fine grid is self._d * self._m
        The grid spacing for coarse grid is self._d

        The function has the following formula:
            for all x <= self._cutoff1:
                f(x) = 1,
            for all x >= self._cutoff2:
                f(x) = 0,
            for all other x:
                f(x) = (e^(-1.0 / (self._cutoff2 - x))) /
                (e^(-1.0 / (self._cutoff2 - x)) + e^(-1.0 / (x - self._cutoff1)))
        """

        cutoff_func_coarse_grid = np.zeros(shape=(self._num_bins_non_neg,), dtype=np.float32)
        cutoff_func_fine_grid = np.zeros(shape=(self._m * (self._num_bins_non_neg - 1) + 1,), dtype=np.float32)

        # Coarse grid calculation
        for ii in range(self._num_bins_non_neg):
            x = ii * self._d
            if x <= self._cutoff1:
                cutoff_func_coarse_grid[ii] = 1.0
            elif x >= self._cutoff2:
                cutoff_func_coarse_grid[ii] = 0.0
            else:
                t1 = np.exp(-1.0 / (self._cutoff2 - x))
                t2 = np.exp(-1.0 / (x - self._cutoff1))
                cutoff_func_coarse_grid[ii] = t1 / (t1 + t2)

        # Fine grid calculation
        d1 = self._d / self._m
        for ii in range(self._m * (self._num_bins_non_neg - 1) + 1):
            x = ii * d1
            if x <= self._cutoff1:
                cutoff_func_fine_grid[ii] = 1.0
            elif x >= self._cutoff2:
                cutoff_func_fine_grid[ii] = 0.0
            else:
                t1 = np.exp(-1.0 / (self._cutoff2 - x))
                t2 = np.exp(-1.0 / (x - self._cutoff1))
                cutoff_func_fine_grid[ii] = t1 / (t1 + t2)

        return cutoff_func_coarse_grid, cutoff_func_fine_grid

    def __initialize_class(self, green_func_flag=True):

        # Calculate number of grid points for the domain [-2, 2] along one horizontal axis,
        # and index to crop the physical domain [-0.5, 0.5]
        self._num_bins = 4 * (self._n - 1)
        self._start_index = 3 * int((self._n - 1) / 2)
        self._end_index = 5 * int((self._n - 1) / 2)

        # Calculate horizontal grid spacing d
        # Calculate horizontal grid of wavenumbers for any 1 dimension
        self._d = 1.0 / (self._n - 1)
        self._kgrid = 2 * np.pi * scfft.fftshift(scfft.fftfreq(n=self._num_bins, d=self._d))

        # Store only non-negative wave numbers
        self._num_bins_non_neg = 2 * (self._n - 1) + 1
        self._kgrid_non_neg = np.abs(self._kgrid[0:self._num_bins_non_neg][::-1])

        # Calculate z grid spacing d
        # Calculate z grid coordinates
        self._dz = (self._b - self._a) / (self._nz - 1)
        self._zgrid = np.linspace(start=self._a, stop=self._b, num=self._nz, endpoint=True)

        # Check grid size
        self.__check_grid_size()

        # Calculate cutoff functions
        self._cutoff_func_coarse_grid, self._cutoff_func_fine_grid = self.__calculate_cutoff_func()

        # Calculate FT of Truncated Green's Function
        # Write Green's function to disk
        if green_func_flag:
            self.__calculate_green_func()

        # Calculate integration weights
        self._mu = np.zeros(shape=(self._nz, 1, 1), dtype=np.float32) + 1.0
        self._mu[0, 0, 0] = 0.5
        self._mu[self._nz - 1, 0, 0] = 0.5


class TruncatedKernelGeneralVz2d:

    def __init__(self, n, nz, a, b, k, vz, m, sigma, precision, green_func_dir, num_threads, no_mpi=False, verbose=False, light_mode=False):
        """
        The Helmholtz equation reads (lap + k^2 / vz^2)u = f, on the domain [a,b] x [-0.5, 0.5].

        :param n: The field will have shape n x nz, with n odd. This means that the cube
        [-0.5, 0.5] is gridded into n points. The points have coordinates {-0.5 + k/(n-1) : 0 <= k <= n-1}.
        :param nz: The field will have shape n x nz, with n odd. This means that the interval [a, b] is gridded
        into nz points. The points have coordinates {a + (b-a)k/(nz-1) : 0 <= k <= nz-1}.
        :param a: The domain in z direction is [a, b] with 0 < a < b.
        :param b: The domain in z direction is [a, b] with 0 < a < b.
        :param k: The frequency k.
        :param vz: The 1D vz velocity profile. Must be ndarray of shape (nz, 1).
        :param m: Decimation factor for calculating Green's function on a fine grid.
        :param sigma: The standard deviation of delta to inject for Green's function calculation.
        :param precision: np.complex64 or np.complex128
        :param green_func_dir: Name of directory where to read / write Green's function from.
        :param num_threads: Maximum number of threads to use
        :param no_mpi: bool (if False do not use multiprocessing)
        :param verbose: bool (if True print messages during Green's function calculation).
        :param light_mode: bool (if True an empty class is initialized)
        """

        TypeChecker.check(x=light_mode, expected_type=(bool,))
        self._initialized_flag = not light_mode
        self._green_func_set_flag = not light_mode

        if not light_mode:

            print("\n\nInitializing the class")

            TypeChecker.check(x=n, expected_type=(int,))
            if n % 2 != 1 or n < 3:
                raise ValueError("n must be an odd integer >= 3")

            TypeChecker.check(x=nz, expected_type=(int,))
            if nz < 2:
                raise ValueError("n must be an integer >= 2")

            TypeChecker.check(x=a, expected_type=(float,))
            TypeChecker.check_float_strict_lower_bound(x=b, lb=a)
            TypeChecker.check_float_positive(k)

            # Minimum velocity of 0.1 km/s
            TypeChecker.check_ndarray(x=vz, shape=(nz, 1), dtypes=(np.float32,), lb=0.1)
            TypeChecker.check_int_positive(m)

            TypeChecker.check_float_positive(x=sigma)

            if precision not in [np.complex64, np.complex128]:
                raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

            TypeChecker.check(x=green_func_dir, expected_type=(str,))
            if not os.path.exists(green_func_dir):
                os.makedirs(green_func_dir)

            TypeChecker.check_int_positive(x=num_threads)
            TypeChecker.check(x=no_mpi, expected_type=(bool,))
            TypeChecker.check(x=verbose, expected_type=(bool,))

            self._n = n
            self._nz = nz
            self._k = k
            self._a = a
            self._b = b
            self._vz = vz
            self._m = m
            self._sigma = sigma
            self._precision = precision
            self._green_func_dir = green_func_dir
            self._num_threads = num_threads
            self._no_mpi = no_mpi
            self._verbose = verbose

            self._cutoff1 = 1.0
            self._cutoff2 = 1.1

            # Run class initializer
            self.__initialize_class()

    def set_parameters(self, n, nz, a, b, k, vz, m, sigma, precision, green_func_dir, num_threads, green_func_set=True, no_mpi=False, verbose=False):
        """
        The Helmholtz equation reads (lap + k^2 / vz^2)u = f, on the domain [a,b] x [-0.5, 0.5].

        :param n: The field will have shape n x nz, with n odd. This means that the cube
        [-0.5, 0.5] is gridded into n points. The points have coordinates {-0.5 + k/(n-1) : 0 <= k <= n-1}.
        :param nz: The field will have shape n x nz, with n odd. This means that the interval [a, b] is gridded
        into nz points. The points have coordinates {a + (b-a)k/(nz-1) : 0 <= k <= nz-1}.
        :param a: The domain in z direction is [a, b] with 0 < a < b.
        :param b: The domain in z direction is [a, b] with 0 < a < b.
        :param k: The frequency k.
        :param vz: The 1D vz velocity profile. Must be ndarray of shape (nz, 1).
        :param m: Decimation factor for calculating Green's function on a fine grid.
        :param sigma: The standard deviation of delta to inject for Green's function calculation.
        :param precision: np.complex64 or np.complex128
        :param green_func_dir: Name of directory where to read / write Green's function from.
        :param num_threads: Maximum number of threads to use
        :param green_func_set: bool (if False do not set Green's function)
        :param no_mpi: bool (if False do not use multiprocessing)
        :param verbose: bool (if True print messages during Green's function calculation).
        """

        print("\n\nInitializing the class")

        TypeChecker.check(x=n, expected_type=(int,))
        if n % 2 != 1 or n < 3:
            raise ValueError("n must be an odd integer >= 3")

        TypeChecker.check(x=nz, expected_type=(int,))
        if nz < 2:
            raise ValueError("n must be an integer >= 2")

        TypeChecker.check(x=a, expected_type=(float,))
        TypeChecker.check_float_strict_lower_bound(x=b, lb=a)
        TypeChecker.check_float_positive(k)

        # Minimum velocity of 0.1 km/s
        TypeChecker.check_ndarray(x=vz, shape=(nz, 1), dtypes=(np.float32,), lb=0.1)
        TypeChecker.check_int_positive(m)

        TypeChecker.check_float_positive(x=sigma)

        if precision not in [np.complex64, np.complex128]:
            raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

        TypeChecker.check(x=green_func_dir, expected_type=(str,))
        if not os.path.exists(green_func_dir):
            print("\n")
            print("Green's function directory does not exist or is empty.")
            print("Class set_parameters method failed. Exiting without changes to class.")
            print("\n")
            return

        file_name = os.path.join(green_func_dir, "green_func.npz")
        if not os.path.exists(file_name):
            print("\n")
            print("Green's function file does not exist.")
            print("Class set_parameters method failed. Exiting without changes to class.")
            print("\n")

        TypeChecker.check_int_positive(x=num_threads)
        TypeChecker.check(x=green_func_set, expected_type=(bool,))
        TypeChecker.check(x=no_mpi, expected_type=(bool,))
        TypeChecker.check(x=verbose, expected_type=(bool,))

        self._n = n
        self._nz = nz
        self._k = k
        self._a = a
        self._b = b
        self._vz = vz
        self._m = m
        self._sigma = sigma
        self._precision = precision
        self._green_func_dir = green_func_dir
        self._num_threads = num_threads
        self._no_mpi = no_mpi
        self._verbose = verbose

        self._cutoff1 = 1.0
        self._cutoff2 = 1.2

        self.__initialize_class(green_func_flag=False)
        if green_func_set:
            self.__read_green_func()
        self._initialized_flag = True
        self._green_func_set_flag = green_func_set

    def apply_kernel(self, u, output, adj=False, add=False):
        """
        :param u: 2d numpy array (must be nz x n dimensions with n odd).
        :param output: 2d numpy array (same dimension as u).
        :param adj: Boolean flag (forward or adjoint operator)
        :param add: Boolean flag (whether to add result to output)
        """

        if not self._green_func_set_flag:
            raise ValueError("Class initialized in light mode or Green's function not set, cannot perform operation.")

        # Check types of input
        if u.dtype != self._precision or output.dtype != self._precision:
            raise TypeError("Types of 'u' and 'output' must match that of class: ", self._precision)

        # Check dimensions
        if u.shape != (self._nz, self._n) or output.shape != (self._nz, self._n):
            raise ValueError("Shapes of 'u' and 'output' must be (nz, n) with nz = ", self._nz, " and n = ", self._n)

        # Count of non-negative wave numbers
        count = self._num_bins_non_neg - 1

        # Copy u into temparray and compute Fourier transform along first axis
        temparray = np.zeros(shape=(self._nz, self._num_bins), dtype=self._precision)
        temparray[:, self._start_index:(self._end_index + 1)] = u
        temparray = scfft.fftn(scfft.fftshift(temparray, axes=(1,)), axes=(1,))
        if not adj:
            temparray *= self._mu
        else:
            temparray = np.conjugate(temparray)

        # Split temparray into negative and positive wave numbers
        # 1. Non-negative wave numbers: temparray1
        # 2. Negative wave numbers: temparray (after reassignment)
        temparray1 = temparray[:, 0:count]
        temparray = temparray[:, self._num_bins - 1:count - 1:-1]

        # Allocate temporary array
        temparray2 = np.zeros(shape=(self._nz, self._num_bins), dtype=self._precision)

        # Forward mode
        # Do the following for each z slice
        # 1. Multiply with Fourier transform of Truncated Kernel Green's function slice for that z
        # 2. Compute weighted sum along z with trapezoidal integration weights
        if not adj:

            for j in range(self._nz):
                temparray3 = temparray1 * self._green_func[j, :, 0:count]
                temparray4 = temparray * self._green_func[j, :, 1:count + 1]

                temparray2[j, 0:count] = temparray3.sum(axis=0)
                temparray2[j, count:self._num_bins] = temparray4.sum(axis=0)[::-1]

            # Compute Inverse Fourier transform along first axis
            temparray2 = scfft.fftshift(scfft.ifftn(temparray2, axes=(1,)), axes=(1,))

            # Copy into output appropriately
            if not add:
                output *= 0
            output += self._dz * temparray2[:, self._start_index:(self._end_index + 1)]

        # Adjoint mode
        # Do the following for each z slice
        # 1. Multiply with Fourier transform of Truncated Kernel Green's function slice for that z (complex conjugate)
        # 2. Compute sum along z
        # 3. Multiply with integration weight for the z slice
        if adj:

            for j in range(self._nz):
                temparray3 = temparray1 * self._green_func[j, :, 0:count]
                temparray4 = temparray * self._green_func[j, :, 1:count + 1]

                temparray2[j, 0:count] = self._mu[j, 0] * temparray3.sum(axis=0)
                temparray2[j, count:self._num_bins] = self._mu[j, 0] * temparray4.sum(axis=0)[::-1]

            # Compute Inverse Fourier transform along first axis
            temparray2 = scfft.fftshift(scfft.ifftn(np.conjugate(temparray2), axes=(1,)), axes=(1,))

            # Copy into output appropriately
            if not add:
                output *= 0
            output += self._dz * temparray2[:, self._start_index:(self._end_index + 1)]

    def write_green_func(self, green_func_dir=None):
        """
        :param green_func_dir: Directory to save green's function to. If None, use value from class.
        """
        if not self._green_func_set_flag:
            raise ValueError("Class initialized in light mode or Green's function not set, cannot perform operation.")

        else:
            file_dir = self._green_func_dir
            if green_func_dir is not None:
                TypeChecker.check(x=green_func_dir, expected_type=(str,))
                if not os.path.exists(green_func_dir):
                    os.makedirs(green_func_dir)
                file_dir = green_func_dir

            np.savez(os.path.join(file_dir, "green_func.npz"), self._green_func)

    @property
    def greens_func(self):
        if not self._green_func_set_flag:
            raise ValueError("Class initialized in light mode or Green's function not set, cannot perform operation.")
        else:
            return self._green_func

    @greens_func.setter
    def greens_func(self, green_func):

        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")

        TypeChecker.check_ndarray(
            x=green_func,
            dtypes=(self._precision,),
            shape=(self._nz, self._nz, self._num_bins_non_neg)
        )

        self._green_func = green_func
        self._green_func_set_flag = True

    @property
    def green_func_bytes(self):

        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")

        # Calculate number of bytes
        num_bytes = self._nz * self._nz * self._num_bins_non_neg
        if self._precision == np.complex64:
            num_bytes *= 8
        if self._precision == np.complex128:
            num_bytes *= 16

        return num_bytes

    @property
    def green_func_shape(self):

        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")

        return (self._nz, self._nz, self._num_bins_non_neg)

    @property
    def n(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._n

    @property
    def nz(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._nz

    @property
    def k(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._k

    @property
    def a(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._a

    @property
    def b(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._b

    @property
    def vz(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._vz

    @property
    def m(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._m

    @property
    def sigma(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._sigma

    @property
    def precision(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._precision

    @property
    def green_func_dir(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._green_func_dir

    def __calculate_green_func(self):

        t11 = time.time()
        print("\nStarting Green's Function calculation ")

        # Extra cells to deal with Helmholtz solver issues on top & bottom
        num_extra_cells_z = 10
        num_extra_cells_x = max(10, self._m)

        # Calculate parameters for 2D even Helmholtz solver
        d = self._d / self._m
        dz = self._dz / self._m
        pml_width = 2 * np.pi * np.max(self._vz) / self._k
        pml_cells_x = int(np.ceil(pml_width / d))
        pml_cells_z = int(np.ceil(pml_width / dz))

        num_cells_x = int(np.ceil(self._cutoff2 / d)) + num_extra_cells_x + pml_cells_x
        num_cells_z = self._m * (self._nz - 1) + 2 * (num_extra_cells_z + pml_cells_z)

        extent_x = d * num_cells_x
        extent_z = dz * num_cells_z

        # Prepare velocity
        vz_extend = self.__make_velocity_from_trace(
            vz=self._vz,
            nz=self._m * (self._nz - 1) + 1,
            nx=1
        )

        vz_extend = self.__extend_vel_trace_1d(
            vz=vz_extend,
            pad_cells=num_extra_cells_z + pml_cells_z
        )

        vel = self.__make_velocity_from_trace(
            vz=vz_extend,
            nz=num_cells_z + 1,
            nx=num_cells_x + 1
        )

        # ----------------------------------------------------------------------
        # Solve Helmholtz equation for Green's function for all depth levels

        # Create Helmholtz matrix
        print("---------------------------------------------------")
        print("Creating helmholtz matrix and factorizing it...")
        t1 = time.time()
        mat = create_helmholtz2d_matrix_even(
            a1=extent_z,
            a2=extent_x,
            pad1=pml_cells_z,
            pad2=pml_cells_x,
            omega=self._k,
            precision=self._precision,
            vel=vel,
            pml_damping=50.0,
            adj=False,
            warnings=True
        )
        mat_lu = splu(mat)
        t2 = time.time()
        print("Helmholtz matrix created and factorized in ", "{:6.2f}".format(t2 - t1), " s")
        print("\n")

        # Source creation in parallel using numba
        print("---------------------------------------------------")
        print("Creating sources for Helmholtz solves...")
        t1 = time.time()

        grid_z = np.linspace(start=0, stop=extent_z, num=num_cells_z + 1, endpoint=True)
        grid_x = np.linspace(start=0, stop=extent_x, num=num_cells_x + 1, endpoint=True)
        z1, x1 = np.meshgrid(grid_z, grid_x, indexing="ij")
        sources = np.zeros(shape=(self._nz, num_cells_z + 1, num_cells_x + 1), dtype=np.float32)

        self.__create_sources_for_helmholtz(
            sources,
            z1=z1,
            x1=x1,
            num_slices=self._nz,
            start_index_z=num_extra_cells_z + pml_cells_z,
            dz=dz,
            m=self._m,
            sigma=self._sigma
        )
        sources = sources.astype(self._precision)

        t2 = time.time()
        print("\nSources created in ", "{:6.2f}".format(t2 - t1), " s")
        print("\n")

        # Solve the Helmholtz equation for all rhs
        print("---------------------------------------------------")
        print("Starting Helmholtz solves...")
        t1 = time.time()

        sol = sources * 0.0
        for i in range(self._nz):
            print("Solving for depth slice ", i)
            sol1 = mat_lu.solve(
                np.reshape(
                    sources[i, :, :], newshape=((num_cells_z + 1) * (num_cells_x + 1), 1)
                )
            )
            sol[i, :, :] += np.reshape(sol1, newshape=(num_cells_z + 1, num_cells_x + 1))

        t2 = time.time()
        print("\nHelmholtz solves completed in ", "{:6.2f}".format(t2 - t1), " s")
        print("\n")

        # ----------------------------------------------------------------------
        # Compute truncated kernel

        # Parallelize using multiprocessing
        if not self._no_mpi:

            # Calculate number of bytes
            num_bytes = self._nz * self._nz * self._num_bins_non_neg
            if self._precision == np.complex64:
                num_bytes *= 8
            if self._precision == np.complex128:
                num_bytes *= 16

            # Read in multiprocessing mode
            with SharedMemoryManager() as smm:

                # Create shared memory
                sm = smm.SharedMemory(size=num_bytes)
                temp = ndarray(
                    shape=(self._nz, self._nz, self._num_bins_non_neg),
                    dtype=self._precision,
                    buffer=sm.buf
                )
                temp *= 0

                param_tuple_list = [
                    (
                        sol[ii, :, :],
                        self._precision,
                        num_extra_cells_z + pml_cells_z,
                        self._m * (self._nz - 1) + num_extra_cells_z + pml_cells_z,
                        self._m,
                        self._cutoff_func_coarse_grid,
                        d,
                        ii,
                        sm.name,
                        self._verbose
                    ) for ii in range(self._nz)
                ]

                with Pool(min(len(param_tuple_list), mp.cpu_count(), self._num_threads)) as pool:
                    max_ = len(param_tuple_list)

                    with tqdm(total=max_) as pbar:
                        for _ in pool.imap_unordered(func_helper2D, param_tuple_list):
                            pbar.update()

                self._green_func = temp * 1.0

            # Write Green's function to disk
            self.write_green_func()

        # Do not parallelize
        else:

            self._green_func = np.zeros(
                shape=(self._nz, self._nz, self._num_bins_non_neg), dtype=self._precision
            )

            for ii in range(self._nz):
                func_helper2D_no_mpi(
                    green_func=sol[ii, :, :],
                    precision=self._precision,
                    start_index_z=num_extra_cells_z + pml_cells_z,
                    end_index_z=self._m * (self._nz - 1) + num_extra_cells_z + pml_cells_z,
                    m=self._m,
                    cutoff_func=self._cutoff_func_coarse_grid,
                    dx=d,
                    num_slice=ii,
                    green_func_truncated_ft=self._green_func,
                    verbose=False
                )

            # Write Green's function to disk
            self.write_green_func()

        t22 = time.time()
        print("\nComputing 2d Green's function took ", "{:6.2f}".format(t22 - t11), " s\n")
        print("\nGreen's Function size in memory (Mb) : ", "{:6.2f}".format(sys.getsizeof(self._green_func) / 1e6))
        print("\n")

    @staticmethod
    @numba.njit(parallel=True)
    def __create_sources_for_helmholtz(sources, z1, x1, num_slices, start_index_z, dz, m, sigma):

        for num_slice in numba.prange(num_slices):

            print("Computing source number ", num_slice)

            gaussian_center_z = (start_index_z + num_slice * m) * dz
            sources[num_slice, :, :] += np.exp(-0.5 * ((z1 - gaussian_center_z) ** 2.0 + x1 ** 2.0) / (sigma ** 2.0))

        sources /= 2 * np.pi * sigma * sigma

    @staticmethod
    @numba.njit
    def __extend_vel_trace_1d(vz, pad_cells):
        """
        :param vz: velocity values as a 1D numpy array of shape [nz_in, 1], assumed dtype = np.float32
        :param pad_cells: number of cells to pad along each end
        """

        nz_in, _ = vz.shape
        nz_out = nz_in + 2 * pad_cells

        vz_out = np.zeros(shape=(nz_out, 1), dtype=np.float32)
        vz_out[pad_cells: pad_cells + nz_in, 0] = vz.flatten()
        vz_out[0: pad_cells, 0] = vz_out[pad_cells, 0]
        vz_out[pad_cells + nz_in: nz_out, 0] = vz_out[pad_cells + nz_in - 1, 0]

        return vz_out

    @staticmethod
    def __make_velocity_from_trace(vz, nz, nx):
        """
        :param vz: velocity values as a 2D numpy array of shape [N, 1], assumed dtype = np.float32
        :param nz: points along z direction
        :param nx: points along x direction

        The trace is first interpolated to a grid with nz points in z direction (same vertical extent).
        Then it is copied nx times in x direction.
        Result is returned as a numpy array or shape (nz, nx).
        """

        nz_in, _ = vz.shape

        z_start = 0.0
        z_end = nz_in

        coord_in = np.linspace(start=z_start, stop=z_end, num=nz_in, endpoint=True)
        val_in = np.reshape(vz, newshape=(nz_in,))

        f = interpolate.interp1d(coord_in, val_in, kind="linear")

        coord_out = np.linspace(start=z_start, stop=z_end, num=nz, endpoint=True)
        val_out = np.reshape(f(coord_out), newshape=(nz, 1))

        vel = np.zeros(shape=(nz, nx), dtype=np.float32)

        for ii in range(nz):
            vel[ii, :] = val_out[ii, 0]

        return vel

    def __check_grid_size(self):

        vmin = np.min(self._vz)
        lambda_min = 2 * np.pi * vmin / self._k
        grid_min = min(self._d, self._dz)

        if grid_min >= lambda_min / 5.0:
            raise ValueError("Minimum grid spacing condition violated")

        if grid_min >= lambda_min / 10.0:
            print("\nWarning: Recommended minimum grid spacing is 10 times smallest wave length.")
            print("Recommended minimum grid spacing", lambda_min / 10.0)
            print("Grid spacing detected", grid_min)

    def __read_green_func(self):

        file_name = os.path.join(self._green_func_dir, "green_func.npz")
        with np.load(file_name) as f:
            data = f["arr_0"]

        TypeChecker.check_ndarray(
            x=data,
            dtypes=(self._precision,),
            shape=(self._nz, self._nz, self._num_bins_non_neg)
        )

        self._green_func = data * 1.0

    def __calculate_cutoff_func(self):
        """
        Calculate cutoff function for use in Green's func calculation.
        Calculate for both fine and coarse grid, which starts at x=0.0, stops at x=2.0.
        The grid spacing for fine grid is self._d * self._m
        The grid spacing for coarse grid is self._d

        The function has the following formula:
            for all x <= self._cutoff1:
                f(x) = 1,
            for all x >= self._cutoff2:
                f(x) = 0,
            for all other x:
                f(x) = (e^(-1.0 / (self._cutoff2 - x))) /
                (e^(-1.0 / (self._cutoff2 - x)) + e^(-1.0 / (x - self._cutoff1)))
        """

        cutoff_func_coarse_grid = np.zeros(shape=(self._num_bins_non_neg,), dtype=np.float32)

        # Coarse grid calculation
        for ii in range(self._num_bins_non_neg):
            x = ii * self._d
            if x <= self._cutoff1:
                cutoff_func_coarse_grid[ii] = 1.0
            elif x >= self._cutoff2:
                cutoff_func_coarse_grid[ii] = 0.0
            else:
                t1 = np.exp(-1.0 / (self._cutoff2 - x))
                t2 = np.exp(-1.0 / (x - self._cutoff1))
                cutoff_func_coarse_grid[ii] = t1 / (t1 + t2)

        return cutoff_func_coarse_grid

    def __initialize_class(self, green_func_flag=True):

        # Calculate number of grid points for the domain [-2, 2] along one horizontal axis,
        # and index to crop the physical domain [-0.5, 0.5]
        self._num_bins = 4 * (self._n - 1)
        self._start_index = 3 * int((self._n - 1) / 2)
        self._end_index = 5 * int((self._n - 1) / 2)

        # Calculate horizontal grid spacing d
        # Calculate horizontal grid of wave numbers for any 1 dimension
        self._d = 1.0 / (self._n - 1)
        self._kgrid = 2 * np.pi * scfft.fftshift(scfft.fftfreq(n=self._num_bins, d=self._d))

        # Store only non-negative wave numbers
        self._num_bins_non_neg = 2 * (self._n - 1) + 1
        self._kgrid_non_neg = np.abs(self._kgrid[0:self._num_bins_non_neg][::-1])

        # Calculate z grid spacing d
        # Calculate z grid coordinates
        self._dz = (self._b - self._a) / (self._nz - 1)
        self._zgrid = np.linspace(start=self._a, stop=self._b, num=self._nz, endpoint=True)

        # Check grid size
        self.__check_grid_size()

        # Calculate cutoff functions
        self._cutoff_func_coarse_grid = self.__calculate_cutoff_func()

        # Calculate FT of Truncated Green's Function and apply fftshift
        if green_func_flag:
            self.__calculate_green_func()

        # Calculate integration weights
        self._mu = np.zeros(shape=(self._nz, 1), dtype=np.float32) + 1.0
        self._mu[0, 0] = 0.5
        self._mu[self._nz - 1, 0] = 0.5


if __name__ == "__main__":

    # ----------------------------------------------
    # 2d Test

    n_ = 101
    nz_ = 101
    a_ = 0.
    b_ = a_ + (1.0 / (n_ - 1)) * (nz_ - 1)
    freq_ = 10.0
    omega_ = 2 * np.pi * freq_
    m_ = 5
    sigma_ = 0.004
    precision_ = np.complex64
    green_func_dir_ = "Lippmann-Schwinger/Test/Data"
    num_threads_ = 4
    vz_ = np.zeros(shape=(nz_, 1), dtype=np.float32) + 1.0

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

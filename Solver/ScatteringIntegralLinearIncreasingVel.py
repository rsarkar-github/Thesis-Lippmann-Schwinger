import numpy as np
import scipy.fft as scfft
import numba
import scipy.special as sp
import time
import sys
from ..Utilities import TypeChecker
from . import SpecialFunc
import matplotlib.pyplot as plt


class TruncatedKernelLinearIncreasingVel3d:

    def __init__(self, n, nz, k, a, b, m, precision, verbose=False, light_mode=False):
        """
        :param n: The field will have shape n x n x nz, with n odd. This means that the cube
        [-0.5, 0.5]^2 is gridded into n^2 points. The points have coordinates {-0.5 + k/(n-1) : 0 <= k <= n-1}.
        :param nz: The field will have shape n x n x nz, with n odd. This means that the interval [a, b] is gridded
        into nz points. The points have coordinates {a + (b-a)k/(nz-1) : 0 <= k <= nz-1}.
        :param k: The Helmholtz equation reads (lap + k^2 / z^2)u = f.
        :param a: The domain in z direction is [a, b] with 0 < a < b.
        :param b: The domain in z direction is [a, b] with 0 < a < b.
        :param m: Number of points to use in Riemann sum calculator for FT of truncated kernel.
        :param precision: np.complex64 or np.complex128
        :param verbose: bool (if True print messages during Green's function calculation).
        :param light_mode: bool (if True an empty class is initialized)
        """

        TypeChecker.check(x=light_mode, expected_type=(bool,))
        self._initialized_flag = not light_mode

        if not light_mode:

            print("\n\nInitializing the class")

            TypeChecker.check(x=n, expected_type=(int,))
            if n % 2 != 1 or n < 3:
                raise ValueError("n must be an odd integer >= 3")

            TypeChecker.check(x=nz, expected_type=(int,))
            if nz < 2:
                raise ValueError("n must be an integer >= 2")

            TypeChecker.check_float_positive(k)
            TypeChecker.check_float_positive(a)
            TypeChecker.check_float_strict_lower_bound(x=b, lb=a)

            TypeChecker.check_int_positive(m)

            if precision not in [np.complex64, np.complex128]:
                raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

            TypeChecker.check(x=verbose, expected_type=(bool,))

            self._n = n
            self._nz = nz
            self._k = k
            self._a = a
            self._b = b
            self._m = m
            self._precision = precision
            self._verbose = verbose

            self._cutoff = np.sqrt(2.0)

            # Run class initializer
            self.__initialize_class()

    def set_parameters(self, n, nz, k, a, b, m, precision, verbose=False, green_func=None):
        """
        :param n: The field will have shape n x n x nz, with n odd. This means that the cube
        [-0.5, 0.5]^2 is gridded into n^2 points. The points have coordinates {-0.5 + k/(n-1) : 0 <= k <= n-1}.
        :param nz: The field will have shape n x n x nz, with n odd. This means that the interval [a, b] is gridded
        into nz points. The points have coordinates {a + (b-a)k/(nz-1) : 0 <= k <= nz-1}.
        :param k: The Helmholtz equation reads (lap + k^2 / z^2)u = f.
        :param a: The domain in z direction is [a, b] with 0 < a < b.
        :param b: The domain in z direction is [a, b] with 0 < a < b.
        :param m: Number of points to use in Riemann sum calculator for FT of truncated kernel.
        :param precision: np.complex64 or np.complex128
        :param verbose: bool (if True print messages during Green's function calculation).
        :param green_func: np.ndarray of dtype precision of size nz x nz x (2n - 1) x (2n - 1) or None
        """

        print("\n\nInitializing the class")

        TypeChecker.check(x=n, expected_type=(int,))
        if n % 2 != 1 or n < 3:
            raise ValueError("n must be an odd integer >= 3")

        TypeChecker.check(x=nz, expected_type=(int,))
        if nz < 2:
            raise ValueError("n must be an integer >= 2")

        TypeChecker.check_float_positive(k)
        TypeChecker.check_float_positive(a)
        TypeChecker.check_float_strict_lower_bound(x=b, lb=a)

        TypeChecker.check_int_positive(m)

        if precision not in [np.complex64, np.complex128]:
            raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

        TypeChecker.check(x=verbose, expected_type=(bool,))

        self._n = n
        self._nz = nz
        self._k = k
        self._a = a
        self._b = b
        self._m = m
        self._precision = precision
        self._verbose = verbose

        self._cutoff = np.sqrt(2.0)

        # Run class initializer
        if green_func is None:
            self.__initialize_class()
        else:
            # TODO: add type checks for green_func (just need to check dtype and shape, else addition throws error)
            self.__initialize_class(green_func_flag=False)
            self._green_func += green_func

        self._initialized_flag = True

    def apply_kernel(self, u, output, adj=False, add=False):
        """
        :param u: 3d numpy array (must be nz x n x n dimensions with n odd).
        :param output: 3d numpy array (same dimension as u).
        :param adj: Boolean flag (forward or adjoint operator)
        :param add: Boolean flag (whether to add result to output)
        """

        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")

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

    def write_green_func(self, green_func_file):
        """
        :param green_func_file: Full filepath to save green's function to
        """
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            np.savez(file=green_func_file, green_func=self._green_func)

    @property
    def greens_func(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._green_func

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
    def m(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._m

    @property
    def precision(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._precision

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def __green_func_calc(green_func, nz, m, z, r, bessel0, k, verbose=False):

        j = complex(0, 1)
        nu = j * np.sqrt(k ** 2 - 0.25)
        cutoff = np.sqrt(2.0)
        r_scaled = cutoff * r

        if verbose:
            print("Total z slices = ", nz, "\n")
        for j1 in numba.prange(nz):

            if verbose:
                print("Starting slice number = ", j1 + 1)
            for j2 in range(j1, nz):

                if j1 != j2:
                    f1 = z[j1] * z[j2]
                    utilde = 1.0 + (0.5 / f1) * (r_scaled ** 2.0 + (z[j1] - z[j2]) ** 2.0)
                    utildesq = utilde ** 2.0
                    f2 = (utildesq - 1.0) ** 0.5
                    galpha = ((utilde + f2) ** (-1.0 * nu)) / (f2 * (f1 ** 0.5))

                    for j3 in range(m):
                        green_func[j1, j2, :, :] += bessel0[j3, :, :] * r_scaled[j3, 0, 0] * galpha[j3, 0, 0]

                    green_func[j1, j2, :, :] *= (-1.0 / (m * cutoff))
                    green_func[j2, j1, :, :] = green_func[j1, j2, :, :]

                else:
                    f1 = z[j1]
                    utilde = (2.0 / (f1 ** 2) + (r ** 2) / (f1 ** 4)) ** 0.5
                    galpha_times_r_num = (1 + (r ** 2) / (f1 ** 2) + r * utilde) ** (-1 * nu)
                    galpha_times_r = galpha_times_r_num / (f1 * utilde)

                    for j3 in range(m):
                        green_func[j1, j2, :, :] += bessel0[j3, :, :] * galpha_times_r[j3, 0, 0]

                    green_func[j1, j2, :, :] *= (-1.0 / m)
                    green_func[j2, j1, :, :] = green_func[j1, j2, :, :]

            if verbose:
                print("Finished slice number = ", j1 + 1)

    def __calculate_green_func(self):
        t1 = time.time()
        print("\nStarting Green's Function calculation ")

        kx, ky = np.meshgrid(self._kgrid_non_neg, self._kgrid_non_neg, indexing="ij")
        kabs = (kx ** 2 + ky ** 2) ** 0.5
        r = np.reshape(np.linspace(start=0.0, stop=1.0, num=self._m, endpoint=False), newshape=(self._m, 1, 1))
        r_scaled = self._cutoff * r
        bessel0 = sp.j0(r_scaled * kabs)

        self.__green_func_calc(
            green_func=self._green_func,
            nz=self._nz,
            m=self._m,
            z=self._zgrid,
            r=r,
            bessel0=bessel0,
            k=self._k,
            verbose=self._verbose
        )

        t2 = time.time()
        print("\nComputing 3d Green's Function took ", "{:6.2f}".format(t2 - t1), " s\n")
        print("\nGreen's Function size in memory (Gb) : ", "{:6.2f}".format(sys.getsizeof(self._green_func) / 1e9))
        print("\n")

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

        # Calculate FT of Truncated Green's Function and apply fftshift
        self._green_func = np.zeros(
            shape=(self._nz, self._nz, self._num_bins_non_neg, self._num_bins_non_neg),
            dtype=self._precision
        )
        if green_func_flag:
            self.__calculate_green_func()

        # Calculate integration weights
        self._mu = np.zeros(shape=(self._nz, 1, 1), dtype=np.float32) + 1.0
        self._mu[0, 0, 0] = 0.5
        self._mu[self._nz - 1, 0, 0] = 0.5


class TruncatedKernelLinearIncreasingVel2d:

    def __init__(self, n, nz, k, a, b, m, precision, verbose=False, light_mode=False):
        """
        :param n: The field will have shape n x nz, with n odd. This means that the unit cube
        [-0.5, 0.5] is gridded into n points. The points have coordinates {-0.5 + k/(n-1) : 0 <= k <= n-1}.
        :param nz: The field will have shape n x nz, with n odd. This means that the interval [a, b] is gridded
        into nz points. The points have coordinates {a + (b-a)k/(nz-1) : 0 <= k <= nz-1}.
        :param k: The Helmholtz equation reads (lap + k^2 / z^2)u = f.
        :param a: The domain in z direction is [a, b] with 0 < a < b.
        :param b: The domain in z direction is [a, b] with 0 < a < b.
        :param m: Number of points to use in Riemann sum calculator for FT of truncated kernel.
        :param precision: np.complex64 or np.complex128
        :param verbose: bool (if True print messages during Green's function calculation).
        :param light_mode: bool (if True an empty class is initialized)
        """

        TypeChecker.check(x=light_mode, expected_type=(bool,))
        self._initialized_flag = not light_mode

        if not light_mode:

            print("\n\nInitializing the class")

            TypeChecker.check(x=n, expected_type=(int,))
            if n % 2 != 1 or n < 3:
                raise ValueError("n must be an odd integer >= 3")

            TypeChecker.check(x=nz, expected_type=(int,))
            if nz < 2:
                raise ValueError("n must be an integer >= 2")

            TypeChecker.check_float_positive(k)
            TypeChecker.check_float_positive(a)
            TypeChecker.check_float_strict_lower_bound(x=b, lb=a)

            TypeChecker.check_int_positive(m)

            if precision not in [np.complex64, np.complex128]:
                raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

            TypeChecker.check(x=verbose, expected_type=(bool,))

            self._n = n
            self._nz = nz
            self._k = k
            self._a = a
            self._b = b
            self._m = m
            self._precision = precision
            self._verbose = verbose

            self._cutoff = np.sqrt(1.0)

            # Run class initializer
            self.__initialize_class()

    def set_parameters(self, n, nz, k, a, b, m, precision, verbose=False, green_func=None):
        """
        :param n: The field will have shape n x nz, with n odd. This means that the unit cube
        [-0.5, 0.5] is gridded into n points. The points have coordinates {-0.5 + k/(n-1) : 0 <= k <= n-1}.
        :param nz: The field will have shape n x nz, with n odd. This means that the interval [a, b] is gridded
        into nz points. The points have coordinates {a + (b-a)k/(nz-1) : 0 <= k <= nz-1}.
        :param k: The Helmholtz equation reads (lap + k^2 / z^2)u = f.
        :param a: The domain in z direction is [a, b] with 0 < a < b.
        :param b: The domain in z direction is [a, b] with 0 < a < b.
        :param m: Number of points to use in Riemann sum calculator for FT of truncated kernel.
        :param precision: np.complex64 or np.complex128
        :param verbose: bool (if True print messages during Green's function calculation).
        :param green_func: np.ndarray of dtype precision of size nz x nz x (2n - 1) or None
        """

        print("\n\nInitializing the class")

        TypeChecker.check(x=n, expected_type=(int,))
        if n % 2 != 1 or n < 3:
            raise ValueError("n must be an odd integer >= 3")

        TypeChecker.check(x=nz, expected_type=(int,))
        if nz < 2:
            raise ValueError("n must be an integer >= 2")

        TypeChecker.check_float_positive(k)
        TypeChecker.check_float_positive(a)
        TypeChecker.check_float_strict_lower_bound(x=b, lb=a)

        TypeChecker.check_int_positive(m)

        if precision not in [np.complex64, np.complex128]:
            raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

        TypeChecker.check(x=verbose, expected_type=(bool,))

        self._n = n
        self._nz = nz
        self._k = k
        self._a = a
        self._b = b
        self._m = m
        self._precision = precision
        self._verbose = verbose

        self._cutoff = np.sqrt(1.0)

        # Run class initializer
        if green_func is None:
            self.__initialize_class()
        else:
            # TODO: add type checks for green_func (just need to check dtype and shape, else addition throws error)
            self.__initialize_class(green_func_flag=False)
            self._green_func += green_func

        self._initialized_flag = True

    def apply_kernel(self, u, output, adj=False, add=False):
        """
        :param u: 2d numpy array (must be nz x n dimensions with n odd).
        :param output: 2d numpy array (same dimension as u).
        :param adj: Boolean flag (forward or adjoint operator)
        :param add: Boolean flag (whether to add result to output)
        """

        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")

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

    def write_green_func(self, green_func_file):
        """
        :param green_func_file: Full filepath to save green's function to
        """
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            np.savez(file=green_func_file, green_func=self._green_func)

    @property
    def greens_func(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._green_func

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
    def m(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._m

    @property
    def precision(self):
        if not self._initialized_flag:
            raise ValueError("Class initialized in light mode, cannot perform operation.")
        else:
            return self._precision

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def __green_func_calc(green_func, nz, m, z, r, cos, k, verbose=False):

        j = complex(0, 1)
        nu = j * np.sqrt(k ** 2 - 0.25)
        lamb = nu - 0.5

        if verbose:
            print("Total z slices = ", nz, "\n")
        for j1 in numba.prange(nz):

            if verbose:
                print("Starting slice number = ", j1 + 1)
            for j2 in range(j1, nz):

                f1 = z[j1] * z[j2]
                utilde = 1.0 + (0.5 / f1) * (r ** 2.0 + (z[j1] - z[j2]) ** 2.0)
                leg = SpecialFunc.legendre_q_v1(lamb, utilde, 1e-12)

                for j3 in range(m - 1):
                    green_func[j1, j2, :] += cos[j3, :] * leg[j3, 0]

                green_func[j1, j2, :] *= (-1.0 / (np.pi * m))
                green_func[j2, j1, :] = green_func[j1, j2, :]

            if verbose:
                print("Finished slice number = ", j1 + 1)

    def __calculate_green_func(self):
        t1 = time.time()
        print("\nStarting Green's Function calculation ")

        kx = np.meshgrid(self._kgrid_non_neg, indexing="ij")
        r = np.reshape(np.linspace(start=0.0, stop=1.0, num=self._m, endpoint=False), newshape=(self._m, 1))
        r = r[1:]
        cos = np.cos(r * kx)  # notice scaling is 1 compared to 3d case, so r_scaled is not needed here

        self.__green_func_calc(
            green_func=self._green_func,
            nz=self._nz,
            m=self._m,
            z=self._zgrid,
            r=r,
            cos=cos.astype(dtype=np.complex128),
            k=self._k,
            verbose=self._verbose
        )

        t2 = time.time()
        print("\nComputing 2d Green's Function took ", "{:6.2f}".format(t2 - t1), " s\n")
        print("\nGreen's Function size in memory (Mb) : ", "{:6.2f}".format(sys.getsizeof(self._green_func) / 1e6))
        print("\n")

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

        # Calculate FT of Truncated Green's Function and apply fftshift
        self._green_func = np.zeros(shape=(self._nz, self._nz, self._num_bins_non_neg), dtype=self._precision)
        if green_func_flag:
            self.__calculate_green_func()

        # Calculate integration weights
        self._mu = np.zeros(shape=(self._nz, 1), dtype=np.float32) + 1.0
        self._mu[0, 0] = 0.5
        self._mu[self._nz - 1, 0] = 0.5


if __name__ == "__main__":
    n_ = 101
    nz_ = 101
    k_ = 100.0
    a_ = 0.8
    b_ = 1.8
    m_ = 1000
    precision_ = np.complex128

    # 3d test0
    op = TruncatedKernelLinearIncreasingVel3d(
        n=n_,
        nz=nz_,
        k=k_,
        a=a_,
        b=b_,
        m=m_,
        precision=precision_
    )

    u_ = np.zeros(shape=(nz_, n_, n_), dtype=precision_)
    u_[int(nz_ / 8), int(n_ / 2), int(n_ / 2)] = 1.0
    output_ = u_ * 0

    start_t_ = time.time()
    op.apply_kernel(u=u_, output=output_)
    end_t_ = time.time()
    print("Total time to execute convolution: ", "{:4.2f}".format(end_t_ - start_t_), " s \n")

    scale = 1e-4
    fig = plt.figure()
    plt.imshow(np.real(output_[:, :, int(n_ / 2)]), cmap="Greys", vmin=-scale, vmax=scale)
    plt.grid(True)
    plt.title("Real")
    plt.colorbar()
    plt.show()

    # 2d test0
    op = TruncatedKernelLinearIncreasingVel2d(
        n=n_,
        nz=nz_,
        k=k_,
        a=a_,
        b=b_,
        m=m_,
        precision=precision_
    )

    u_ = np.zeros(shape=(nz_, n_), dtype=precision_)
    u_[int(nz_ / 8), int(n_ / 2)] = 1.0
    output_ = u_ * 0
    output1_ = u_ * 0

    ntimes = 1

    start_t_ = time.time()
    for _ in range(ntimes):
        op.apply_kernel(u=u_, output=output_)
    end_t_ = time.time()
    print("Average time to execute convolution: ", "{:4.2f}".format((end_t_ - start_t_) / ntimes), " s \n")

    scale = 1e-5
    plt.figure()
    plt.imshow(np.real(output_), cmap="Greys", vmin=-scale, vmax=scale)
    plt.grid(True)
    plt.title("Real")
    plt.colorbar()
    plt.show()

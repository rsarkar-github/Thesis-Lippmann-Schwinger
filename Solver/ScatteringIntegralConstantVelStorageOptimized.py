import numpy as np
import numba
import scipy.special as sp
import scipy.fft as scfft
import time
from ..Utilities import TypeChecker
import matplotlib.pyplot as plt


class TruncatedKernelConstantVel3d:

    def __init__(self, n, k, precision):
        """
        :param n: The field will have shape n x n x n, with n odd. This means that each dimension of the unit cube
        [-0.5, 0.5]^3 is gridded into n points. The points have coordinates {-0.5 + k/(n-1) : 0 <= k <= n-1}.
        :param k: The Helmholtz equation reads (lap + k^2)u = f.
        :param precision: np.complex64 or np.complex128
        """

        print("\n\nInitializing the class")

        TypeChecker.check(x=n, expected_type=(int,))
        if n % 2 != 1 or n < 3:
            raise ValueError("n must be an odd integer >= 3")

        TypeChecker.check_float_positive(k)

        if precision not in [np.complex64, np.complex128]:
            raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

        self._n = n
        self._k = k
        self._precision = precision

        # Run class initializer
        self.__initialize_class()

    def convolve_kernel(self, u, output, adj=False, add=False):
        """
        :param u: 3d numpy array (must be n x n x n dimensions with n odd).
        :param output: 3d numpy array (same dimension as u). Assumed to be zeros.
        :param adj: Boolean flag (forward or adjoint operator)
        :param add: Boolean flag (whether to add result to output)
        """

        # Check types of input
        if u.dtype != self._precision or output.dtype != self._precision:
            raise TypeError("Types of 'u' and 'output' must match that of class: ", self._precision)

        # Check dimensions
        if u.shape != (self._n, self._n, self._n) or output.shape != (self._n, self._n, self._n):
            raise ValueError("Shapes of 'u' and 'output' must be (n, n, n) with n = ", self._n)

        # Copy u into self._temparray
        temparray = np.zeros(shape=(self._num_bins, self._num_bins, self._num_bins), dtype=self._precision)
        temparray[
            self._start_index:(self._end_index + 1),
            self._start_index:(self._end_index + 1),
            self._start_index:(self._end_index + 1),
        ] = u

        # Do the following
        # 1. Compute Fourier transform
        # 2. Multiply with Fourier transform of Truncated Kernel Green's function
        # 3. Compute Inverse Fourier transform
        out = scfft.fftn(scfft.fftshift(temparray))
        if adj:
            out *= self._green_func_conj
        else:
            out *= self._green_func
        out = scfft.fftshift(scfft.ifftn(out))

        # Copy into output appropriately
        if not add:
            output *= 0

        output += out[
            self._start_index:(self._end_index + 1),
            self._start_index:(self._end_index + 1),
            self._start_index:(self._end_index + 1)
        ]

    @property
    def greens_func(self):
        return self._green_func

    @property
    def nkp(self):
        return self._n, self._k, self._precision

    @nkp.setter
    def nkp(self, tup):

        n = tup[0]
        k = tup[1]
        precision = tup[2]

        TypeChecker.check(x=n, expected_type=(int,))
        if n % 2 != 1 or n < 3:
            raise ValueError("n must be an odd integer >= 3")

        TypeChecker.check_float_positive(k)

        if precision not in [np.complex64, np.complex128]:
            raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

        self._n = n
        self._k = k
        self._precision = precision

        self.__initialize_class()

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        TypeChecker.check(x=n, expected_type=(int,))
        if n % 2 != 1 or n < 3:
            raise ValueError("n must be an odd integer >= 3")
        else:
            self._n = n
            self.__initialize_class()

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k):
        TypeChecker.check_float_positive(k)
        self._k = k
        self.__initialize_class()

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):

        if precision not in [np.complex64, np.complex128]:
            raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

        self._precision = precision
        self.__initialize_class()

    @staticmethod
    @numba.jit(nopython=True, parallel=True)
    def __green_func_calc(green_func, num_bins, kx, ky, kz, k, tol):

        j = complex(0, 1)
        cutoff = np.sqrt(3.0)
        f1 = k * cutoff
        f1exp = np.exp(j * f1)

        for i1 in numba.prange(num_bins):
            for i2 in range(num_bins):
                for i3 in range(num_bins):

                    kabs = np.sqrt(kx[i1, i2, i3] ** 2.0 + ky[i1, i2, i3] ** 2.0 + kz[i1, i2, i3] ** 2.0)

                    if np.abs(kabs - k) > tol:
                        f2 = kabs * cutoff
                        f3 = -1.0 + f1exp * (np.cos(f2) - np.sinc(f2 / np.pi) * j * f1)
                        green_func[i1, i2, i3] = -f3 / ((k - kabs) * (k + kabs))

                    else:
                        f2 = (j / k) * (f1 - f1exp * np.sin(f1))
                        f3 = (2 * j / (k ** 2.0)) * f1exp * (f1 * np.cos(f1) - np.sin(f1)) - (cutoff ** 2.0)
                        f3 = 0.5 * f3 * (k - kabs)
                        green_func[i1, i2, i3] = -(f2 + f3) / (k + kabs)

    def __calculate_green_func(self):

        t1 = time.time()

        tol = 1e-6
        if self._precision == np.complex64:
            tol = 1e-6
        if self._precision == np.complex128:
            tol = 1e-15

        kx, ky, kz = np.meshgrid(self._kgrid, self._kgrid, self._kgrid, indexing="ij")
        self.__green_func_calc(
            green_func=self._green_func,
            num_bins=self._num_bins,
            kx=kx,
            ky=ky,
            kz=kz,
            k=self._k,
            tol=tol
        )
        self._green_func = scfft.fftshift(self._green_func)
        self._green_func_conj = np.conjugate(self._green_func)

        t2 = time.time()
        print("Computing 3d Green's Function took ", "{:6.2f}".format(t2 - t1), " s\n")

    def __initialize_class(self):

        # Calculate number of grid points for the domain [-2, 2] along one axis,
        # and index to crop the physical domain [-0.5, 0.5]
        self._num_bins = 4 * (self._n - 1)
        self._start_index = 3 * int((self._n - 1) / 2)
        self._end_index = 5 * int((self._n - 1) / 2)

        # Calculate grid spacing d
        # Calculate grid of wavenumbers for any 1 dimension
        self._d = 1.0 / (self._n - 1)
        self._kgrid = 2 * np.pi * scfft.fftshift(scfft.fftfreq(n=self._num_bins, d=self._d))

        # Calculate FT of Truncated Green's Function and apply fftshift
        self._green_func = np.zeros(shape=(self._num_bins, self._num_bins, self._num_bins), dtype=self._precision)
        self.__calculate_green_func()


class TruncatedKernelConstantVel2d:

    def __init__(self, n, k, precision):
        """
        :param n: The field will have shape n x n, with n odd. This means that each dimension of the unit cube
        [-0.5, 0.5]^2 is gridded into n points. The points have coordinates {-0.5 + k/(n-1) : 0 <= k <= n-1}.
        :param k: The Helmholtz equation reads (lap + k^2)u = f.
        :param precision: np.complex64 or np.complex128
        """

        print("\n\nInitializing the class")

        TypeChecker.check(x=n, expected_type=(int,))
        if n % 2 != 1 or n < 3:
            raise ValueError("n must be an odd integer >= 3")

        TypeChecker.check_float_positive(k)

        if precision not in [np.complex64, np.complex128]:
            raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

        self._n = n
        self._k = k
        self._precision = precision

        # Run class initializer
        self.__initialize_class()

    def convolve_kernel(self, u, output, adj=False, add=False):
        """
        :param u: 3d numpy array (must be n x n dimensions with n odd).
        :param output: 3d numpy array (same dimension as u). Assumed to be zeros.
        :param adj: Boolean flag (forward or adjoint operator)
        :param add: Boolean flag (whether to add result to output)
        """

        # Check types of input
        if u.dtype != self._precision or output.dtype != self._precision:
            raise TypeError("Types of 'u' and 'output' must match that of class: ", self._precision)

        # Check dimensions
        if u.shape != (self._n, self._n) or output.shape != (self._n, self._n):
            raise ValueError("Shapes of 'u' and 'output' must be (n, n, n) with n = ", self._n)

        # Copy u into self._temparray
        temparray = np.zeros(shape=(self._num_bins, self._num_bins), dtype=self._precision)
        temparray[
            self._start_index:(self._end_index + 1),
            self._start_index:(self._end_index + 1)
        ] = u

        # Do the following
        # 1. Compute Fourier transform
        # 2. Multiply with Fourier transform of Truncated Kernel Green's function
        # 3. Compute Inverse Fourier transform
        out = scfft.fft2(scfft.fftshift(temparray))
        if adj:
            out *= self._green_func_conj
        else:
            out *= self._green_func
        out = scfft.fftshift(scfft.ifft2(out))

        # Copy into output appropriately
        if not add:
            output *= 0

        output += out[
            self._start_index:(self._end_index + 1),
            self._start_index:(self._end_index + 1)
        ]

    @property
    def greens_func(self):
        return self._green_func

    @property
    def nkp(self):
        return self._n, self._k, self._precision

    @nkp.setter
    def nkp(self, tup):

        n = tup[0]
        k = tup[1]
        precision = tup[2]

        TypeChecker.check(x=n, expected_type=(int,))
        if n % 2 != 1 or n < 3:
            raise ValueError("n must be an odd integer >= 3")

        TypeChecker.check_float_positive(k)

        if precision not in [np.complex64, np.complex128]:
            raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

        self._n = n
        self._k = k
        self._precision = precision

        self.__initialize_class()

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, n):
        TypeChecker.check(x=n, expected_type=(int,))
        if n % 2 != 1 or n < 3:
            raise ValueError("n must be an odd integer >= 3")
        else:
            self._n = n
            self.__initialize_class()

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k):
        TypeChecker.check_float_positive(k)
        self._k = k
        self.__initialize_class()

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):

        if precision not in [np.complex64, np.complex128]:
            raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

        self._precision = precision
        self.__initialize_class()

    @staticmethod
    def __green_func_calc(green_func, num_bins, kx, ky, k, tol):

        j = complex(0, 1)
        cutoff = np.sqrt(2.0)
        f1 = k * cutoff

        for i1 in range(num_bins):
            for i2 in range(num_bins):

                kabs = np.sqrt(kx[i1, i2] ** 2.0 + ky[i1, i2] ** 2.0)

                if np.abs(kabs - k) > tol:
                    f2 = kabs * cutoff
                    f3 = -1.0 - 0.5 * j * np.pi * (f2 * sp.jv(1, f2) *
                                                   sp.hankel1(0, f1) - f1 * sp.jv(0, f2) * sp.hankel1(1, f1))
                    green_func[i1, i2] = -f3 / ((k - kabs) * (k + kabs))

                else:
                    f2 = f1 * (sp.jv(1, f1) * sp.hankel1(1, f1) -
                               sp.jv(2, f1) * sp.hankel1(0, f1))
                    f3 = 2 * sp.jv(1, f1) * sp.hankel1(0, f1)
                    green_func[i1, i2] = -0.5 * j * np.pi * cutoff * (f2 + f3) / (k + kabs)

    def __calculate_green_func(self):

        t1 = time.time()

        tol = 1e-6
        if self._precision == np.complex64:
            tol = 1e-6
        if self._precision == np.complex128:
            tol = 1e-15

        kx, ky = np.meshgrid(self._kgrid, self._kgrid, indexing="ij")
        self.__green_func_calc(
            green_func=self._green_func,
            num_bins=self._num_bins,
            kx=kx,
            ky=ky,
            k=self._k,
            tol=tol
        )
        self._green_func = scfft.fftshift(self._green_func)
        self._green_func_conj = np.conjugate(self._green_func)

        t2 = time.time()
        print("Computing 2d Green's Function took ", "{:6.2f}".format(t2 - t1), " s\n")

    def __initialize_class(self):

        # Calculate number of grid points for the domain [-2, 2] along one axis,
        # and index to crop the physical domain [-0.5, 0.5]
        self._num_bins = 4 * (self._n - 1)
        self._start_index = 3 * int((self._n - 1) / 2)
        self._end_index = 5 * int((self._n - 1) / 2)

        # Calculate grid spacing d
        # Calculate grid of wavenumbers for any 1 dimension
        self._d = 1.0 / (self._n - 1)
        self._kgrid = 2 * np.pi * scfft.fftshift(scfft.fftfreq(n=self._num_bins, d=self._d))

        # Calculate FT of Truncated Green's Function and apply fftshift
        self._green_func = np.zeros(shape=(self._num_bins, self._num_bins), dtype=self._precision)
        self._green_func_conj = np.zeros(shape=(self._num_bins, self._num_bins), dtype=self._precision)
        self.__calculate_green_func()


if __name__ == "__main__":

    n_ = 201
    k_ = 150.0
    precision_ = np.complex64

    # op = TruncatedKernelConstantVel3d(n=n_, k=k_, precision=precision_)
    # # noinspection PyTypeChecker
    # u_ = np.zeros(shape=(n_, n_, n_), dtype=precision_)
    # u_[int(n_/2), int(n_/2), int(n_/2)] = 1.0
    # output_ = u_ * 0
    #
    # start_t_ = time.time()
    # op.convolve_kernel(u=u_, output=output_)
    # end_t_ = time.time()
    # print("Total time to execute convolution: ", "{:4.2f}".format(end_t_ - start_t_), " s \n")
    #
    # scale = 1e-6
    # plt.imshow(np.real(output_[int(n_/2), :, :]), cmap="Greys", vmin=-scale, vmax=scale)
    # plt.grid(True)
    # plt.title("Real")
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(np.imag(output_[int(n_ / 2), :, :]), cmap="Greys", vmin=-scale, vmax=scale)
    # plt.grid(True)
    # plt.title("Imag")
    # plt.colorbar()
    # plt.show()

    op = TruncatedKernelConstantVel2d(n=n_, k=k_, precision=precision_)
    # noinspection PyTypeChecker
    u_ = np.zeros(shape=(n_, n_), dtype=precision_)
    u_[int(n_ / 2), int(n_ / 2)] = 1.0
    output_ = u_ * 0

    start_t_ = time.time()
    op.convolve_kernel(u=u_, output=output_)
    end_t_ = time.time()
    print("Total time to execute convolution: ", "{:4.2f}".format(end_t_ - start_t_), " s \n")

    scale = 1e-6
    plt.imshow(np.real(output_), cmap="Greys", vmin=-scale, vmax=scale)
    plt.grid(True)
    plt.title("Real")
    plt.colorbar()
    plt.show()

    plt.imshow(np.imag(output_), cmap="Greys", vmin=-scale, vmax=scale)
    plt.grid(True)
    plt.title("Imag")
    plt.colorbar()
    plt.show()

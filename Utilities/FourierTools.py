import numpy as np
from scipy import interpolate
import scipy.fft as scfft
import numba as nb
import matplotlib.pyplot as plt
from ..Utilities import TypeChecker


def hfft_truncated_gfunc_even_2d(greens_func, smooth_cutoff, dx):
    """
    This method computes the fft of the truncated kernel.
    The even 2d case. This means that only one half (x >= 0) of the green's function
    is provided in the horizontal x direction, as it is same by symmetry.

    :param greens_func: (np.ndarray)
        2D numpy array of shape (nz x nx)
    :param smooth_cutoff: (np.ndarray)
        1D numpy array of shape (nx,)
    :param dx: (float)
        grid spacing along x. Needed to compute fft.
    """

    # Check inputs
    if len(greens_func.shape) != 2:
        raise ValueError("Green's function must be a 2D numoy array")
    nz, nx = greens_func.shape

    TypeChecker.check_ndarray(x=greens_func, shape=(nz, nx), dtypes=(np.complex64, np.complex128))

    smooth_cutoff1 = smooth_cutoff * 0
    if greens_func.dtype == np.complex64:
        TypeChecker.check_ndarray(x=smooth_cutoff, shape=(nx,), dtypes=(np.float32, np.float64))
        smooth_cutoff1 = smooth_cutoff.astype(np.float32)

    if greens_func.dtype == np.complex128:
        TypeChecker.check_ndarray(x=smooth_cutoff, shape=(nx,), dtypes=(np.float32, np.float64,))
        smooth_cutoff1 = smooth_cutoff.astype(np.float64)

    TypeChecker.check_float_positive(x=dx)

    # Multiply Green's function by smooth cutoff functions
    greens_func_truncated = greens_func * smooth_cutoff1

    # Copy the data (reflect about x = 0)
    greens_func_truncated_copy = np.zeros(shape=(nz, 2 * (nx - 1)), dtype=greens_func.dtype)
    greens_func_truncated_copy[:, nx - 1:] = greens_func_truncated[:, 0: nx - 1]
    greens_func_truncated_copy[:, 0: nx-1] = greens_func_truncated[:, nx - 1: 0: -1]

    # Compute fft
    greens_func_fft = scfft.fftn(scfft.fftshift(greens_func_truncated_copy, axes=(1,)), axes=(1,))
    greens_func_fft = scfft.fftshift(greens_func_fft, axes=(1,))

    # Keep only non-negative frequencies
    greens_func_fft = greens_func_fft[:, 0:nx]
    greens_func_fft = greens_func_fft[:, ::-1]

    # Return (after scaling for FT)
    return greens_func_fft * dx

@nb.njit
def radius_tables_for_interp_quad1(dx, nx, fac):
    """
    Get radius tables for interpolation
    """
    dx1 = dx * fac
    nx1 = int((nx - 1) / fac) + 1

    quad1_r = np.zeros(shape=(nx1, nx1), dtype=np.float32)

    for ii in range(nx1):
        for jj in range(nx1):
            quad1_r[ii, jj] = dx1 * (ii ** 2 + jj ** 2) ** 0.5

    return quad1_r.flatten()


def hfft_truncated_gfunc_radial_3d(greens_func, smooth_cutoff, dx, fac):
    """
    This method computes the fft of the truncated kernel.
    The radial 3d case. This means that only one half (x >= 0) of the green's function
    is provided in the horizontal x direction, and it is radially same by symmetry.

    :param greens_func: (np.ndarray)
        2D numpy array of shape (nz x nx)
    :param smooth_cutoff: (np.ndarray)
        1D numpy array of shape (nx,)
    :param dx: (float)
        Grid spacing along x. Needed to compute fft.
    :param fac: (int)
        Decimation factor along x. fft computed using this grid, but interpolation
        uses fine grid.
    """

    # Check inputs
    def check_inputs():
        if len(greens_func.shape) != 2:
            raise ValueError("Green's function must be a 2D numoy array")
        nz, nx = greens_func.shape

        TypeChecker.check_int_positive(x=nx)
        TypeChecker.check_int_positive(x=nz)

        TypeChecker.check_ndarray(
            x=greens_func, shape=(nz, nx), dtypes=(np.complex64, np.complex128)
        )

        smooth_cutoff1 = smooth_cutoff * 0
        datatype = greens_func.dtype
        datatype_float = np.float32

        if datatype == np.complex64:
            TypeChecker.check_ndarray(
                x=smooth_cutoff, shape=(nx,), dtypes=(np.float32, np.float64)
            )
            smooth_cutoff1 = smooth_cutoff.astype(np.float32)
            datatype_float = np.float32

        if datatype == np.complex128:
            TypeChecker.check_ndarray(
                x=smooth_cutoff, shape=(nx,), dtypes=(np.float32, np.float64,)
            )
            smooth_cutoff1 = smooth_cutoff.astype(np.float64)
            datatype_float = np.float64

        TypeChecker.check_float_positive(x=dx)
        TypeChecker.check_int_positive(x=fac)

        if (nx - 1) % fac != 0:
            raise ValueError(
                "The number of grid points along x ""direction should satisfy: "
                "(nx - 1) % fac == 0."
            )

        return nz, nx, smooth_cutoff1, datatype, datatype_float

    nz, nx, smooth_cutoff1, datatype, datatype_float = check_inputs()

    # Multiply Green's function by smooth cutoff functions
    greens_func_truncated = greens_func * smooth_cutoff1

    # Interpolate first quadrant
    def interp_quad1():
        """
        Interpolate truncated Green's function to first quadrant
        """
        nx1 = int((nx - 1) / fac) + 1
        greens_func_truncated_interp = np.zeros(shape=(nz, nx1, nx1), dtype=datatype)

        r_input = np.linspace(start=0, stop=dx * (nx - 1), num=nx, endpoint=True)
        r_interp = radius_tables_for_interp_quad1(dx=dx, nx=nx, fac=fac)

        for ii in range(nz):

            func_interp = interpolate.interp1d(
                x=r_input,
                y=np.real(greens_func_truncated[ii, :].flatten()),
                kind="linear",
                bounds_error=False,
                fill_value=0
            )
            greens_func_truncated_interp_quad1_real = func_interp(r_interp).astype(datatype)

            func_interp = interpolate.interp1d(
                x=r_input,
                y=np.imag(greens_func_truncated[ii, :].flatten()),
                kind="linear",
                bounds_error=False,
                fill_value=0
            )
            greens_func_truncated_interp_quad1_imag = ((0 + 1j) * func_interp(r_interp)).astype(datatype)

            greens_func_truncated_interp[ii, :, :] = np.reshape(
                greens_func_truncated_interp_quad1_real + greens_func_truncated_interp_quad1_imag,
                newshape=(1, nx1, nx1)
            )

        return greens_func_truncated_interp, nx1

    greens_func_truncated_interp, nx1 = interp_quad1()

    # Copy first quadrant to create full Green's function
    greens_func_truncated_interp_full = np.zeros(
        shape=(nz, 2 * (nx1 - 1), 2 * (nx1 - 1)),
        dtype=datatype
    )

    greens_func_truncated_interp_full[:, nx1:, nx1:] = greens_func_truncated_interp[
        :, 1: nx1 - 1, 1: nx1 - 1
    ]

    greens_func_truncated_interp_full[:, 0:nx1, 0:nx1] = greens_func_truncated_interp[
        :, 0:nx1, 0:nx1
    ][:, ::-1, ::-1]

    greens_func_truncated_interp_full[:, nx1:, 0:nx1] = greens_func_truncated_interp[
        :, 1: nx1 - 1, 0:nx1
    ][:, :, ::-1]

    greens_func_truncated_interp_full[:, 0:nx1, nx1:] = greens_func_truncated_interp[
        :, 0:nx1, 1: nx1 - 1
    ][:, ::-1, :]

    # Fourier transform the Green's function
    greens_func_fft = scfft.fftn(
        scfft.fftshift(greens_func_truncated_interp_full, axes=(1, 2)),
        axes=(1, 2)
    )
    greens_func_fft = scfft.fftshift(greens_func_fft, axes=(1, 2))

    # Extract the first quadrant of Green's function and return
    greens_func_fft = greens_func_fft[:, 0:nx1, 0:nx1]
    greens_func_fft = greens_func_fft[:, ::-1, ::-1]

    return greens_func_fft * ((dx * fac) ** 2)


if __name__ == "__main__":

    # -----------------------------------------------------------
    # Test 2d case

    nx_ = 101
    nz_ = 2
    dx_ = 0.1

    a_ = np.zeros(shape=(nz_, nx_), dtype=np.float32)
    xv_ = np.linspace(start=0., stop=10., num=nx_, endpoint=True)
    a_ += np.exp((-1.0) * xv_ ** 2 / 2)

    smooth_cutoff_ = a_[0, :] * 0 + 1.0

    output_ = hfft_truncated_gfunc_even_2d(
        greens_func=a_.astype(np.complex64),
        smooth_cutoff=smooth_cutoff_.astype(np.float32),
        dx=dx_
    )

    fft_freq_ = 2 * np.pi * scfft.rfftfreq(n=2 * (nx_ - 1), d=dx_)
    output1_ = np.sqrt(2.0 * np.pi) * np.exp((-1.0) * (fft_freq_ ** 2) / 2)

    print("Norm of computed & true solution = ", np.linalg.norm(output1_ - np.real(output_[0, :])))

    # -----------------------------------------------------------
    # Test 3d case

    nx_ = 101
    nz_ = 2
    dx_ = 0.1
    fac_ = 4

    a_ = np.zeros(shape=(nz_, nx_), dtype=np.float32)
    xv_ = np.linspace(start=0., stop=10., num=nx_, endpoint=True)
    a_ += np.exp((-1.0) * xv_ ** 2 / 2)

    smooth_cutoff_ = a_[0, :] * 0 + 1.0

    output_ = hfft_truncated_gfunc_radial_3d(
        greens_func=a_.astype(np.complex64),
        smooth_cutoff=smooth_cutoff_.astype(np.float32),
        dx=dx_,
        fac=fac_
    )

    plt.imshow(np.real(output_[0, :, :]), cmap="Greys", vmin=0, vmax=10.0)
    plt.show()

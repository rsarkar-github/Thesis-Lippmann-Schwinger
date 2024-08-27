import numpy as np
from scipy import interpolate
from scipy.sparse.linalg import LinearOperator, gmres
from matplotlib import pyplot as plt
from ...Solver.HelmholtzOperators import create_helmholtz2d_matrix_even
from ...Utilities import TypeChecker


def make_velocity_from_trace(vel_trace_, n1_, n2_):
    """
    :param vel_trace_: velocity values as a 2D numpy array of shape [N, 1], assumed dtype = np.float32
    :param n1_: points along x1 direction
    :param n2_: points along x2 direction

    The trace is first interpolated to a grid with n1 points in x1 direction (same vertical extent).
    Then it is copied n2 times in x2 direction.
    Result is returned as a numpy array or shape (n1_, n2_).
    """

    n1_in_, _ = vel_trace_.shape

    x1_start_ = 0.0
    x1_end_ = n1_in_

    coord_in_ = np.linspace(start=x1_start_, stop=x1_end_, num=n1_in_, endpoint=True)
    val_in_ = np.reshape(vel_trace_, newshape=(n1_in_,))

    f = interpolate.interp1d(coord_in_, val_in_, kind="linear")

    coord_out_ = np.linspace(start=x1_start_, stop=x1_end_, num=n1_, endpoint=True)
    val_out_ = np.reshape(f(coord_out_), newshape=(n1_, 1))

    vel_ = np.zeros(shape=(n1_, n2_), dtype=np.float32)

    for i in range(n1_):
        vel_[i, :] = val_out_[i, 0]

    return vel_


def make_grid_params(a1_, a2_, delta_, lambda_min_, lambda_max_):
    """
    :param a1_: original domain is [0, a1_] x [0, a2_]
    :param a2_: original domain is [0, a1_] x [0, a2_]
    :param delta_: grid spacing along both directions
    :param lambda_min_: minimum wavelength to support
    :param lambda_max_: maximum wavelength to support

    :return: (a1_pad_, a2_pad_, pad1_cells_, pad2_cells_)

    Output new grid by adding needed pml cells, for the even Helmholtz solver test.
    """

    TypeChecker.check_float_positive(delta_)
    TypeChecker.check_float_upper_bound(x=delta_, ub=lambda_min_ / 10.0)

    pad_cells_ = int(lambda_max_ / delta_) + 1

    a1_pad_ = a1_ + 2 * pad_cells_ * delta_
    a2_pad_ = a2_ + pad_cells_ * delta_

    return a1_pad_, a2_pad_, pad_cells_, pad_cells_


def extend_vel_trace_1d(vel_trace_, pad_cells_):
    """
    :param vel_trace_: velocity values as a 2D numpy array of shape [N, 1], assumed dtype = np.float32
    :param pad_cells_: number of cells to pad along each end
    """

    n1_in_, _ = vel_trace_.shape
    n1_out_ = n1_in_ + 2 * pad_cells_

    vel_trace_out_ = np.zeros(shape=(n1_out_, 1), dtype=np.float32)
    vel_trace_out_[pad_cells_: pad_cells_ + n1_in_, 0] = vel_trace_.flatten()
    vel_trace_out_[0: pad_cells_, 0] = vel_trace_out_[pad_cells_, 0]
    vel_trace_out_[pad_cells_ + n1_in_: n1_out_, 0] = vel_trace_out_[pad_cells_ + n1_in_ - 1, 0]

    return vel_trace_out_


if __name__ == "__main__":

    # Define velocity trace
    n1_vel_trace = 100
    vel_trace = np.zeros(shape=(n1_vel_trace, 1), dtype=np.float32)
    vel_trace += 2.0   # in km/s

    # Define frequency, calculate min & max wavelength
    freq = 15.0
    omega = freq * 2 * np.pi
    precision = np.complex128

    vmin = np.min(vel_trace)
    vmax = np.max(vel_trace)
    lambda_min = vmin / freq
    lambda_max = vmax / freq

    # Set grid extent, & calculate minimum grid spacing
    delta_base = lambda_min / 10.0
    a1 = 100 * delta_base  # in km (should not change)
    a2 = 100 * delta_base  # in km (should not change)

    # Scale a2 (different experiments)
    fac = [1, 2]

    # Function for creating all objects for the experiment
    def create_objects_for_experiment(factor):
        """
        :param factor: factor by which to refine the computational grid
        :return: New grid parameters, velocity
        """

        # New grid spacing
        delta = delta_base

        # Get new extents
        a1_pad, a2_pad, pad1_cells, pad2_cells = make_grid_params(
            a1_=a1,
            a2_=a2 * factor,
            delta_=delta,
            lambda_min_=lambda_min,
            lambda_max_=lambda_max
        )

        # Calculate grid points in each direction
        n1 = int(np.round(a1_pad / delta) + 1)
        n2 = int(np.round(a2_pad / delta) + 1)
        n1_vel = int(np.round(a1 / delta) + 1)

        # Modify vel_trace
        vel_trace_mod = make_velocity_from_trace(vel_trace_=vel_trace, n1_=n1_vel, n2_=1)
        vel_trace_mod = extend_vel_trace_1d(vel_trace_=vel_trace_mod, pad_cells_=pad1_cells)
        vel = make_velocity_from_trace(vel_trace_=vel_trace_mod, n1_=n1, n2_=n2)

        # Create source centered at original grid (x2 = 0, x1 = a1 / 2, gaussian std = delta)
        x1 = np.linspace(start=0, stop=a1_pad, num=n1, endpoint=True)
        x2 = np.linspace(start=0, stop=a2_pad, num=n2, endpoint=True)
        x1v, x2v = np.meshgrid(x1, x2)
        sigma = delta_base
        source = np.exp((-1) * ((x1v - a1_pad / 2) ** 2 + x2v ** 2) / (2 * sigma * sigma))
        source = source.T

        return a1_pad, a2_pad, pad1_cells, pad2_cells, vel, source

    for i, item in enumerate(fac):

        a1_full, a2_full, pad1, pad2, vel_array, src = create_objects_for_experiment(factor=item)

        scale = 1e-1
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(np.real(src), cmap="Greys", vmin=-scale, vmax=scale)
        plt.show()

        mat = create_helmholtz2d_matrix_even(
            a1=a1_full,
            a2=a2_full,
            pad1=pad1,
            pad2=pad2,
            omega=omega,
            precision=precision,
            vel=vel_array,
            pml_damping=50.0,
            adj=False,
            warnings=True
        )

        mat_t = create_helmholtz2d_matrix_even(
            a1=a1_full,
            a2=a2_full,
            pad1=pad1,
            pad2=pad2,
            omega=omega,
            precision=precision,
            vel=vel_array,
            pml_damping=20.0,
            adj=True,
            warnings=True
        )

        n1, n2 = vel_array.shape
        print("n1 = ", n1, "n2 = ", n2)

        def forward_op(v):
            mat.dot(v)

        def adjoint_op(v):
            mat_t.dot(v)

        # Define linear operator
        linop = LinearOperator(
            shape=(n1 * n2, n1 * n2),
            matvec=forward_op,
            rmatvec=adjoint_op,
            dtype=precision
        )
        # Callback generator
        def make_callback():
            closure_variables = dict(counter=0, residuals=[])

            def callback(residuals):
                closure_variables["counter"] += 1
                closure_variables["residuals"].append(residuals)
                print(closure_variables["counter"], residuals)

            return callback

        tol = 1e-3
        sol, exitcode = gmres(
            mat,
            np.reshape(src, newshape=(n1 * n2, 1)),
            maxiter=5000,
            restart=5000,
            atol=0,
            tol=tol,
            callback=make_callback()
        )
        sol = np.reshape(sol, newshape=(n1, n2))
        print(exitcode)

        scale = 1e-4
        fig, ax = plt.subplots(1, 1)
        im = ax.imshow(np.real(sol), cmap="Greys", vmin=-scale, vmax=scale)
        plt.show()

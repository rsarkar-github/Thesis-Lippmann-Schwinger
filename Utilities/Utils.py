import numpy as np
import numba
from scipy import interpolate


@numba.njit
def ricker_time(freq_peak: float = 10.0, nt: int = 250, dt: float = 0.004, delay: float = 0.05):
    """
    :param freq_peak: (float) Peak frequency of Ricker wavelet (in Hertz)
    :param nt: (int) Number of time samples
    :param dt: (float) timestep (in s)
    :param delay: (float) delay (in s)
    :return:
        t: time samples as a 1D numpy array of size (nt,),
        y: Ricker wavelet series as a 1D numpy array of size (nt,)
    """
    t = np.arange(0.0, nt * dt, dt, dtype=np.float32)
    y = (1.0 - 2.0 * ((np.pi * freq_peak * (t - delay)) ** 2)) * np.exp(-(np.pi * freq_peak * (t - delay)) ** 2)
    return t, y


@numba.njit
def cosine_taper_2d(array2d, ncells_pad_x: int, ncells_pad_z: int, get_mask_only: bool = False):
    """
    If get_mask_only is False, apply cosine taper to array2d.
    To get the mask, set get_mask_only = True.

    :param array2d: (ndarray) 2D numpy array of dtype float32
    :param ncells_pad_x: (int) Number of padding cells in x direction
    :param ncells_pad_z: (int) Number of padding cells in z direction
    :param get_mask_only: (bool)
    :return:
        t: 2D numpy array after application of taper filter (if get_mask_only is True) of same shape as array2d,
    """

    # Get grid information
    grid_points_z, grid_points_x = array2d.shape
    grid_cells_x = grid_points_x - 1
    grid_cells_z = grid_points_z - 1

    if grid_cells_x < 2 * ncells_pad_x or grid_cells_z < 2 * ncells_pad_z:
        raise ValueError

    # Create a filter array
    filter_x = np.zeros((1, grid_points_x), dtype=np.float32) + 1.0
    filter_z = np.zeros((grid_points_z, 1), dtype=np.float32) + 1.0

    # Update filter array to be sine square taper
    if ncells_pad_x > 0:
        t = np.zeros((1, ncells_pad_x), dtype=np.float32)
        for i in range(0, ncells_pad_x):
            t[0, i] = i
        t = np.sin(t * (np.pi * 0.5 / ncells_pad_x)) ** 2.0
        filter_x[0, 0:ncells_pad_x] = t[0, :]
        filter_x[0, grid_cells_x - ncells_pad_x + 1:grid_points_x] = t[0, ::-1]

    if ncells_pad_z > 0:
        t = np.zeros((ncells_pad_z, 1), dtype=np.float32)
        for i in range(0, ncells_pad_z):
            t[i, 0] = i
        t = np.sin(t * (np.pi * 0.5 / ncells_pad_z)) ** 2.0
        filter_z[0:ncells_pad_z, 0] = t[:, 0]
        filter_z[grid_cells_z - ncells_pad_z + 1:grid_points_z, 0] = t[::-1, 0]

    if not get_mask_only:
        # Apply filter
        array2d *= filter_x
        array2d *= filter_z
        return

    else:
        # Get filter
        t = np.zeros((grid_points_z, grid_points_x), dtype=np.float32) + 1.0
        t *= filter_x
        t *= filter_z
        return t


@numba.njit
def boxcar_taper_2d(array2d, ncells_pad_x: int, ncells_pad_z: int, get_mask_only: bool = False):
    """
    If get_mask_only is False, apply boxcar taper to array2d.
    To get the mask, set get_mask_only = True.

    :param array2d: (ndarray) 2D numpy array of dtype float32
    :param ncells_pad_x: (int) Number of padding cells in x direction
    :param ncells_pad_z: (int) Number of padding cells in z direction
    :param get_mask_only: (bool)
    :return:
        t: 2D numpy array after application of filter (if get_mask_only is True) of same shape as array2d,
    """

    # Get grid information
    grid_points_z, grid_points_x = array2d.shape
    grid_cells_x = grid_points_x - 1
    grid_cells_z = grid_points_z - 1

    if grid_cells_x < 2 * ncells_pad_x or grid_cells_z < 2 * ncells_pad_z:
        raise ValueError

    # Create a filter array
    filter_x = np.zeros((1, grid_points_x), dtype=np.float32) + 1.0
    filter_z = np.zeros((grid_points_z, 1), dtype=np.float32) + 1.0

    # Update filter array to be sine square taper
    if ncells_pad_x > 0:
        t = np.zeros((1, ncells_pad_x), dtype=np.float32)
        filter_x[0, 0:ncells_pad_x] = t[0, :]
        filter_x[0, grid_cells_x - ncells_pad_x + 1:grid_points_x] = t[0, ::-1]

    if ncells_pad_z > 0:
        t = np.zeros((ncells_pad_z, 1), dtype=np.float32)
        filter_z[0:ncells_pad_z, 0] = t[:, 0]
        filter_z[grid_cells_z - ncells_pad_z + 1:grid_points_z, 0] = t[::-1, 0]

    if not get_mask_only:
        # Apply filter
        array2d *= filter_x
        array2d *= filter_z
        return

    else:
        # Get filter
        t = np.zeros((grid_points_z, grid_points_x), dtype=np.float32) + 1.0
        t *= filter_x
        t *= filter_z
        return t


@numba.njit
def extrapolate_same(array2d,  ncells_pad_x: int, ncells_pad_z: int, create_new: bool = True):
    """
    Extrapolate in the padding zone based on value of last layer. If create_new = False, then modify input array2d.

    :param array2d: (ndarray) 2D numpy array of dtype float32
    :param ncells_pad_x: (int) Number of padding cells in x direction
    :param ncells_pad_z: (int) Number of padding cells in z direction
    :param create_new: (bool) If True then create a new output array

    :return:
        array2d_out: (ndarray) of same shape as array2d after extrapolation (if create_new is True)
    """

    # Get grid information
    grid_points_z, grid_points_x = array2d.shape
    grid_cells_x = grid_points_x - 1
    grid_cells_z = grid_points_z - 1

    if grid_cells_x < 2 * ncells_pad_x or grid_cells_z < 2 * ncells_pad_z:
        raise ValueError

    # Get the indexes
    index_start_x = ncells_pad_x
    index_end_x = grid_cells_x - ncells_pad_x
    index_start_z = ncells_pad_z
    index_end_z = grid_cells_z - ncells_pad_z

    if create_new:
        # Create output array and copy interior part of array except padding
        array2d_out = np.zeros((grid_points_z, grid_points_x), dtype=np.float32)
        array2d_out[index_start_z: index_end_z+1, index_start_x: index_end_x+1] = \
            array2d[index_start_z: index_end_z+1, index_start_x: index_end_x+1]

        # Extrapolate left and right
        for i1 in range(index_start_z, index_end_z+1):
            for j1 in range(index_start_x):
                array2d_out[i1, j1] = array2d_out[i1, index_start_x]
            for j1 in range(index_end_x+1, grid_points_x):
                array2d_out[i1, j1] = array2d_out[i1, index_start_x]

        # Extrapolate top and bottom
        for j1 in range(grid_points_x):
            for i1 in range(index_start_z):
                array2d_out[i1, j1] = array2d_out[index_start_z, j1]
            for i1 in range(index_end_z+1, grid_points_z):
                array2d_out[i1, j1] = array2d_out[index_end_z, j1]

        return array2d_out

    else:
        # Extrapolate left and right
        for i1 in range(index_start_z, index_end_z + 1):
            for j1 in range(index_start_x):
                array2d[i1, j1] = array2d[i1, index_start_x]
            for j1 in range(index_end_x + 1, grid_points_x):
                array2d[i1, j1] = array2d[i1, index_start_x]

        # Extrapolate top and bottom
        for j1 in range(grid_points_x):
            for i1 in range(index_start_z):
                array2d[i1, j1] = array2d[index_start_z, j1]
            for i1 in range(index_end_z + 1, grid_points_z):
                array2d[i1, j1] = array2d[index_end_z, j1]


@numba.njit
def laplacian(array2d_in, array2d_out, dx: float, dz: float, order: int = 10):
    """
    Apply 2D tenth order laplacian to array2d_in and put results in array2d_out.

    :param array2d_in: (ndarray) 2D numpy array of dtype float32
    :param array2d_out: (ndarray) 2D numpy array of dtype float32 of same shape as array2d_in
    :param dx: (float) grid spacing in x
    :param dz: (float) grid spacing in z
    :param order: (int) order of laplacians
    :return:
    """

    if array2d_in.shape != array2d_out.shape:
        raise ValueError

    if order not in [10]:
        raise NotImplementedError

    # Get grid information
    grid_points_z, grid_points_x = array2d_in.shape

    if order == 10:
        # Compute 2d stencil
        stencil = np.zeros((6,), dtype=np.float32)
        stencil[0] = -2.92722
        stencil[1] = 1.66667
        stencil[2] = -0.238095
        stencil[3] = 0.0396825
        stencil[4] = -0.00496031
        stencil[5] = 0.00031746

        # Compute some variables
        ncells_pad_z = 5
        ncells_pad_x = 5

        index_start_x = ncells_pad_x
        index_end_x = grid_points_x + 9 - ncells_pad_x
        index_start_z = ncells_pad_z
        index_end_z = grid_points_z + 9 - ncells_pad_z

        fx = 1.0 / (dx ** 2.0)
        fz = 1.0 / (dz ** 2.0)

        # Create temporary array and copy interior part of array_in except padding, and extrapolate in padding zone
        array2d = np.zeros((grid_points_z + 2 * ncells_pad_z, grid_points_x + 2 * ncells_pad_x), dtype=np.float32)
        array2d[index_start_z: index_end_z + 1, index_start_x: index_end_x + 1] = array2d_in
        _ = extrapolate_same(array2d=array2d, ncells_pad_x=ncells_pad_x, ncells_pad_z=ncells_pad_z, create_new=False)

        # Compute Laplacian
        for i1 in range(grid_points_z):
            i = i1 + ncells_pad_z

            for j1 in range(grid_points_x):
                j = j1 + ncells_pad_x

                t = array2d[i, j] * stencil[0] + \
                    (array2d[i + 1, j] + array2d[i - 1, j]) * stencil[1] + \
                    (array2d[i + 2, j] + array2d[i - 2, j]) * stencil[2] + \
                    (array2d[i + 3, j] + array2d[i - 3, j]) * stencil[3] + \
                    (array2d[i + 4, j] + array2d[i - 4, j]) * stencil[4] + \
                    (array2d[i + 5, j] + array2d[i - 5, j]) * stencil[5]

                array2d_out[i1, j1] = t * fz

                t = array2d[i, j] * stencil[0] + \
                    (array2d[i, j + 1] + array2d[i, j - 1]) * stencil[1] + \
                    (array2d[i, j + 2] + array2d[i, j - 2]) * stencil[2] + \
                    (array2d[i, j + 3] + array2d[i, j - 3]) * stencil[3] + \
                    (array2d[i, j + 4] + array2d[i, j - 4]) * stencil[4] + \
                    (array2d[i, j + 5] + array2d[i, j - 5]) * stencil[5]

                array2d_out[i1, j1] += t * fx


def make_velocity_from_trace(vel_trace_, n1_, n2_):
    """
    :param vel_trace_: velocity values as a 2D numpy array of shape [N, 1], assumed dtype = np.float32
    :param n1_: points along x1 direction
    :param n2_: points along x2 direction

    The trace is first interpolated to a grid with n1_ points in x1 direction (same vertical extent).
    Then it is copied n2_ times in x2 direction.
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

    for ii in range(n1_):
        vel_[ii, :] = val_out_[ii, 0]

    return vel_


def make_velocity3d_from_trace(vel_trace_, n1_, n2_, n3_):
    """
        :param vel_trace_: velocity values as a 2D numpy array of shape [N, 1], assumed dtype = np.float32
        :param n1_: points along x1 direction
        :param n2_: points along x2 direction
        :param n3_: points along x3 direction

        The trace is first interpolated to a grid with n1_ points in x1 direction (same vertical extent).
        Then it is copied n2_ times in x2 direction.
        Then it is copied n3_ times in x3 direction.
        Result is returned as a numpy array or shape (n1_, n2_, n3_).
        """

    n1_in_, _ = vel_trace_.shape

    x1_start_ = 0.0
    x1_end_ = n1_in_

    coord_in_ = np.linspace(start=x1_start_, stop=x1_end_, num=n1_in_, endpoint=True)
    val_in_ = np.reshape(vel_trace_, newshape=(n1_in_,))

    f = interpolate.interp1d(coord_in_, val_in_, kind="linear")

    coord_out_ = np.linspace(start=x1_start_, stop=x1_end_, num=n1_, endpoint=True)
    val_out_ = np.reshape(f(coord_out_), newshape=(n1_, 1))

    vel_ = np.zeros(shape=(n1_, n2_, n3_), dtype=np.float32)

    for ii in range(n1_):
        vel_[ii, :, :] = val_out_[ii, 0]

    return vel_

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

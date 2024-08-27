from scipy.sparse import csc_matrix
import numpy as np
from tqdm import tqdm
from ..Utilities import TypeChecker


def create_helmholtz2d_matrix(
        a1,
        a2,
        pad1,
        pad2,
        omega,
        precision,
        vel,
        pml_damping=1.0,
        adj=False,
        warnings=True
):
    """
    :param a1: The domain on which Helmholtz equation is solved is [0,a1] x [0,a2].
    :param a2: The domain on which Helmholtz equation is solved is [0,a1] x [0,a2].
    :param pad1: Number of pad cells along x1 direction.
    :param pad2: Number of pad cells along x2 direction.
    :param omega:  Angular frequency. The Helmholtz equation reads (lap + omega^2 / vel^2)u = f.
    :param precision: np.complex64 or np.complex128
    :param vel: 2d numpy array real valued of shape n1 x n2. n1, n2 >= 4 required.
    :param pml_damping: Positive float. Damping parameter for pml.
    :param adj: Boolean flag (whether to compute the adjoint)
    :param warnings: Boolean flag (whether to print warnings due to grid param check)

    :return: Sparse Helmholtz matrix of shape (n1 * n2) x (n1 * n2), with grid points
    enumerated row wise, i.e. x2 followed by x1 directions.

    *
    *
    *
    x1
    *
    *
    * * * x2 * * *

    Note: Dirichlet boundary conditions imposed on boundary layer of nodes, by adding an extra layer of nodes.
    """

    # Check vel for type, make copy consistent with type of precision, and check if values > 0
    if precision not in [np.complex64, np.complex128]:
        raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

    TypeChecker.check(x=vel, expected_type=(np.ndarray,))

    n1, n2 = vel.shape
    TypeChecker.check_int_lower_bound(x=n1, lb=4)
    TypeChecker.check_int_lower_bound(x=n2, lb=4)

    if vel.shape != (n1, n2):
        raise ValueError("Shape of 'vel' must match (" + str(n1) + ", " + str(n2) + ")")
    if vel.dtype not in (np.float32, np.float64):
        raise TypeError("dtype of 'vel' must be np.float32 or np.float64")

    if np.any(vel <= 0.0):
        raise ValueError("Non-positive velocities detected")

    # Check inputs
    TypeChecker.check_float_positive(x=a1)
    TypeChecker.check_float_positive(x=a2)
    TypeChecker.check_int_bounds(x=pad1, lb=2, ub=int(n1 / 2))
    TypeChecker.check_int_bounds(x=pad2, lb=2, ub=int(n2 / 2))
    TypeChecker.check_float_positive(x=omega)
    TypeChecker.check_float_positive(x=pml_damping)
    TypeChecker.check(x=adj, expected_type=(bool,))
    TypeChecker.check(x=warnings, expected_type=(bool,))

    # Print grid size warnings
    d1 = a1 / (n1 - 1)
    d2 = a2 / (n2 - 1)
    vel_max = np.max(vel)
    vel_min = np.min(vel)
    lambda_min = 2 * np.pi * vel_min / omega
    lambda_max = 2 * np.pi * vel_max / omega
    pad_cells1 = pad1
    pad_cells2 = pad2
    pml_width1 = d1 * pad_cells1
    pml_width2 = d2 * pad_cells2
    dmin = lambda_min / 10.0  # 10 points per minimum wavelength

    if warnings:
        print("\n\n")

        if d1 > dmin:
            print("Warning: Required dmin = ", "{:.2e}".format(dmin), ", Computed d1 = ", "{:.2e}".format(d1))

        if d2 > dmin:
            print("Warning: Required dmin = ", "{:.2e}".format(dmin), ", Computed d2 = ", "{:.2e}".format(d2))

        if pml_width1 < lambda_max:
            print("Warning: Required minimum pml width = ", "{:.2e}".format(lambda_max),
                  ", Computed pml_width1 = ", "{:.2e}".format(pml_width1))

        if pml_width2 < lambda_max:
            print("Warning: Required minimum pml width = ", "{:.2e}".format(lambda_max),
                  ", Computed pml_width2 = ", "{:.2e}".format(pml_width2))

    # Pad velocity by 1 layer with correct precision type
    vel1 = np.zeros((n1 + 2, n2 + 2))
    if precision is np.complex64:
        vel1 = vel1.astype(np.float32)
        vel1[1: n1 + 1, 1: n2 + 1] = vel.astype(np.float32)
    if precision is np.complex128:
        vel1 = vel1.astype(np.float64)
        vel1[1: n1 + 1, 1: n2 + 1] = vel.astype(np.float64)

    vel1[1: n1 + 1, 0] = vel1[1: n1 + 1, 1]
    vel1[1: n1 + 1, n2 + 1] = vel1[1: n1 + 1, n2]
    vel1[0, :] = vel1[1, :]
    vel1[n1 + 1, :] = vel1[n1, :]

    # Number of nodes including DBC layer
    n1_ = n1 + 2
    n2_ = n2 + 2
    a1_ = a1 + 2 * d1
    a2_ = a2 + 2 * d2

    # Calculate s1 and s2 arrays
    def s1_array():

        d1_ = d1 / 2.0
        s1_ = np.zeros(shape=(2 * n1_ - 1,), dtype=precision)

        for kk in range(2 * pad_cells1 + 1):
            s1_[kk] = (1.0 - kk * d1_ / pml_width1) ** 2

        for kk in range(2 * (n1_ - 1 - pad_cells1), 2 * n1_ - 1):
            s1_[kk] = (1.0 - (a1_ - kk * d1_) / pml_width1) ** 2

        s1_ *= (complex(0, 1) / omega) * (pml_damping / pml_width1)
        s1_ += 1.0
        return 1.0 / s1_

    def s2_array():

        d2_ = d2 / 2.0
        s2_ = np.zeros(shape=(2 * n2_ - 1,), dtype=precision)

        for kk in range(2 * pad_cells2 + 1):
            s2_[kk] = (1.0 - kk * d2_ / pml_width2) ** 2

        for kk in range(2 * (n2_ - 1 - pad_cells2), 2 * n2_ - 1):
            s2_[kk] = (1.0 - (a2_ - kk * d2_) / pml_width2) ** 2

        s2_ *= (complex(0, 1) / omega) * (pml_damping / pml_width2)
        s2_ += 1.0
        return 1.0 / s2_

    s1 = s1_array()
    s2 = s2_array()

    # Create lists to hold matrix entries
    data = []
    rows = []
    cols = []

    ####################################################################################################
    # Loop over interior nodes except edges
    def interior_nodes():

        for i1 in range(2, n1):
            count1 = (i1 - 1) * n2
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            for i2 in range(2, n2):
                count2 = count1 + i2 - 1
                p1x = s2[2 * i2] / d2
                p2x = s2[2 * i2 + 1] / d2
                p3x = s2[2 * i2 - 1] / d2

                rows.append(count2)
                cols.append(count2)
                data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[i1, i2]) ** 2)

                rows.append(count2)
                cols.append(count2 + 1)
                data.append(p1x * p2x)

                rows.append(count2)
                cols.append(count2 - 1)
                data.append(p1x * p3x)

                rows.append(count2)
                cols.append(count2 + n2)
                data.append(p1z * p2z)

                rows.append(count2)
                cols.append(count2 - n2)
                data.append(p1z * p3z)

    interior_nodes()

    ####################################################################################################
    # Edges except corners
    def edge_nodes():

        # 1. Bottom
        count1 = 0
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1

        for i2 in range(2, n2):
            count2 = count1 + i2 - 1
            p1x = s2[2 * i2] / d2
            p2x = s2[2 * i2 + 1] / d2
            p3x = s2[2 * i2 - 1] / d2

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[1, i2]) ** 2)

            rows.append(count2)
            cols.append(count2 + 1)
            data.append(p1x * p2x)

            rows.append(count2)
            cols.append(count2 - 1)
            data.append(p1x * p3x)

            rows.append(count2)
            cols.append(count2 + n2)
            data.append(p1z * p2z)

        # 2. Top
        count1 = (n1 - 1) * n2
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1

        for i2 in range(2, n2):
            count2 = count1 + i2 - 1
            p1x = s2[2 * i2] / d2
            p2x = s2[2 * i2 + 1] / d2
            p3x = s2[2 * i2 - 1] / d2

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[n1, i2]) ** 2)

            rows.append(count2)
            cols.append(count2 + 1)
            data.append(p1x * p2x)

            rows.append(count2)
            cols.append(count2 - 1)
            data.append(p1x * p3x)

            rows.append(count2)
            cols.append(count2 - n2)
            data.append(p1z * p3z)

        # 3. Left
        count1 = 0
        p1x = s2[2] / d2
        p2x = s2[3] / d2
        p3x = s2[1] / d2

        for i1 in range(2, n1):
            count2 = count1 + (i1 - 1) * n2
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[i1, 1]) ** 2)

            rows.append(count2)
            cols.append(count2 + 1)
            data.append(p1x * p2x)

            rows.append(count2)
            cols.append(count2 + n2)
            data.append(p1z * p2z)

            rows.append(count2)
            cols.append(count2 - n2)
            data.append(p1z * p3z)

        # 4. Right
        count1 = n2 - 1
        p1x = s2[2 * n2] / d2
        p2x = s2[2 * n2 + 1] / d2
        p3x = s2[2 * n2 - 1] / d2

        for i1 in range(2, n1):
            count2 = count1 + (i1 - 1) * n2
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[i1, n2]) ** 2)

            rows.append(count2)
            cols.append(count2 - 1)
            data.append(p1x * p3x)

            rows.append(count2)
            cols.append(count2 + n2)
            data.append(p1z * p2z)

            rows.append(count2)
            cols.append(count2 - n2)
            data.append(p1z * p3z)

    edge_nodes()

    ####################################################################################################
    # Corners
    def corner_nodes():

        # 1. Bottom Left
        count2 = 0
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = s2[2] / d2
        p2x = s2[3] / d2
        p3x = s2[1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[1, 1]) ** 2)

        rows.append(count2)
        cols.append(count2 + 1)
        data.append(p1x * p2x)

        rows.append(count2)
        cols.append(count2 + n2)
        data.append(p1z * p2z)

        # 2. Bottom Right
        count2 = n2 - 1
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = s2[2 * n2] / d2
        p2x = s2[2 * n2 + 1] / d2
        p3x = s2[2 * n2 - 1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[1, n2]) ** 2)

        rows.append(count2)
        cols.append(count2 - 1)
        data.append(p1x * p3x)

        rows.append(count2)
        cols.append(count2 + n2)
        data.append(p1z * p2z)

        # 3. Top Left
        count2 = (n1 - 1) * n2
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = s2[2] / d2
        p2x = s2[3] / d2
        p3x = s2[1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[n1, 1]) ** 2)

        rows.append(count2)
        cols.append(count2 + 1)
        data.append(p1x * p2x)

        rows.append(count2)
        cols.append(count2 - n2)
        data.append(p1z * p3z)

        # 4. Top Right
        count2 = (n1 - 1) * n2 + n2 - 1
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = s2[2 * n2] / d2
        p2x = s2[2 * n2 + 1] / d2
        p3x = s2[2 * n2 - 1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[n1, n2]) ** 2)

        rows.append(count2)
        cols.append(count2 - 1)
        data.append(p1x * p3x)

        rows.append(count2)
        cols.append(count2 - n2)
        data.append(p1z * p3z)

    corner_nodes()

    ####################################################################################################
    # Convert to csc format
    if adj:
        mat = csc_matrix((data, (cols, rows)), shape=(n1 * n2, n1 * n2))
        mat = mat.conjugate()
    else:
        mat = csc_matrix((data, (rows, cols)), shape=(n1 * n2, n1 * n2))

    return mat


def create_helmholtz3d_matrix(
        a1,
        a2,
        a3,
        pad1,
        pad2,
        pad3,
        omega,
        precision,
        vel,
        pml_damping=1.0,
        adj=False,
        warnings=True
):
    """
    :param a1: The domain on which Helmholtz equation is solved is [0,a1] x [0,a2] x [0,a3].
    :param a2: The domain on which Helmholtz equation is solved is [0,a1] x [0,a2] x [0,a3].
    :param a3: The domain on which Helmholtz equation is solved is [0,a1] x [0,a2] x [0,a3].
    :param pad1: Number of pad cells along x1 direction.
    :param pad2: Number of pad cells along x2 direction.
    :param pad3: Number of pad cells along x3 direction.
    :param omega:  Angular frequency. The Helmholtz equation reads (lap + omega^2 / vel^2)u = f.
    :param precision: np.complex64 or np.complex128
    :param vel: 3d numpy array real valued of shape n1 x n2 x n3. n1, n2, n3 >= 4 required.
    :param pml_damping: Positive float. Damping parameter for pml.
    :param adj: Boolean flag (whether to compute the adjoint)
    :param warnings: Boolean flag (whether to print warnings due to grid param check)

    :return: Sparse Helmholtz matrix of shape (n1 * n2 * n3) x (n1 * n2 * n3), with grid points
    enumerated in x3 followed by x2 followed by x1 directions.

    Note: Dirichlet boundary conditions imposed on boundary layer of nodes, by adding an extra layer of nodes.
    """

    # Check vel for type, make copy consistent with type of precision, and check if values > 0
    if precision not in [np.complex64, np.complex128]:
        raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

    TypeChecker.check(x=vel, expected_type=(np.ndarray,))

    n1, n2, n3 = vel.shape
    TypeChecker.check_int_lower_bound(x=n1, lb=4)
    TypeChecker.check_int_lower_bound(x=n2, lb=4)
    TypeChecker.check_int_lower_bound(x=n3, lb=4)

    if vel.shape != (n1, n2, n3):
        raise ValueError("Shape of 'vel' must match (" + str(n1) + ", " + str(n2) + ", " + str(n3) + ")")
    if vel.dtype not in (np.float32, np.float64):
        raise TypeError("dtype of 'vel' must be np.float32 or np.float64")

    if np.any(vel <= 0.0):
        raise ValueError("Non-positive velocities detected")

    # Check inputs
    TypeChecker.check_float_positive(x=a1)
    TypeChecker.check_float_positive(x=a2)
    TypeChecker.check_float_positive(x=a3)
    TypeChecker.check_int_bounds(x=pad1, lb=2, ub=int(n1 / 2))
    TypeChecker.check_int_bounds(x=pad2, lb=2, ub=int(n2 / 2))
    TypeChecker.check_int_bounds(x=pad3, lb=2, ub=int(n3 / 2))
    TypeChecker.check_float_positive(x=omega)
    TypeChecker.check_float_positive(x=pml_damping)
    TypeChecker.check(x=adj, expected_type=(bool,))
    TypeChecker.check(x=warnings, expected_type=(bool,))

    # Print grid size warnings
    d1 = a1 / (n1 - 1)
    d2 = a2 / (n2 - 1)
    d3 = a3 / (n3 - 1)
    vel_max = np.max(vel)
    vel_min = np.min(vel)
    lambda_min = 2 * np.pi * vel_min / omega
    lambda_max = 2 * np.pi * vel_max / omega
    pad_cells1 = pad1
    pad_cells2 = pad2
    pad_cells3 = pad3
    pml_width1 = d1 * pad_cells1
    pml_width2 = d2 * pad_cells2
    pml_width3 = d3 * pad_cells3
    dmin = lambda_min / 10.0  # 10 points per minimum wavelength

    if warnings:
        print("\n\n")

        if d1 > dmin:
            print("Warning: Required dmin = ", "{:.2e}".format(dmin), ", Computed d1 = ", "{:.2e}".format(d1))

        if d2 > dmin:
            print("Warning: Required dmin = ", "{:.2e}".format(dmin), ", Computed d2 = ", "{:.2e}".format(d2))

        if d3 > dmin:
            print("Warning: Required dmin = ", "{:.2e}".format(dmin), ", Computed d3 = ", "{:.2e}".format(d3))

        if pml_width1 < lambda_max:
            print("Warning: Required minimum pml width = ", "{:.2e}".format(lambda_max),
                  ", Computed pml_width1 = ", "{:.2e}".format(pml_width1))

        if pml_width2 < lambda_max:
            print("Warning: Required minimum pml width = ", "{:.2e}".format(lambda_max),
                  ", Computed pml_width2 = ", "{:.2e}".format(pml_width2))

        if pml_width3 < lambda_max:
            print("Warning: Required minimum pml width = ", "{:.2e}".format(lambda_max),
                  ", Computed pml_width3 = ", "{:.2e}".format(pml_width3))

    # Pad velocity by 1 layer with correct precision type
    vel1 = np.zeros((n1 + 2, n2 + 2, n3 + 2))
    if precision is np.complex64:
        vel1 = vel1.astype(np.float32)
        vel1[1: n1 + 1, 1: n2 + 1, 1: n3 + 1] = vel.astype(np.float32)
    if precision is np.complex128:
        vel1 = vel1.astype(np.float64)
        vel1[1: n1 + 1, 1: n2 + 1, 1: n3 + 1] = vel.astype(np.float64)

    vel1[1: n1 + 1, 1: n2 + 1, 0] = vel1[1: n1 + 1, 1: n2 + 1, 1]
    vel1[1: n1 + 1, 1: n2 + 1, n3 + 1] = vel1[1: n1 + 1, 1: n2 + 1, n3]
    vel1[1: n1 + 1, 0, :] = vel1[1: n1 + 1, 1, :]
    vel1[1: n1 + 1, n2 + 1, :] = vel1[1: n1 + 1, n2, :]
    vel1[0, :, :] = vel1[1, :, :]
    vel1[n1 + 1, :, :] = vel1[n1, :, :]

    # Number of nodes including DBC layer
    n1_ = n1 + 2
    n2_ = n2 + 2
    n3_ = n3 + 2
    a1_ = a1 + 2 * d1
    a2_ = a2 + 2 * d2
    a3_ = a3 + 2 * d3

    # Calculate s1 and s2 arrays
    def s1_array():

        d1_ = d1 / 2.0
        s1_ = np.zeros(shape=(2 * n1_ - 1,), dtype=precision)

        for kk in range(2 * pad_cells1 + 1):
            s1_[kk] = (1.0 - kk * d1_ / pml_width1) ** 2

        for kk in range(2 * (n1_ - 1 - pad_cells1), 2 * n1_ - 1):
            s1_[kk] = (1.0 - (a1_ - kk * d1_) / pml_width1) ** 2

        s1_ *= (complex(0, 1) / omega) * (pml_damping / pml_width1)
        s1_ += 1.0
        return 1.0 / s1_

    def s2_array():

        d2_ = d2 / 2.0
        s2_ = np.zeros(shape=(2 * n2_ - 1,), dtype=precision)

        for kk in range(2 * pad_cells2 + 1):
            s2_[kk] = (1.0 - kk * d2_ / pml_width2) ** 2

        for kk in range(2 * (n2_ - 1 - pad_cells2), 2 * n2_ - 1):
            s2_[kk] = (1.0 - (a2_ - kk * d2_) / pml_width2) ** 2

        s2_ *= (complex(0, 1) / omega) * (pml_damping / pml_width2)
        s2_ += 1.0
        return 1.0 / s2_

    def s3_array():

        d3_ = d3 / 2.0
        s3_ = np.zeros(shape=(2 * n3_ - 1,), dtype=precision)

        for kk in range(2 * pad_cells3 + 1):
            s3_[kk] = (1.0 - kk * d3_ / pml_width3) ** 2

        for kk in range(2 * (n3_ - 1 - pad_cells3), 2 * n3_ - 1):
            s3_[kk] = (1.0 - (a3_ - kk * d3_) / pml_width3) ** 2

        s3_ *= (complex(0, 1) / omega) * (pml_damping / pml_width3)
        s3_ += 1.0
        return 1.0 / s3_

    s1 = s1_array()
    s2 = s2_array()
    s3 = s3_array()

    # Create lists to hold matrix entries
    data = []
    rows = []
    cols = []

    ####################################################################################################
    # Loop over interior nodes except faces
    def interior_nodes():

        for i1 in tqdm(range(2, n1)):
            count1 = (i1 - 1) * n2 * n3
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            for i2 in range(2, n2):
                count2 = count1 + (i2 - 1) * n3
                p1x = s2[2 * i2] / d2
                p2x = s2[2 * i2 + 1] / d2
                p3x = s2[2 * i2 - 1] / d2

                for i3 in range(2, n3):
                    count3 = count2 + i3 - 1
                    p1y = s3[2 * i3] / d3
                    p2y = s3[2 * i3 + 1] / d3
                    p3y = s3[2 * i3 - 1] / d3

                    rows.append(count3)
                    cols.append(count3)
                    data.append(
                        - p1y * (p3y + p2y)
                        - p1x * (p3x + p2x)
                        - p1z * (p3z + p2z)
                        + (omega / vel1[i1, i2, i3]) ** 2
                    )

                    rows.append(count3)
                    cols.append(count3 + 1)
                    data.append(p1y * p2y)

                    rows.append(count3)
                    cols.append(count3 - 1)
                    data.append(p1y * p3y)

                    rows.append(count3)
                    cols.append(count3 + n3)
                    data.append(p1x * p2x)

                    rows.append(count3)
                    cols.append(count3 - n3)
                    data.append(p1x * p3x)

                    rows.append(count3)
                    cols.append(count3 + n2 * n3)
                    data.append(p1z * p2z)

                    rows.append(count3)
                    cols.append(count3 - n2 * n3)
                    data.append(p1z * p3z)

    interior_nodes()

    ####################################################################################################
    # Loop over face nodes except edges
    def face_nodes():

        # 1. Bottom face
        count1 = 0
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1

        for i2 in range(2, n2):
            count2 = count1 + (i2 - 1) * n3
            p1x = s2[2 * i2] / d2
            p2x = s2[2 * i2 + 1] / d2
            p3x = s2[2 * i2 - 1] / d2

            for i3 in range(2, n3):
                count3 = count2 + i3 - 1
                p1y = s3[2 * i3] / d3
                p2y = s3[2 * i3 + 1] / d3
                p3y = s3[2 * i3 - 1] / d3

                rows.append(count3)
                cols.append(count3)
                data.append(
                    - p1y * (p3y + p2y)
                    - p1x * (p3x + p2x)
                    - p1z * (p3z + p2z)
                    + (omega / vel1[1, i2, i3]) ** 2
                )

                rows.append(count3)
                cols.append(count3 + 1)
                data.append(p1y * p2y)

                rows.append(count3)
                cols.append(count3 - 1)
                data.append(p1y * p3y)

                rows.append(count3)
                cols.append(count3 + n3)
                data.append(p1x * p2x)

                rows.append(count3)
                cols.append(count3 - n3)
                data.append(p1x * p3x)

                rows.append(count3)
                cols.append(count3 + n2 * n3)
                data.append(p1z * p2z)

        # 2. Top face
        count1 = (n1 - 1) * n2 * n3
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1

        for i2 in range(2, n2):
            count2 = count1 + (i2 - 1) * n3
            p1x = s2[2 * i2] / d2
            p2x = s2[2 * i2 + 1] / d2
            p3x = s2[2 * i2 - 1] / d2

            for i3 in range(2, n3):
                count3 = count2 + i3 - 1
                p1y = s3[2 * i3] / d3
                p2y = s3[2 * i3 + 1] / d3
                p3y = s3[2 * i3 - 1] / d3

                rows.append(count3)
                cols.append(count3)
                data.append(
                    - p1y * (p3y + p2y)
                    - p1x * (p3x + p2x)
                    - p1z * (p3z + p2z)
                    + (omega / vel1[n1, i2, i3]) ** 2
                )

                rows.append(count3)
                cols.append(count3 + 1)
                data.append(p1y * p2y)

                rows.append(count3)
                cols.append(count3 - 1)
                data.append(p1y * p3y)

                rows.append(count3)
                cols.append(count3 + n3)
                data.append(p1x * p2x)

                rows.append(count3)
                cols.append(count3 - n3)
                data.append(p1x * p3x)

                rows.append(count3)
                cols.append(count3 - n2 * n3)
                data.append(p1z * p3z)

        # 3. Left face
        count1 = 0
        p1x = s2[2] / d2
        p2x = s2[3] / d2
        p3x = s2[1] / d2

        for i1 in range(2, n1):
            count2 = count1 + (i1 - 1) * n2 * n3
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            for i3 in range(2, n3):
                count3 = count2 + i3 - 1
                p1y = s3[2 * i3] / d3
                p2y = s3[2 * i3 + 1] / d3
                p3y = s3[2 * i3 - 1] / d3

                rows.append(count3)
                cols.append(count3)
                data.append(
                    - p1y * (p3y + p2y)
                    - p1x * (p3x + p2x)
                    - p1z * (p3z + p2z)
                    + (omega / vel1[i1, 1, i3]) ** 2
                )

                rows.append(count3)
                cols.append(count3 + 1)
                data.append(p1y * p2y)

                rows.append(count3)
                cols.append(count3 - 1)
                data.append(p1y * p3y)

                rows.append(count3)
                cols.append(count3 + n3)
                data.append(p1x * p2x)

                rows.append(count3)
                cols.append(count3 + n2 * n3)
                data.append(p1z * p2z)

                rows.append(count3)
                cols.append(count3 - n2 * n3)
                data.append(p1z * p3z)

        # 4. Right face
        count1 = (n2 - 1) * n3
        p1x = s2[2 * n2] / d2
        p2x = s2[2 * n2 + 1] / d2
        p3x = s2[2 * n2 - 1] / d2

        for i1 in range(2, n1):
            count2 = count1 + (i1 - 1) * n2 * n3
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            for i3 in range(2, n3):
                count3 = count2 + i3 - 1
                p1y = s3[2 * i3] / d3
                p2y = s3[2 * i3 + 1] / d3
                p3y = s3[2 * i3 - 1] / d3

                rows.append(count3)
                cols.append(count3)
                data.append(
                    - p1y * (p3y + p2y)
                    - p1x * (p3x + p2x)
                    - p1z * (p3z + p2z)
                    + (omega / vel1[i1, n2, i3]) ** 2
                )

                rows.append(count3)
                cols.append(count3 + 1)
                data.append(p1y * p2y)

                rows.append(count3)
                cols.append(count3 - 1)
                data.append(p1y * p3y)

                rows.append(count3)
                cols.append(count3 - n3)
                data.append(p1x * p3x)

                rows.append(count3)
                cols.append(count3 + n2 * n3)
                data.append(p1z * p2z)

                rows.append(count3)
                cols.append(count3 - n2 * n3)
                data.append(p1z * p3z)

        # 5. Front face
        count1 = 0
        p1y = s3[2] / d3
        p2y = s3[3] / d3
        p3y = s3[1] / d3

        for i1 in range(2, n1):
            count2 = count1 + (i1 - 1) * n2 * n3
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            for i2 in range(2, n2):
                count3 = count2 + (i2 - 1) * n3
                p1x = s2[2 * i2] / d2
                p2x = s2[2 * i2 + 1] / d2
                p3x = s2[2 * i2 - 1] / d2

                rows.append(count3)
                cols.append(count3)
                data.append(
                    - p1y * (p3y + p2y)
                    - p1x * (p3x + p2x)
                    - p1z * (p3z + p2z)
                    + (omega / vel1[i1, i2, 1]) ** 2
                )

                rows.append(count3)
                cols.append(count3 + 1)
                data.append(p1y * p2y)

                rows.append(count3)
                cols.append(count3 + n3)
                data.append(p1x * p2x)

                rows.append(count3)
                cols.append(count3 - n3)
                data.append(p1x * p3x)

                rows.append(count3)
                cols.append(count3 + n2 * n3)
                data.append(p1z * p2z)

                rows.append(count3)
                cols.append(count3 - n2 * n3)
                data.append(p1z * p3z)

        # 6. Back face
        count1 = n3 - 1
        p1y = s3[2 * n3] / d3
        p2y = s3[2 * n3 + 1] / d3
        p3y = s3[2 * n3 - 1] / d3

        for i1 in range(2, n1):
            count2 = count1 + (i1 - 1) * n2 * n3
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            for i2 in range(2, n2):
                count3 = count2 + (i2 - 1) * n3
                p1x = s2[2 * i2] / d2
                p2x = s2[2 * i2 + 1] / d2
                p3x = s2[2 * i2 - 1] / d2

                rows.append(count3)
                cols.append(count3)
                data.append(
                    - p1y * (p3y + p2y)
                    - p1x * (p3x + p2x)
                    - p1z * (p3z + p2z)
                    + (omega / vel1[i1, i2, n3]) ** 2
                )

                rows.append(count3)
                cols.append(count3 - 1)
                data.append(p1y * p3y)

                rows.append(count3)
                cols.append(count3 + n3)
                data.append(p1x * p2x)

                rows.append(count3)
                cols.append(count3 - n3)
                data.append(p1x * p3x)

                rows.append(count3)
                cols.append(count3 + n2 * n3)
                data.append(p1z * p2z)

                rows.append(count3)
                cols.append(count3 - n2 * n3)
                data.append(p1z * p3z)

    face_nodes()

    ####################################################################################################
    # Loop over edge nodes except corners
    def edge_nodes():

        # 1. Bottom-front edge
        count2 = 0
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1y = s3[2] / d3
        p2y = s3[3] / d3
        p3y = s3[1] / d3

        for i2 in range(2, n2):
            count3 = count2 + (i2 - 1) * n3
            p1x = s2[2 * i2] / d2
            p2x = s2[2 * i2 + 1] / d2
            p3x = s2[2 * i2 - 1] / d2

            rows.append(count3)
            cols.append(count3)
            data.append(
                - p1y * (p3y + p2y)
                - p1x * (p3x + p2x)
                - p1z * (p3z + p2z)
                + (omega / vel1[1, i2, 1]) ** 2
            )

            rows.append(count3)
            cols.append(count3 + 1)
            data.append(p1y * p2y)

            rows.append(count3)
            cols.append(count3 + n3)
            data.append(p1x * p2x)

            rows.append(count3)
            cols.append(count3 - n3)
            data.append(p1x * p3x)

            rows.append(count3)
            cols.append(count3 + n2 * n3)
            data.append(p1z * p2z)

        # 2. Bottom-back edge
        count2 = n3 - 1
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1y = s3[2 * n3] / d3
        p2y = s3[2 * n3 + 1] / d3
        p3y = s3[2 * n3 - 1] / d3

        for i2 in range(2, n2):
            count3 = count2 + (i2 - 1) * n3
            p1x = s2[2 * i2] / d2
            p2x = s2[2 * i2 + 1] / d2
            p3x = s2[2 * i2 - 1] / d2

            rows.append(count3)
            cols.append(count3)
            data.append(
                - p1y * (p3y + p2y)
                - p1x * (p3x + p2x)
                - p1z * (p3z + p2z)
                + (omega / vel1[1, i2, 1]) ** 2
            )

            rows.append(count3)
            cols.append(count3 - 1)
            data.append(p1y * p3y)

            rows.append(count3)
            cols.append(count3 + n3)
            data.append(p1x * p2x)

            rows.append(count3)
            cols.append(count3 - n3)
            data.append(p1x * p3x)

            rows.append(count3)
            cols.append(count3 + n2 * n3)
            data.append(p1z * p2z)

        # 3. Bottom-left edge
        count2 = 0
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = s2[2] / d2
        p2x = s2[3] / d2
        p3x = s2[1] / d2

        for i3 in range(2, n3):
            count3 = count2 + i3 - 1
            p1y = s3[2 * i3] / d3
            p2y = s3[2 * i3 + 1] / d3
            p3y = s3[2 * i3 - 1] / d3

            rows.append(count3)
            cols.append(count3)
            data.append(
                - p1y * (p3y + p2y)
                - p1x * (p3x + p2x)
                - p1z * (p3z + p2z)
                + (omega / vel1[1, 1, i3]) ** 2
            )

            rows.append(count3)
            cols.append(count3 + 1)
            data.append(p1y * p2y)

            rows.append(count3)
            cols.append(count3 - 1)
            data.append(p1y * p3y)

            rows.append(count3)
            cols.append(count3 + n3)
            data.append(p1x * p2x)

            rows.append(count3)
            cols.append(count3 + n2 * n3)
            data.append(p1z * p2z)

        # 4. Bottom-right edge
        count2 = (n2 - 1) * n3
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = s2[2 * n2] / d2
        p2x = s2[2 * n2 + 1] / d2
        p3x = s2[2 * n2 - 1] / d2

        for i3 in range(2, n3):
            count3 = count2 + i3 - 1
            p1y = s3[2 * i3] / d3
            p2y = s3[2 * i3 + 1] / d3
            p3y = s3[2 * i3 - 1] / d3

            rows.append(count3)
            cols.append(count3)
            data.append(
                - p1y * (p3y + p2y)
                - p1x * (p3x + p2x)
                - p1z * (p3z + p2z)
                + (omega / vel1[1, n2, i3]) ** 2
            )

            rows.append(count3)
            cols.append(count3 + 1)
            data.append(p1y * p2y)

            rows.append(count3)
            cols.append(count3 - 1)
            data.append(p1y * p3y)

            rows.append(count3)
            cols.append(count3 - n3)
            data.append(p1x * p3x)

            rows.append(count3)
            cols.append(count3 + n2 * n3)
            data.append(p1z * p2z)

        # 5. Top-front edge
        count2 = (n1 - 1) * n2 * n3
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1y = s3[2] / d3
        p2y = s3[3] / d3
        p3y = s3[1] / d3

        for i2 in range(2, n2):
            count3 = count2 + (i2 - 1) * n3
            p1x = s2[2 * i2] / d2
            p2x = s2[2 * i2 + 1] / d2
            p3x = s2[2 * i2 - 1] / d2

            rows.append(count3)
            cols.append(count3)
            data.append(
                - p1y * (p3y + p2y)
                - p1x * (p3x + p2x)
                - p1z * (p3z + p2z)
                + (omega / vel1[1, i2, 1]) ** 2
            )

            rows.append(count3)
            cols.append(count3 + 1)
            data.append(p1y * p2y)

            rows.append(count3)
            cols.append(count3 + n3)
            data.append(p1x * p2x)

            rows.append(count3)
            cols.append(count3 - n3)
            data.append(p1x * p3x)

            rows.append(count3)
            cols.append(count3 - n2 * n3)
            data.append(p1z * p3z)

        # 6. Top-back edge
        count2 = (n1 - 1) * n2 * n3 + n3 - 1
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1y = s3[2 * n3] / d3
        p2y = s3[2 * n3 + 1] / d3
        p3y = s3[2 * n3 - 1] / d3

        for i2 in range(2, n2):
            count3 = count2 + (i2 - 1) * n3
            p1x = s2[2 * i2] / d2
            p2x = s2[2 * i2 + 1] / d2
            p3x = s2[2 * i2 - 1] / d2

            rows.append(count3)
            cols.append(count3)
            data.append(
                - p1y * (p3y + p2y)
                - p1x * (p3x + p2x)
                - p1z * (p3z + p2z)
                + (omega / vel1[n1, i2, n3]) ** 2
            )

            rows.append(count3)
            cols.append(count3 - 1)
            data.append(p1y * p3y)

            rows.append(count3)
            cols.append(count3 + n3)
            data.append(p1x * p2x)

            rows.append(count3)
            cols.append(count3 - n3)
            data.append(p1x * p3x)

            rows.append(count3)
            cols.append(count3 - n2 * n3)
            data.append(p1z * p3z)

        # 7. Top-left edge
        count2 = (n1 - 1) * n2 * n3
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = s2[2] / d2
        p2x = s2[3] / d2
        p3x = s2[1] / d2

        for i3 in range(2, n3):
            count3 = count2 + i3 - 1
            p1y = s3[2 * i3] / d3
            p2y = s3[2 * i3 + 1] / d3
            p3y = s3[2 * i3 - 1] / d3

            rows.append(count3)
            cols.append(count3)
            data.append(
                - p1y * (p3y + p2y)
                - p1x * (p3x + p2x)
                - p1z * (p3z + p2z)
                + (omega / vel1[n1, 1, i3]) ** 2
            )

            rows.append(count3)
            cols.append(count3 + 1)
            data.append(p1y * p2y)

            rows.append(count3)
            cols.append(count3 - 1)
            data.append(p1y * p3y)

            rows.append(count3)
            cols.append(count3 + n3)
            data.append(p1x * p2x)

            rows.append(count3)
            cols.append(count3 - n2 * n3)
            data.append(p1z * p3z)

        # 8. Top-right edge
        count2 = (n1 - 1) * n2 * n3 + (n2 - 1) * n3
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = s2[2 * n2] / d2
        p2x = s2[2 * n2 + 1] / d2
        p3x = s2[2 * n2 - 1] / d2

        for i3 in range(2, n3):
            count3 = count2 + i3 - 1
            p1y = s3[2 * i3] / d3
            p2y = s3[2 * i3 + 1] / d3
            p3y = s3[2 * i3 - 1] / d3

            rows.append(count3)
            cols.append(count3)
            data.append(
                - p1y * (p3y + p2y)
                - p1x * (p3x + p2x)
                - p1z * (p3z + p2z)
                + (omega / vel1[n1, n2, i3]) ** 2
            )

            rows.append(count3)
            cols.append(count3 + 1)
            data.append(p1y * p2y)

            rows.append(count3)
            cols.append(count3 - 1)
            data.append(p1y * p3y)

            rows.append(count3)
            cols.append(count3 - n3)
            data.append(p1x * p3x)

            rows.append(count3)
            cols.append(count3 - n2 * n3)
            data.append(p1z * p3z)

        # 9. Front-left edge
        count2 = 0
        p1x = s2[2] / d2
        p2x = s2[3] / d2
        p3x = s2[1] / d2
        p1y = s3[2] / d3
        p2y = s3[3] / d3
        p3y = s3[1] / d3

        for i1 in range(2, n1):
            count3 = count2 + (i1 - 1) * n2 * n3
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            rows.append(count3)
            cols.append(count3)
            data.append(
                - p1y * (p3y + p2y)
                - p1x * (p3x + p2x)
                - p1z * (p3z + p2z)
                + (omega / vel1[i1, 1, 1]) ** 2
            )

            rows.append(count3)
            cols.append(count3 + 1)
            data.append(p1y * p2y)

            rows.append(count3)
            cols.append(count3 + n3)
            data.append(p1x * p2x)

            rows.append(count3)
            cols.append(count3 + n2 * n3)
            data.append(p1z * p2z)

            rows.append(count3)
            cols.append(count3 - n2 * n3)
            data.append(p1z * p3z)

        # 10. Front-right edge
        count2 = (n2 - 1) * n3
        p1x = s2[2 * n2] / d2
        p2x = s2[2 * n2 + 1] / d2
        p3x = s2[2 * n2 - 1] / d2
        p1y = s3[2] / d3
        p2y = s3[3] / d3
        p3y = s3[1] / d3

        for i1 in range(2, n1):
            count3 = count2 + (i1 - 1) * n2 * n3
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            rows.append(count3)
            cols.append(count3)
            data.append(
                - p1y * (p3y + p2y)
                - p1x * (p3x + p2x)
                - p1z * (p3z + p2z)
                + (omega / vel1[i1, n2, 1]) ** 2
            )

            rows.append(count3)
            cols.append(count3 + 1)
            data.append(p1y * p2y)

            rows.append(count3)
            cols.append(count3 - n3)
            data.append(p1x * p3x)

            rows.append(count3)
            cols.append(count3 + n2 * n3)
            data.append(p1z * p2z)

            rows.append(count3)
            cols.append(count3 - n2 * n3)
            data.append(p1z * p3z)

        # 11. Back-left edge
        count2 = n3 - 1
        p1x = s2[2] / d2
        p2x = s2[3] / d2
        p3x = s2[1] / d2
        p1y = s3[2 * n3] / d3
        p2y = s3[2 * n3 + 1] / d3
        p3y = s3[2 * n3 - 1] / d3

        for i1 in range(2, n1):
            count3 = count2 + (i1 - 1) * n2 * n3
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            rows.append(count3)
            cols.append(count3)
            data.append(
                - p1y * (p3y + p2y)
                - p1x * (p3x + p2x)
                - p1z * (p3z + p2z)
                + (omega / vel1[i1, 1, n3]) ** 2
            )

            rows.append(count3)
            cols.append(count3 - 1)
            data.append(p1y * p3y)

            rows.append(count3)
            cols.append(count3 + n3)
            data.append(p1x * p2x)

            rows.append(count3)
            cols.append(count3 + n2 * n3)
            data.append(p1z * p2z)

            rows.append(count3)
            cols.append(count3 - n2 * n3)
            data.append(p1z * p3z)

        # 12. Back-right edge
        count2 = (n2 - 1) * n3 + n3 - 1
        p1x = s2[2 * n2] / d2
        p2x = s2[2 * n2 + 1] / d2
        p3x = s2[2 * n2 - 1] / d2
        p1y = s3[2 * n3] / d3
        p2y = s3[2 * n3 + 1] / d3
        p3y = s3[2 * n3 - 1] / d3

        for i1 in range(2, n1):
            count3 = count2 + (i1 - 1) * n2 * n3
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            rows.append(count3)
            cols.append(count3)
            data.append(
                - p1y * (p3y + p2y)
                - p1x * (p3x + p2x)
                - p1z * (p3z + p2z)
                + (omega / vel1[i1, n2, n3]) ** 2
            )

            rows.append(count3)
            cols.append(count3 - 1)
            data.append(p1y * p3y)

            rows.append(count3)
            cols.append(count3 - n3)
            data.append(p1x * p3x)

            rows.append(count3)
            cols.append(count3 + n2 * n3)
            data.append(p1z * p2z)

            rows.append(count3)
            cols.append(count3 - n2 * n3)
            data.append(p1z * p3z)

    edge_nodes()

    ####################################################################################################
    # Corners
    def corner_nodes():

        # 1. Bottom-left-front corner
        count3 = 0
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = s2[2] / d2
        p2x = s2[3] / d2
        p3x = s2[1] / d2
        p1y = s3[2] / d3
        p2y = s3[3] / d3
        p3y = s3[1] / d3

        rows.append(count3)
        cols.append(count3)
        data.append(
            - p1y * (p3y + p2y)
            - p1x * (p3x + p2x)
            - p1z * (p3z + p2z)
            + (omega / vel1[1, 1, 1]) ** 2
        )

        rows.append(count3)
        cols.append(count3 + 1)
        data.append(p1y * p2y)

        rows.append(count3)
        cols.append(count3 + n3)
        data.append(p1x * p2x)

        rows.append(count3)
        cols.append(count3 + n2 * n3)
        data.append(p1z * p2z)

        # 2. Bottom-left-back corner
        count3 = n3 - 1
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = s2[2] / d2
        p2x = s2[3] / d2
        p3x = s2[1] / d2
        p1y = s3[2 * n3] / d3
        p2y = s3[2 * n3 + 1] / d3
        p3y = s3[2 * n3 - 1] / d3

        rows.append(count3)
        cols.append(count3)
        data.append(
            - p1y * (p3y + p2y)
            - p1x * (p3x + p2x)
            - p1z * (p3z + p2z)
            + (omega / vel1[1, 1, n3]) ** 2
        )

        rows.append(count3)
        cols.append(count3 - 1)
        data.append(p1y * p3y)

        rows.append(count3)
        cols.append(count3 + n3)
        data.append(p1x * p2x)

        rows.append(count3)
        cols.append(count3 + n2 * n3)
        data.append(p1z * p2z)

        # 3. Bottom-right-front corner
        count3 = (n2 - 1) * n3
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = s2[2 * n2] / d2
        p2x = s2[2 * n2 + 1] / d2
        p3x = s2[2 * n2 - 1] / d2
        p1y = s3[2] / d3
        p2y = s3[3] / d3
        p3y = s3[1] / d3

        rows.append(count3)
        cols.append(count3)
        data.append(
            - p1y * (p3y + p2y)
            - p1x * (p3x + p2x)
            - p1z * (p3z + p2z)
            + (omega / vel1[1, n2, 1]) ** 2
        )

        rows.append(count3)
        cols.append(count3 + 1)
        data.append(p1y * p2y)

        rows.append(count3)
        cols.append(count3 - n3)
        data.append(p1x * p3x)

        rows.append(count3)
        cols.append(count3 + n2 * n3)
        data.append(p1z * p2z)

        # 4. Bottom-right-back corner
        count3 = (n2 - 1) * n3 + n3 - 1
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = s2[2 * n2] / d2
        p2x = s2[2 * n2 + 1] / d2
        p3x = s2[2 * n2 - 1] / d2
        p1y = s3[2 * n3] / d3
        p2y = s3[2 * n3 + 1] / d3
        p3y = s3[2 * n3 - 1] / d3

        rows.append(count3)
        cols.append(count3)
        data.append(
            - p1y * (p3y + p2y)
            - p1x * (p3x + p2x)
            - p1z * (p3z + p2z)
            + (omega / vel1[1, n2, n3]) ** 2
        )

        rows.append(count3)
        cols.append(count3 - 1)
        data.append(p1y * p3y)

        rows.append(count3)
        cols.append(count3 - n3)
        data.append(p1x * p3x)

        rows.append(count3)
        cols.append(count3 + n2 * n3)
        data.append(p1z * p2z)

        # 5. Top-left-front corner
        count3 = (n1 - 1) * n2 * n3
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = s2[2] / d2
        p2x = s2[3] / d2
        p3x = s2[1] / d2
        p1y = s3[2] / d3
        p2y = s3[3] / d3
        p3y = s3[1] / d3

        rows.append(count3)
        cols.append(count3)
        data.append(
            - p1y * (p3y + p2y)
            - p1x * (p3x + p2x)
            - p1z * (p3z + p2z)
            + (omega / vel1[n1, 1, 1]) ** 2
        )

        rows.append(count3)
        cols.append(count3 + 1)
        data.append(p1y * p2y)

        rows.append(count3)
        cols.append(count3 + n3)
        data.append(p1x * p2x)

        rows.append(count3)
        cols.append(count3 - n2 * n3)
        data.append(p1z * p3z)

        # 6. Top-left-back corner
        count3 = (n1 - 1) * n2 * n3 + n3 - 1
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = s2[2] / d2
        p2x = s2[3] / d2
        p3x = s2[1] / d2
        p1y = s3[2 * n3] / d3
        p2y = s3[2 * n3 + 1] / d3
        p3y = s3[2 * n3 - 1] / d3

        rows.append(count3)
        cols.append(count3)
        data.append(
            - p1y * (p3y + p2y)
            - p1x * (p3x + p2x)
            - p1z * (p3z + p2z)
            + (omega / vel1[n1, 1, n3]) ** 2
        )

        rows.append(count3)
        cols.append(count3 - 1)
        data.append(p1y * p3y)

        rows.append(count3)
        cols.append(count3 + n3)
        data.append(p1x * p2x)

        rows.append(count3)
        cols.append(count3 - n2 * n3)
        data.append(p1z * p3z)

        # 7. Top-right-front corner
        count3 = (n1 - 1) * n2 * n3 + (n2 - 1) * n3
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = s2[2 * n2] / d2
        p2x = s2[2 * n2 + 1] / d2
        p3x = s2[2 * n2 - 1] / d2
        p1y = s3[2] / d3
        p2y = s3[3] / d3
        p3y = s3[1] / d3

        rows.append(count3)
        cols.append(count3)
        data.append(
            - p1y * (p3y + p2y)
            - p1x * (p3x + p2x)
            - p1z * (p3z + p2z)
            + (omega / vel1[n1, n2, 1]) ** 2
        )

        rows.append(count3)
        cols.append(count3 + 1)
        data.append(p1y * p2y)

        rows.append(count3)
        cols.append(count3 - n3)
        data.append(p1x * p3x)

        rows.append(count3)
        cols.append(count3 - n2 * n3)
        data.append(p1z * p3z)

        # 8. Top-right-back corner
        count3 = (n1 - 1) * n2 * n3 + (n2 - 1) * n3 + n3 - 1
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = s2[2 * n2] / d2
        p2x = s2[2 * n2 + 1] / d2
        p3x = s2[2 * n2 - 1] / d2
        p1y = s3[2 * n3] / d3
        p2y = s3[2 * n3 + 1] / d3
        p3y = s3[2 * n3 - 1] / d3

        rows.append(count3)
        cols.append(count3)
        data.append(
            - p1y * (p3y + p2y)
            - p1x * (p3x + p2x)
            - p1z * (p3z + p2z)
            + (omega / vel1[n1, n2, n3]) ** 2
        )

        rows.append(count3)
        cols.append(count3 - 1)
        data.append(p1y * p3y)

        rows.append(count3)
        cols.append(count3 - n3)
        data.append(p1x * p3x)

        rows.append(count3)
        cols.append(count3 - n2 * n3)
        data.append(p1z * p3z)

    corner_nodes()

    ####################################################################################################
    # Convert to csc format
    if adj:
        mat = csc_matrix((data, (cols, rows)), shape=(n1 * n2 * n3, n1 * n2 * n3))
        mat = mat.conjugate()
    else:
        mat = csc_matrix((data, (rows, cols)), shape=(n1 * n2 * n3, n1 * n2 * n3))

    return mat


def create_helmholtz2d_matrix_even(
        a1,
        a2,
        pad1,
        pad2,
        omega,
        precision,
        vel,
        pml_damping=1.0,
        adj=False,
        warnings=True
):
    """
    The problem has even symmetry along x2 direction about x2=0.

    :param a1: The domain on which Helmholtz equation is solved is [0,a1] x [0,a2].
    :param a2: The domain on which Helmholtz equation is solved is [0,a1] x [0,a2].
    :param pad1: Number of pad cells along x1 direction.
    :param pad2: Number of pad cells along x2 direction.
    :param omega: Angular frequency. The Helmholtz equation reads (lap + omega^2 / vel^2)u = f.
    :param precision: np.complex64 or np.complex128
    :param vel: 2d numpy array real valued of shape n1 x n2. n1, n2 >= 4 required.
    :param pml_damping: Positive float. Damping parameter for pml.
    :param adj: Boolean flag (whether to compute the adjoint)
    :param warnings: Boolean flag (whether to print warnings due to grid param check)

    :return: Sparse Helmholtz matrix of shape (n1 * n2) x (n1 * n2), with grid points
    enumerated row wise, i.e. x2 followed by x1 directions.

    *
    *
    *
    x1
    *
    *
    * * * x2 * * *

    Note: Dirichlet boundary conditions imposed on boundary layer of nodes, by adding an extra layer of nodes.
    """

    # Check vel for type, make copy consistent with type of precision, and check if values > 0
    if precision not in [np.complex64, np.complex128]:
        raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

    TypeChecker.check(x=vel, expected_type=(np.ndarray,))

    n1, n2 = vel.shape
    TypeChecker.check_int_lower_bound(x=n1, lb=4)
    TypeChecker.check_int_lower_bound(x=n2, lb=4)

    if vel.shape != (n1, n2):
        raise ValueError("Shape of 'vel' must match (" + str(n1) + ", " + str(n2) + ")")
    if vel.dtype not in (np.float32, np.float64):
        raise TypeError("dtype of 'vel' must be np.float32 or np.float64")

    if np.any(vel <= 0.0):
        raise ValueError("Non-positive velocities detected")

    # Check inputs
    TypeChecker.check_float_positive(x=a1)
    TypeChecker.check_float_positive(x=a2)
    TypeChecker.check_int_bounds(x=pad1, lb=2, ub=int(n1 / 2))
    TypeChecker.check_int_bounds(x=pad2, lb=2, ub=int(n2 / 2))
    TypeChecker.check_float_positive(x=omega)
    TypeChecker.check_float_positive(x=pml_damping)
    TypeChecker.check(x=adj, expected_type=(bool,))
    TypeChecker.check(x=warnings, expected_type=(bool,))

    # Print grid size warnings
    d1 = a1 / (n1 - 1)
    d2 = a2 / (n2 - 1)
    vel_max = np.max(vel)
    vel_min = np.min(vel)
    lambda_min = 2 * np.pi * vel_min / omega
    lambda_max = 2 * np.pi * vel_max / omega
    pad_cells1 = pad1
    pad_cells2 = pad2
    pml_width1 = d1 * pad_cells1
    pml_width2 = d2 * pad_cells2
    dmin = lambda_min / 10.0  # 10 points per minimum wavelength

    if warnings:
        print("\n\n"
              "Printing warnings \n")

        if d1 > dmin:
            print("Warning: Required dmin = ", "{:.2e}".format(dmin), ", Computed d1 = ", "{:.2e}".format(d1))

        if d2 > dmin:
            print("Warning: Required dmin = ", "{:.2e}".format(dmin), ", Computed d2 = ", "{:.2e}".format(d2))

        if pml_width1 < lambda_max:
            print("Warning: Required minimum pml width = ", "{:.2e}".format(lambda_max),
                  ", Computed pml_width1 = ", "{:.2e}".format(pml_width1))

        if pml_width2 < lambda_max:
            print("Warning: Required minimum pml width = ", "{:.2e}".format(lambda_max),
                  ", Computed pml_width2 = ", "{:.2e}".format(pml_width2))

    # Pad velocity by 1 layer with correct precision type
    vel1 = np.zeros((n1 + 2, n2 + 1))
    if precision is np.complex64:
        vel1 = vel1.astype(np.float32)
        vel1[1: n1 + 1, 0: n2] = vel.astype(np.float32)
    if precision is np.complex128:
        vel1 = vel1.astype(np.float64)
        vel1[1: n1 + 1, 0: n2] = vel.astype(np.float64)

    vel1[1: n1 + 1, n2] = vel1[1: n1 + 1, n2 - 1]
    vel1[0, :] = vel1[1, :]
    vel1[n1 + 1, :] = vel1[n1, :]

    # Number of nodes including DBC layer
    n1_ = n1 + 2
    n2_ = n2 + 1
    a1_ = a1 + 2 * d1
    a2_ = a2 + d2
    if warnings:
        print("Helmholtz system dimension: n1= ", n1_, ", n2= ", n2_)

    # Calculate s1 and s2 arrays
    def s1_array():

        d1_ = d1 / 2.0
        s1_ = np.zeros(shape=(2 * n1_ - 1,), dtype=precision)

        for kk in range(2 * pad_cells1 + 1):
            s1_[kk] = (1.0 - kk * d1_ / pml_width1) ** 2

        for kk in range(2 * (n1_ - 1 - pad_cells1), 2 * n1_ - 1):
            s1_[kk] = (1.0 - (a1_ - kk * d1_) / pml_width1) ** 2

        s1_ *= (complex(0, 1) / omega) * (pml_damping / pml_width1)
        s1_ += 1.0
        return 1.0 / s1_

    def s2_array():

        d2_ = d2 / 2.0
        s2_ = np.zeros(shape=(2 * n2_ - 1,), dtype=precision)

        for kk in range(2 * (n2_ - 1 - pad_cells2), 2 * n2_ - 1):
            s2_[kk] = (1.0 - (a2_ - kk * d2_) / pml_width2) ** 2

        s2_ *= (complex(0, 1) / omega) * (pml_damping / pml_width2)
        s2_ += 1.0
        return 1.0 / s2_

    s1 = s1_array()
    s2 = s2_array()

    # Create lists to hold matrix entries
    data = []
    rows = []
    cols = []

    ####################################################################################################
    # Loop over interior nodes except edges
    def interior_nodes():

        for i1 in range(2, n1):
            count1 = (i1 - 1) * n2
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            for i2 in range(1, n2 - 1):
                count2 = count1 + i2
                p1x = s2[2 * i2] / d2
                p2x = s2[2 * i2 + 1] / d2
                p3x = s2[2 * i2 - 1] / d2

                rows.append(count2)
                cols.append(count2)
                data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[i1, i2]) ** 2)

                rows.append(count2)
                cols.append(count2 + 1)
                data.append(p1x * p2x)

                rows.append(count2)
                cols.append(count2 - 1)
                data.append(p1x * p3x)

                rows.append(count2)
                cols.append(count2 + n2)
                data.append(p1z * p2z)

                rows.append(count2)
                cols.append(count2 - n2)
                data.append(p1z * p3z)

    interior_nodes()

    ####################################################################################################
    # Edges except corners
    def edge_nodes():

        # 1. Bottom
        count1 = 0
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1

        for i2 in range(1, n2 - 1):
            count2 = count1 + i2
            p1x = s2[2 * i2] / d2
            p2x = s2[2 * i2 + 1] / d2
            p3x = s2[2 * i2 - 1] / d2

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[1, i2]) ** 2)

            rows.append(count2)
            cols.append(count2 + 1)
            data.append(p1x * p2x)

            rows.append(count2)
            cols.append(count2 - 1)
            data.append(p1x * p3x)

            rows.append(count2)
            cols.append(count2 + n2)
            data.append(p1z * p2z)

        # 2. Top
        count1 = (n1 - 1) * n2
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1

        for i2 in range(1, n2 - 1):
            count2 = count1 + i2
            p1x = s2[2 * i2] / d2
            p2x = s2[2 * i2 + 1] / d2
            p3x = s2[2 * i2 - 1] / d2

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[n1, i2]) ** 2)

            rows.append(count2)
            cols.append(count2 + 1)
            data.append(p1x * p2x)

            rows.append(count2)
            cols.append(count2 - 1)
            data.append(p1x * p3x)

            rows.append(count2)
            cols.append(count2 - n2)
            data.append(p1z * p3z)

        # 3. Right
        count1 = n2 - 1
        p1x = s2[2 * (n2 - 1)] / d2
        p2x = s2[2 * (n2 - 1) + 1] / d2
        p3x = s2[2 * (n2 - 1) - 1] / d2

        for i1 in range(2, n1):
            count2 = count1 + (i1 - 1) * n2
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[i1, n2 - 1]) ** 2)

            rows.append(count2)
            cols.append(count2 - 1)
            data.append(p1x * p3x)

            rows.append(count2)
            cols.append(count2 + n2)
            data.append(p1z * p2z)

            rows.append(count2)
            cols.append(count2 - n2)
            data.append(p1z * p3z)

        # 4. Left
        count1 = 0
        p1x = s2[0] / d2
        p2x = s2[1] / d2
        p3x = s2[1] / d2

        for i1 in range(2, n1):
            count2 = count1 + (i1 - 1) * n2
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[i1, 0]) ** 2)

            rows.append(count2)
            cols.append(count2 + 1)
            data.append(p1x * p2x + p1x * p3x)

            rows.append(count2)
            cols.append(count2 + n2)
            data.append(p1z * p2z)

            rows.append(count2)
            cols.append(count2 - n2)
            data.append(p1z * p3z)

    edge_nodes()

    ####################################################################################################
    # Corners
    def corner_nodes():

        # 1. Bottom Left
        count2 = 0
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = s2[0] / d2
        p2x = s2[1] / d2
        p3x = s2[1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[1, 0]) ** 2)

        rows.append(count2)
        cols.append(count2 + 1)
        data.append(p1x * p2x + p1x * p3x)

        rows.append(count2)
        cols.append(count2 + n2)
        data.append(p1z * p2z)

        # 2. Bottom Right
        count2 = n2 - 1
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = s2[2 * (n2 - 1)] / d2
        p2x = s2[2 * (n2 - 1) + 1] / d2
        p3x = s2[2 * (n2 - 1) - 1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[1, n2 - 1]) ** 2)

        rows.append(count2)
        cols.append(count2 - 1)
        data.append(p1x * p3x)

        rows.append(count2)
        cols.append(count2 + n2)
        data.append(p1z * p2z)

        # 3. Top Left
        count2 = (n1 - 1) * n2
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = s2[0] / d2
        p2x = s2[1] / d2
        p3x = s2[1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[n1, 0]) ** 2)

        rows.append(count2)
        cols.append(count2 + 1)
        data.append(p1x * p2x + p1x * p3x)

        rows.append(count2)
        cols.append(count2 - n2)
        data.append(p1z * p3z)

        # 4. Top Right
        count2 = (n1 - 1) * n2 + n2 - 1
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = s2[2 * (n2 - 1)] / d2
        p2x = s2[2 * (n2 - 1) + 1] / d2
        p3x = s2[2 * (n2 - 1) - 1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[n1, n2 - 1]) ** 2)

        rows.append(count2)
        cols.append(count2 - 1)
        data.append(p1x * p3x)

        rows.append(count2)
        cols.append(count2 - n2)
        data.append(p1z * p3z)

    corner_nodes()

    ####################################################################################################
    # Convert to csc format
    if adj:
        mat = csc_matrix((data, (cols, rows)), shape=(n1 * n2, n1 * n2))
        mat = mat.conjugate()
    else:
        mat = csc_matrix((data, (rows, cols)), shape=(n1 * n2, n1 * n2))

    return mat


def create_helmholtz2d_matrix_radial(
        a1,
        a2,
        pad1,
        pad2,
        omega,
        precision,
        vel,
        pml_damping=1.0,
        adj=False,
        warnings=True
):
    """
        The problem has even symmetry along x2 direction about x2=0.

        :param a1: The domain on which Helmholtz equation is solved is [0,a1] x [0,a2].
        :param a2: The domain on which Helmholtz equation is solved is [0,a1] x [0,a2].
        :param pad1: Number of pad cells along x1 direction.
        :param pad2: Number of pad cells along x2 direction.
        :param omega: Angular frequency. This is the radial Helmholtz equation (as found in thesis).
        :param precision: np.complex64 or np.complex128
        :param vel: 2d numpy array real valued of shape n1 x n2. n1, n2 >= 4 required.
        :param pml_damping: Positive float. Damping parameter for pml.
        :param adj: Boolean flag (whether to compute the adjoint)
        :param warnings: Boolean flag (whether to print warnings due to grid param check)

        :return: Sparse Helmholtz matrix of shape (n1 * n2) x (n1 * n2), with grid points
        enumerated row wise, i.e. x2 followed by x1 directions.

        *
        *
        *
        x1
        *
        *
        * * * x2 * * *

        Note: Dirichlet boundary conditions imposed on boundary layer of nodes, by adding an extra layer of nodes.
        """

    # Check vel for type, make copy consistent with type of precision, and check if values > 0
    if precision not in [np.complex64, np.complex128]:
        raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

    TypeChecker.check(x=vel, expected_type=(np.ndarray,))

    n1, n2 = vel.shape
    TypeChecker.check_int_lower_bound(x=n1, lb=4)
    TypeChecker.check_int_lower_bound(x=n2, lb=4)

    if vel.shape != (n1, n2):
        raise ValueError("Shape of 'vel' must match (" + str(n1) + ", " + str(n2) + ")")
    if vel.dtype not in (np.float32, np.float64):
        raise TypeError("dtype of 'vel' must be np.float32 or np.float64")

    if np.any(vel <= 0.0):
        raise ValueError("Non-positive velocities detected")

    # Check inputs
    TypeChecker.check_float_positive(x=a1)
    TypeChecker.check_float_positive(x=a2)
    TypeChecker.check_int_bounds(x=pad1, lb=2, ub=int(n1 / 2))
    TypeChecker.check_int_bounds(x=pad2, lb=2, ub=int(n2 / 2))
    TypeChecker.check_float_positive(x=omega)
    TypeChecker.check_float_positive(x=pml_damping)
    TypeChecker.check(x=adj, expected_type=(bool,))
    TypeChecker.check(x=warnings, expected_type=(bool,))

    # Print grid size warnings
    d1 = a1 / (n1 - 1)
    d2 = a2 / (n2 - 1)
    vel_max = np.max(vel)
    vel_min = np.min(vel)
    lambda_min = 2 * np.pi * vel_min / omega
    lambda_max = 2 * np.pi * vel_max / omega
    pad_cells1 = pad1
    pad_cells2 = pad2
    pml_width1 = d1 * pad_cells1
    pml_width2 = d2 * pad_cells2
    dmin = lambda_min / 10.0  # 10 points per minimum wavelength

    if warnings:
        print("\n\n"
              "Printing warnings \n")

        if d1 > dmin:
            print("Warning: Required dmin = ", "{:.2e}".format(dmin), ", Computed d1 = ", "{:.2e}".format(d1))

        if d2 > dmin:
            print("Warning: Required dmin = ", "{:.2e}".format(dmin), ", Computed d2 = ", "{:.2e}".format(d2))

        if pml_width1 < lambda_max:
            print("Warning: Required minimum pml width = ", "{:.2e}".format(lambda_max),
                  ", Computed pml_width1 = ", "{:.2e}".format(pml_width1))

        if pml_width2 < lambda_max:
            print("Warning: Required minimum pml width = ", "{:.2e}".format(lambda_max),
                  ", Computed pml_width2 = ", "{:.2e}".format(pml_width2))

    # Pad velocity by 1 layer with correct precision type
    vel1 = np.zeros((n1 + 2, n2 + 1))
    if precision is np.complex64:
        vel1 = vel1.astype(np.float32)
        vel1[1: n1 + 1, 0: n2] = vel.astype(np.float32)
    if precision is np.complex128:
        vel1 = vel1.astype(np.float64)
        vel1[1: n1 + 1, 0: n2] = vel.astype(np.float64)

    vel1[1: n1 + 1, n2] = vel1[1: n1 + 1, n2 - 1]
    vel1[0, :] = vel1[1, :]
    vel1[n1 + 1, :] = vel1[n1, :]

    # Number of nodes including DBC layer
    n1_ = n1 + 2
    n2_ = n2 + 1
    a1_ = a1 + 2 * d1
    a2_ = a2 + d2

    # Calculate s1 and s2 arrays
    def s1_array():

        d1_ = d1 / 2.0
        s1_ = np.zeros(shape=(2 * n1_ - 1,), dtype=precision)

        for kk in range(2 * pad_cells1 + 1):
            s1_[kk] = (1.0 - kk * d1_ / pml_width1) ** 2

        for kk in range(2 * (n1_ - 1 - pad_cells1), 2 * n1_ - 1):
            s1_[kk] = (1.0 - (a1_ - kk * d1_) / pml_width1) ** 2

        s1_ *= (complex(0, 1) / omega) * (pml_damping / pml_width1)
        s1_ += 1.0
        return 1.0 / s1_

    def sr_array():

        d2_ = d2 / 2.0
        sr_ = np.zeros(shape=(2 * n2_ - 1,), dtype=precision)

        for kk in range(2 * (n2_ - 1 - pad_cells2), 2 * n2_ - 1):
            sr_[kk] = (1.0 - (a2_ - kk * d2_) / pml_width2) ** 2

        sr_ *= (complex(0, 1) / omega) * (pml_damping / pml_width2)
        sr_ += 1.0
        return 1.0 / sr_

    def rtilde_array():

        d2_ = d2 / 2.0
        r_ = np.zeros(shape=(2 * n2_ - 1,), dtype=precision)
        temp = 2 * (n2_ - 1 - pad_cells2) * d2_

        for kk in range(2 * (n2_ - 1 - pad_cells2), 2 * n2_ - 1):
            r_[kk] = (kk * d2_ - temp) ** 3

        r_ *= (complex(0, 1) / (3.0 * omega)) * (pml_damping / (pml_width2 ** 3.0))

        for kk in range(2 * n2_ - 1):
            r_[kk] += kk * d2_

        return r_

    s1 = s1_array()
    sr = sr_array()
    rtilde = rtilde_array()

    gamma_bar = sr * rtilde

    gamma = sr * 1.0
    gamma[0] = 0   # avoid division by zero, this sample never used
    for i in range(1, sr.shape[0]):
        gamma[i] = sr[i] / rtilde[i]

    # Create lists to hold matrix entries
    data = []
    rows = []
    cols = []

    ####################################################################################################
    # Loop over interior nodes except edges
    def interior_nodes():

        for i1 in range(2, n1):
            count1 = (i1 - 1) * n2
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            for i2 in range(1, n2 - 1):
                count2 = count1 + i2
                p1x = gamma[2 * i2] / d2
                p2x = gamma_bar[2 * i2 + 1] / d2
                p3x = gamma_bar[2 * i2 - 1] / d2

                rows.append(count2)
                cols.append(count2)
                data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[i1, i2]) ** 2)

                rows.append(count2)
                cols.append(count2 + 1)
                data.append(p1x * p2x)

                rows.append(count2)
                cols.append(count2 - 1)
                data.append(p1x * p3x)

                rows.append(count2)
                cols.append(count2 + n2)
                data.append(p1z * p2z)

                rows.append(count2)
                cols.append(count2 - n2)
                data.append(p1z * p3z)

    interior_nodes()

    ####################################################################################################
    # Edges except corners
    def edge_nodes():

        # 1. Bottom
        count1 = 0
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1

        for i2 in range(1, n2 - 1):
            count2 = count1 + i2
            p1x = gamma[2 * i2] / d2
            p2x = gamma_bar[2 * i2 + 1] / d2
            p3x = gamma_bar[2 * i2 - 1] / d2

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[1, i2]) ** 2)

            rows.append(count2)
            cols.append(count2 + 1)
            data.append(p1x * p2x)

            rows.append(count2)
            cols.append(count2 - 1)
            data.append(p1x * p3x)

            rows.append(count2)
            cols.append(count2 + n2)
            data.append(p1z * p2z)

        # 2. Top
        count1 = (n1 - 1) * n2
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1

        for i2 in range(1, n2 - 1):
            count2 = count1 + i2
            p1x = gamma[2 * i2] / d2
            p2x = gamma_bar[2 * i2 + 1] / d2
            p3x = gamma_bar[2 * i2 - 1] / d2

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[n1, i2]) ** 2)

            rows.append(count2)
            cols.append(count2 + 1)
            data.append(p1x * p2x)

            rows.append(count2)
            cols.append(count2 - 1)
            data.append(p1x * p3x)

            rows.append(count2)
            cols.append(count2 - n2)
            data.append(p1z * p3z)

        # 3. Right
        count1 = n2 - 1
        p1x = gamma[2 * (n2 - 1)] / d2
        p2x = gamma_bar[2 * (n2 - 1) + 1] / d2
        p3x = gamma_bar[2 * (n2 - 1) - 1] / d2

        for i1 in range(2, n1):
            count2 = count1 + (i1 - 1) * n2
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[i1, n2 - 1]) ** 2)

            rows.append(count2)
            cols.append(count2 - 1)
            data.append(p1x * p3x)

            rows.append(count2)
            cols.append(count2 + n2)
            data.append(p1z * p2z)

            rows.append(count2)
            cols.append(count2 - n2)
            data.append(p1z * p3z)

        # 4. Left
        count1 = 0
        p1x = 2.0 / d2
        p2x = 2.0 / d2

        for i1 in range(2, n1):
            count2 = count1 + (i1 - 1) * n2
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * p2x - p1z * (p3z + p2z) + (omega / vel1[i1, 0]) ** 2)

            rows.append(count2)
            cols.append(count2 + 1)
            data.append(p1x * p2x)

            rows.append(count2)
            cols.append(count2 + n2)
            data.append(p1z * p2z)

            rows.append(count2)
            cols.append(count2 - n2)
            data.append(p1z * p3z)

    edge_nodes()

    ####################################################################################################
    # Corners
    def corner_nodes():

        # 1. Bottom Left
        count2 = 0
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = 2.0 / d2
        p2x = 2.0 / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * p2x - p1z * (p3z + p2z) + (omega / vel1[1, 0]) ** 2)

        rows.append(count2)
        cols.append(count2 + 1)
        data.append(p1x * p2x)

        rows.append(count2)
        cols.append(count2 + n2)
        data.append(p1z * p2z)

        # 2. Bottom Right
        count2 = n2 - 1
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = gamma[2 * (n2 - 1)] / d2
        p2x = gamma_bar[2 * (n2 - 1) + 1] / d2
        p3x = gamma_bar[2 * (n2 - 1) - 1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[1, n2 - 1]) ** 2)

        rows.append(count2)
        cols.append(count2 - 1)
        data.append(p1x * p3x)

        rows.append(count2)
        cols.append(count2 + n2)
        data.append(p1z * p2z)

        # 3. Top Left
        count2 = (n1 - 1) * n2
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = 2.0 / d2
        p2x = 2.0 / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * p2x - p1z * (p3z + p2z) + (omega / vel1[n1, 0]) ** 2)

        rows.append(count2)
        cols.append(count2 + 1)
        data.append(p1x * p2x)

        rows.append(count2)
        cols.append(count2 - n2)
        data.append(p1z * p3z)

        # 4. Top Right
        count2 = (n1 - 1) * n2 + n2 - 1
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = gamma[2 * (n2 - 1)] / d2
        p2x = gamma_bar[2 * (n2 - 1) + 1] / d2
        p3x = gamma_bar[2 * (n2 - 1) - 1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[n1, n2 - 1]) ** 2)

        rows.append(count2)
        cols.append(count2 - 1)
        data.append(p1x * p3x)

        rows.append(count2)
        cols.append(count2 - n2)
        data.append(p1z * p3z)

    corner_nodes()

    ####################################################################################################
    # Convert to csc format
    if adj:
        mat = csc_matrix((data, (cols, rows)), shape=(n1 * n2, n1 * n2))
        mat = mat.conjugate()
    else:
        mat = csc_matrix((data, (rows, cols)), shape=(n1 * n2, n1 * n2))

    return mat


def create_helmholtz2d_matrix_radial_full(
        a1,
        a2,
        pad1,
        pad2,
        omega,
        precision,
        vel,
        pml_damping=1.0,
        adj=False,
        warnings=True
):
    """
    :param a1: The domain on which Helmholtz equation is solved is [0,a1] x [-a2/2,a2/2].
    :param a2: The domain on which Helmholtz equation is solved is [0,a1] x [-a2/2,a2/2].
    :param pad1: Number of pad cells along x1 direction.
    :param pad2: Number of pad cells along x2 direction.
    :param omega:  Angular frequency. The Helmholtz equation reads (lap + omega^2 / vel^2)u = f.
    :param precision: np.complex64 or np.complex128
    :param vel: 2d numpy array real valued of shape n1 x n2. n1, n2 >= 4 required (n2 must be odd).
    :param pml_damping: Positive float. Damping parameter for pml.
    :param adj: Boolean flag (whether to compute the adjoint)
    :param warnings: Boolean flag (whether to print warnings due to grid param check)

    :return: Sparse Helmholtz matrix of shape (n1 * n2) x (n1 * n2), with grid points
    enumerated row wise, i.e. x2 followed by x1 directions.

    *
    *
    *
    x1
    *
    *
    * * * x2 * * *

    Note: Dirichlet boundary conditions imposed on boundary layer of nodes, by adding an extra layer of nodes.
    """

    # Check vel for type, make copy consistent with type of precision, and check if values > 0
    if precision not in [np.complex64, np.complex128]:
        raise TypeError("Only precision types numpy.complex64 or numpy.complex128 are supported")

    TypeChecker.check(x=vel, expected_type=(np.ndarray,))

    n1, n2 = vel.shape
    TypeChecker.check_int_lower_bound(x=n1, lb=4)
    TypeChecker.check_int_lower_bound(x=n2, lb=4)
    if n2 % 2 == 0:
        raise ValueError("n2 must be odd")

    if vel.shape != (n1, n2):
        raise ValueError("Shape of 'vel' must match (" + str(n1) + ", " + str(n2) + ")")
    if vel.dtype not in (np.float32, np.float64):
        raise TypeError("dtype of 'vel' must be np.float32 or np.float64")

    if np.any(vel <= 0.0):
        raise ValueError("Non-positive velocities detected")

    # Check inputs
    TypeChecker.check_float_positive(x=a1)
    TypeChecker.check_float_positive(x=a2)
    TypeChecker.check_int_bounds(x=pad1, lb=2, ub=int(n1 / 2))
    TypeChecker.check_int_bounds(x=pad2, lb=2, ub=int(n2 / 2))
    TypeChecker.check_float_positive(x=omega)
    TypeChecker.check_float_positive(x=pml_damping)
    TypeChecker.check(x=adj, expected_type=(bool,))
    TypeChecker.check(x=warnings, expected_type=(bool,))

    # Print grid size warnings
    d1 = a1 / (n1 - 1)
    d2 = a2 / (n2 - 1)
    vel_max = np.max(vel)
    vel_min = np.min(vel)
    lambda_min = 2 * np.pi * vel_min / omega
    lambda_max = 2 * np.pi * vel_max / omega
    pad_cells1 = pad1
    pad_cells2 = pad2
    pml_width1 = d1 * pad_cells1
    pml_width2 = d2 * pad_cells2
    dmin = lambda_min / 10.0  # 10 points per minimum wavelength

    if warnings:
        print("\n\n")

        if d1 > dmin:
            print("Warning: Required dmin = ", "{:.2e}".format(dmin), ", Computed d1 = ", "{:.2e}".format(d1))

        if d2 > dmin:
            print("Warning: Required dmin = ", "{:.2e}".format(dmin), ", Computed d2 = ", "{:.2e}".format(d2))

        if pml_width1 < lambda_max:
            print("Warning: Required minimum pml width = ", "{:.2e}".format(lambda_max),
                  ", Computed pml_width1 = ", "{:.2e}".format(pml_width1))

        if pml_width2 < lambda_max:
            print("Warning: Required minimum pml width = ", "{:.2e}".format(lambda_max),
                  ", Computed pml_width2 = ", "{:.2e}".format(pml_width2))

    # Pad velocity by 1 layer with correct precision type
    vel1 = np.zeros((n1 + 2, n2 + 2))
    if precision is np.complex64:
        vel1 = vel1.astype(np.float32)
        vel1[1: n1 + 1, 1: n2 + 1] = vel.astype(np.float32)
    if precision is np.complex128:
        vel1 = vel1.astype(np.float64)
        vel1[1: n1 + 1, 1: n2 + 1] = vel.astype(np.float64)

    vel1[1: n1 + 1, 0] = vel1[1: n1 + 1, 1]
    vel1[1: n1 + 1, n2 + 1] = vel1[1: n1 + 1, n2]
    vel1[0, :] = vel1[1, :]
    vel1[n1 + 1, :] = vel1[n1, :]

    # Number of nodes including DBC layer
    n1_ = n1 + 2
    n2_ = n2 + 2
    a1_ = a1 + 2 * d1
    a2_ = a2 + 2 * d2

    # Calculate s1 and s2 arrays
    def s1_array():

        d1_ = d1 / 2.0
        s1_ = np.zeros(shape=(2 * n1_ - 1,), dtype=precision)

        for kk in range(2 * pad_cells1 + 1):
            s1_[kk] = (1.0 - kk * d1_ / pml_width1) ** 2

        for kk in range(2 * (n1_ - 1 - pad_cells1), 2 * n1_ - 1):
            s1_[kk] = (1.0 - (a1_ - kk * d1_) / pml_width1) ** 2

        s1_ *= (complex(0, 1) / omega) * (pml_damping / pml_width1)
        s1_ += 1.0
        return 1.0 / s1_

    def sr_array():

        d2_ = d2 / 2.0
        sr_ = np.zeros(shape=(2 * n2_ - 1,), dtype=precision)

        for kk in range(2 * pad_cells2 + 1):
            sr_[kk] = (1.0 - kk * d2_ / pml_width1) ** 2

        for kk in range(2 * (n2_ - 1 - pad_cells2), 2 * n2_ - 1):
            sr_[kk] = (1.0 - (a2_ - kk * d2_) / pml_width2) ** 2

        sr_ *= (complex(0, 1) / omega) * (pml_damping / pml_width2)
        sr_ += 1.0
        return 1.0 / sr_

    def rtilde_array():

        d2_ = d2 / 2.0
        r_ = np.zeros(shape=(2 * n2_ - 1,), dtype=precision)
        temp = (n2_ - 1 - 2 * pad_cells2) * d2_

        for kk in range(2 * pad_cells2 + 1):
            r_[kk] = np.abs((-0.5 * a2_ + kk * d2_ + temp) ** 3)

        for kk in range(2 * (n2_ - 1 - pad_cells2), 2 * n2_ - 1):
            r_[kk] = (-0.5 * a2_ + kk * d2_ - temp) ** 3

        r_ *= (complex(0, 1) / (3.0 * omega)) * (pml_damping / (pml_width2 ** 3.0))

        for kk in range(2 * n2_ - 1):
            r_[kk] += np.abs(-0.5 * a2_ + kk * d2_)

        return r_

    s1 = s1_array()
    sr = sr_array()
    rtilde = rtilde_array()

    gamma_bar = sr * rtilde

    gamma = sr * 1.0
    for i in range(0, sr.shape[0]):
        if i == n2_ - 1:
            gamma[i] = 0  # avoid division by zero, this sample never used
        else:
            gamma[i] = sr[i] / rtilde[i]

    # Create lists to hold matrix entries
    data = []
    rows = []
    cols = []

    ####################################################################################################
    # Loop over interior nodes except edges

    def interior_nodes():

        for i1 in range(2, n1):
            count1 = (i1 - 1) * n2
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            for i2 in range(2, n2):

                if i2 != int((n2 + 1)/2):
                    count2 = count1 + i2 - 1
                    p1x = gamma[2 * i2] / d2
                    p2x = gamma_bar[2 * i2 + 1] / d2
                    p3x = gamma_bar[2 * i2 - 1] / d2

                    rows.append(count2)
                    cols.append(count2)
                    data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[i1, i2]) ** 2)

                    rows.append(count2)
                    cols.append(count2 + 1)
                    data.append(p1x * p2x)

                    rows.append(count2)
                    cols.append(count2 - 1)
                    data.append(p1x * p3x)

                    rows.append(count2)
                    cols.append(count2 + n2)
                    data.append(p1z * p2z)

                    rows.append(count2)
                    cols.append(count2 - n2)
                    data.append(p1z * p3z)

                else:
                    count2 = count1 + i2 - 1
                    p1x = 1.0 / d2
                    p2x = 1.0 / d2
                    p3x = 1.0 / d2

                    rows.append(count2)
                    cols.append(count2)
                    data.append(-2.0 * p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[i1, i2]) ** 2)

                    rows.append(count2)
                    cols.append(count2 + 1)
                    data.append(2.0 * p1x * p2x)

                    rows.append(count2)
                    cols.append(count2 - 1)
                    data.append(2.0 * p1x * p3x)

                    rows.append(count2)
                    cols.append(count2 + n2)
                    data.append(p1z * p2z)

                    rows.append(count2)
                    cols.append(count2 - n2)
                    data.append(p1z * p3z)

    interior_nodes()

    ####################################################################################################
    # Edges except corners
    def edge_nodes():

        # 1. Bottom
        count1 = 0
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1

        for i2 in range(2, n2):

            if i2 != int((n2 + 1) / 2):
                count2 = count1 + i2 - 1
                p1x = gamma[2 * i2] / d2
                p2x = gamma_bar[2 * i2 + 1] / d2
                p3x = gamma_bar[2 * i2 - 1] / d2

                rows.append(count2)
                cols.append(count2)
                data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[1, i2]) ** 2)

                rows.append(count2)
                cols.append(count2 + 1)
                data.append(p1x * p2x)

                rows.append(count2)
                cols.append(count2 - 1)
                data.append(p1x * p3x)

                rows.append(count2)
                cols.append(count2 + n2)
                data.append(p1z * p2z)

            else:
                count2 = count1 + i2 - 1
                p1x = 1.0 / d2
                p2x = 1.0 / d2
                p3x = 1.0 / d2

                rows.append(count2)
                cols.append(count2)
                data.append(-2.0 * p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[1, i2]) ** 2)

                rows.append(count2)
                cols.append(count2 + 1)
                data.append(2.0 * p1x * p2x)

                rows.append(count2)
                cols.append(count2 - 1)
                data.append(2.0 * p1x * p3x)

                rows.append(count2)
                cols.append(count2 + n2)
                data.append(p1z * p2z)

        # 2. Top
        count1 = (n1 - 1) * n2
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1

        for i2 in range(2, n2):

            if i2 != int((n2 + 1) / 2):
                count2 = count1 + i2 - 1
                p1x = gamma[2 * i2] / d2
                p2x = gamma_bar[2 * i2 + 1] / d2
                p3x = gamma_bar[2 * i2 - 1] / d2

                rows.append(count2)
                cols.append(count2)
                data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[n1, i2]) ** 2)

                rows.append(count2)
                cols.append(count2 + 1)
                data.append(p1x * p2x)

                rows.append(count2)
                cols.append(count2 - 1)
                data.append(p1x * p3x)

                rows.append(count2)
                cols.append(count2 - n2)
                data.append(p1z * p3z)

            else:
                count2 = count1 + i2 - 1
                p1x = 1.0 / d2
                p2x = 1.0 / d2
                p3x = 1.0 / d2

                rows.append(count2)
                cols.append(count2)
                data.append(-2.0 * p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[n1, i2]) ** 2)

                rows.append(count2)
                cols.append(count2 + 1)
                data.append(2.0 * p1x * p2x)

                rows.append(count2)
                cols.append(count2 - 1)
                data.append(2.0 * p1x * p3x)

                rows.append(count2)
                cols.append(count2 - n2)
                data.append(p1z * p3z)

        # 3. Left
        count1 = 0
        p1x = gamma[2] / d2
        p2x = gamma_bar[3] / d2
        p3x = gamma_bar[1] / d2

        for i1 in range(2, n1):
            count2 = count1 + (i1 - 1) * n2
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[i1, 1]) ** 2)

            rows.append(count2)
            cols.append(count2 + 1)
            data.append(p1x * p2x)

            rows.append(count2)
            cols.append(count2 + n2)
            data.append(p1z * p2z)

            rows.append(count2)
            cols.append(count2 - n2)
            data.append(p1z * p3z)

        # 4. Right
        count1 = n2 - 1
        p1x = gamma[2 * n2] / d2
        p2x = gamma_bar[2 * n2 + 1] / d2
        p3x = gamma_bar[2 * n2 - 1] / d2

        for i1 in range(2, n1):
            count2 = count1 + (i1 - 1) * n2
            p1z = s1[2 * i1] / d1
            p2z = s1[2 * i1 + 1] / d1
            p3z = s1[2 * i1 - 1] / d1

            rows.append(count2)
            cols.append(count2)
            data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[i1, n2]) ** 2)

            rows.append(count2)
            cols.append(count2 - 1)
            data.append(p1x * p3x)

            rows.append(count2)
            cols.append(count2 + n2)
            data.append(p1z * p2z)

            rows.append(count2)
            cols.append(count2 - n2)
            data.append(p1z * p3z)

    edge_nodes()

    ####################################################################################################
    # Corners
    def corner_nodes():

        # 1. Bottom Left
        count2 = 0
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = gamma[2] / d2
        p2x = gamma_bar[3] / d2
        p3x = gamma_bar[1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[1, 1]) ** 2)

        rows.append(count2)
        cols.append(count2 + 1)
        data.append(p1x * p2x)

        rows.append(count2)
        cols.append(count2 + n2)
        data.append(p1z * p2z)

        # 2. Bottom Right
        count2 = n2 - 1
        p1z = s1[2] / d1
        p2z = s1[3] / d1
        p3z = s1[1] / d1
        p1x = gamma[2 * n2] / d2
        p2x = gamma_bar[2 * n2 + 1] / d2
        p3x = gamma_bar[2 * n2 - 1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[1, n2]) ** 2)

        rows.append(count2)
        cols.append(count2 - 1)
        data.append(p1x * p3x)

        rows.append(count2)
        cols.append(count2 + n2)
        data.append(p1z * p2z)

        # 3. Top Left
        count2 = (n1 - 1) * n2
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = gamma[2] / d2
        p2x = gamma_bar[3] / d2
        p3x = gamma_bar[1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[n1, 1]) ** 2)

        rows.append(count2)
        cols.append(count2 + 1)
        data.append(p1x * p2x)

        rows.append(count2)
        cols.append(count2 - n2)
        data.append(p1z * p3z)

        # 4. Top Right
        count2 = (n1 - 1) * n2 + n2 - 1
        p1z = s1[2 * n1] / d1
        p2z = s1[2 * n1 + 1] / d1
        p3z = s1[2 * n1 - 1] / d1
        p1x = gamma[2 * n2] / d2
        p2x = gamma_bar[2 * n2 + 1] / d2
        p3x = gamma_bar[2 * n2 - 1] / d2

        rows.append(count2)
        cols.append(count2)
        data.append(- p1x * (p3x + p2x) - p1z * (p3z + p2z) + (omega / vel1[n1, n2]) ** 2)

        rows.append(count2)
        cols.append(count2 - 1)
        data.append(p1x * p3x)

        rows.append(count2)
        cols.append(count2 - n2)
        data.append(p1z * p3z)

    corner_nodes()

    ####################################################################################################
    # Convert to csc format
    if adj:
        mat = csc_matrix((data, (cols, rows)), shape=(n1 * n2, n1 * n2))
        mat = mat.conjugate()
    else:
        mat = csc_matrix((data, (rows, cols)), shape=(n1 * n2, n1 * n2))

    return mat

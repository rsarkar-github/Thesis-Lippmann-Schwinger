import numpy as np
import time
import matplotlib.pyplot as plt


def box_der_altproj_2d(x, xmin, xmax, epsilon_min=0.0, epsilon_max=0.0, tol=1e-6):
    """
    This method takes a point x and projects it onto the intersection
    of the following sets:

    1. xmin <= x <= xmax
    2. epsilon_min <= D1x <= epsilon_max (derivative along 1st axis)
    3. epsilon_min <= D2x <= epsilon_max (derivative along 2nd axis)

    @params
    xmin, xmax: 2D numpy arrays of floats
    epsilon_min, epsilon_max: scalars (floats)
    """

    # Check parameters
    if epsilon_min > epsilon_max:
        raise ValueError("Violated constraint: epsilon_min <= epsilon_max")

    # Allocate temporary variables
    x1 = np.copy(x)
    x2 = np.copy(x)

    # Get shape information
    n1 = x.shape[0]
    n2 = x.shape[1]

    # Precompute to save multiplications
    delta_max = 0.5 * epsilon_max
    delta_min = 0.5 * epsilon_min

    # Variable for number of iterations
    niter = 0

    while 1:

        # Project on set 1
        np.clip(x1, xmin, xmax, out=x2)

        # Project on set 2
        for i2 in range(n2 - 1):

            f = 0.5 * (x2[:, i2] - x2[:, i2 + 1])
            f1 = delta_max - f
            np.clip(f1, a_min=None, a_max=0, out=f1)
            f2 = delta_min - f
            np.clip(f2, a_min=0, a_max=None, out=f2)
            f1 += f2

            x2[:, i2] += f1
            x2[:, i2 + 1] -= f1

        # Project on set 3
        for i1 in range(n1 - 1):

            f = 0.5 * (x2[i1, :] - x2[i1 + 1, :])
            f1 = delta_max - f
            np.clip(f1, a_min=None, a_max=0, out=f1)
            f2 = delta_min - f
            np.clip(f2, a_min=0, a_max=None, out=f2)
            f1 += f2

            x2[i1, :] += f1
            x2[i1 + 1, :] -= f1

        # Update iteration count
        niter += 1

        # Check for convergence
        d1 = np.amax(np.abs(x1 - x2))
        if d1 < tol:
            return x2, niter

        # Update x1
        x1 *= 0
        x1 += x2


if __name__ == "__main__":

    n1_ = 500
    n2_ = 500

    # Example 1
    xmin_ = -3.0
    xmax_ = 3.0
    low = -10.0
    high = 10.0

    x_ = np.random.uniform(low=low, high=high, size=(n1_, n2_))
    x1_ = np.zeros((n1_, n2_)) + xmin_
    x2_ = np.zeros((n1_, n2_)) + xmax_

    t_start = time.time()
    y, niter_ = box_der_altproj_2d(x=x_, xmin=x1_, xmax=x2_, epsilon_min=-0.5, epsilon_max=0.5)
    t_end = time.time()

    print("Time taken = " + "{:5.2f}".format(t_end - t_start)
          + " sec, Number of alternating projections = " + str(niter_))

    plt.figure(figsize=(8, 8))
    plt.imshow(x_, cmap="hot", vmin=low, vmax=high)
    plt.colorbar()
    plt.title("Original")
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.imshow(y, cmap="hot", vmin=low, vmax=high)
    plt.colorbar()
    plt.title("Projected")
    plt.show()

    # Example 2
    x_ = np.zeros((n1_, n2_)) + 1
    x_[:, int(n2_ / 2)] = -6.0

    x1_ = np.zeros((n1_, n2_)) + xmin_
    x2_ = np.zeros((n1_, n2_)) + xmax_

    t_start = time.time()
    y, niter_ = box_der_altproj_2d(x=x_, xmin=x1_, xmax=x2_, epsilon_min=-0.1, epsilon_max=0.1)
    t_end = time.time()

    print("Time taken = " + "{:5.2f}".format(t_end - t_start)
          + " sec, Number of alternating projections = " + str(niter_))

    plt.figure(figsize=(8, 8))
    plt.imshow(x_, cmap="hot", vmin=low, vmax=high)
    plt.colorbar()
    plt.title("Original")
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.imshow(y, cmap="hot", vmin=low, vmax=high)
    plt.colorbar()
    plt.title("Projected")
    plt.show()

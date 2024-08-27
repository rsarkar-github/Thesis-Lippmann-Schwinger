import numpy as np
from numba import vectorize, float64, complex128
import scipy.special as sp
import time


@vectorize([complex128(complex128)], nopython=True)
def gamma(x):
    """
    Computes Gamma function using Lanczos method
    :param x: complex number (should not be a pole of gamma function)
    :return: Gamma(x)
    """

    if x == 2.0 or x == 1.0:
        return complex(1.0, 0)

    g = 607.0 / 128.0
    c = [
        0.99999999999999709182,
        57.156235665862923517,
        -59.597960355475491248,
        14.136097974741747174,
        -0.49191381609762019978,
        .33994649984811888699e-4,
        .46523628927048575665e-4,
        -.98374475304879564677e-4,
        .15808870322491248884e-3,
        -.21026444172410488319e-3,
        .21743961811521264320e-3,
        -.16431810653676389022e-3,
        .84418223983852743293e-4,
        -.26190838401581408670e-4,
        .36899182659531622704e-5
    ]

    if np.real(x) < 0:
        xx = x
        x = -x
        x = x - 1
        xh = x + 0.5
        xgh = xh + g
        # Trick for avoiding FP overflow above z=141
        xp = xgh ** (xh * 0.5)

        # Evaluate sum
        ss = 0.0
        for pp in range(14, 0, -1):
            ss = ss + c[pp] / (x + pp)

        sq2pi = 2.5066282746310005024157652848110
        f = (sq2pi * (c[0] + ss)) * ((xp * np.exp(-1.0 * xgh)) * xp)
        f = -1.0 * np.pi / (xx * f * np.sin(np.pi * xx))

    else:
        x = x - 1
        xh = x + 0.5
        xgh = xh + g
        # Trick for avoiding FP overflow above z=141
        xp = xgh ** (xh * 0.5)

        # Evaluate sum
        ss = 0.0
        for pp in range(14, 0, -1):
            ss = ss + c[pp] / (x + pp)

        sq2pi = 2.5066282746310005024157652848110
        f = (sq2pi * (c[0] + ss)) * ((xp * np.exp(-1.0 * xgh)) * xp)

    return f


@vectorize([complex128(complex128)], nopython=True)
def digamma(x):
    """
    Computes digamma function using Lanczos method
    :param x: complex number (should not be a pole of digamma function)
    :return: digamma(x)
    """

    g = 607.0 / 128.0
    c = [
        0.99999999999999709182,
        57.156235665862923517,
        -59.597960355475491248,
        14.136097974741747174,
        -0.49191381609762019978,
        .33994649984811888699e-4,
        .46523628927048575665e-4,
        -.98374475304879564677e-4,
        .15808870322491248884e-3,
        -.21026444172410488319e-3,
        .21743961811521264320e-3,
        -.16431810653676389022e-3,
        .84418223983852743293e-4,
        -.26190838401581408670e-4,
        .36899182659531622704e-5
    ]

    if np.real(x) < 0.5:
        xx = x
        x = 1.0 - x

        n = 0.0
        d = 0.0
        for k in range(15, 1, -1):
            dz = 1.0 / (x + k - 2)
            dd = c[k - 1] * dz
            d = d + dd
            n = n - dd * dz

        d = d + c[0]
        gg = x + g - 0.5
        f = np.log(gg) + (n / d - g / gg)
        f = f - np.pi / np.tan(np.pi * xx)

    else:
        n = 0.0
        d = 0.0
        for k in range(15, 1, -1):
            dz = 1.0 / (x + k - 2)
            dd = c[k - 1] * dz
            d = d + dd
            n = n - dd * dz

        d = d + c[0]
        gg = x + g - 0.5
        f = np.log(gg) + (n / d - g / gg)

    return f


@vectorize(
    [complex128(complex128, complex128, complex128, complex128, float64)],
    nopython=True
)
def hyp2f1(a, b, c, z, tol=1e-15):
    """
    Compute the Gauss hypergeometric function using series summation
    :param a: complex
    :param b: complex
    :param c: complex
    :param z: complex, |z| < 1
    :param tol: float, tolerance for stopping summation
    :return: hyp2f1(a,b,c,z)
    """

    # Define step (after every step iterations convergence is checked)
    step = 64

    # Compute using hypergeometric function
    alpha = np.zeros(shape=(step,), dtype=np.complex128)
    beta = np.zeros(shape=(step,), dtype=np.complex128)

    n = 0
    alpha[0] = 1
    beta[0] = 1
    for k in range(step - 1):
        alpha[k + 1] = alpha[k] * (a + n) * (b + n) * z / ((c + n) * (n + 1))
        beta[k + 1] = beta[k] + alpha[k + 1]
        n = n + 1

    check_convergence = \
        np.abs(alpha[step - 1] / beta[step - 2]) > tol or \
        np.abs(alpha[step - 2] / beta[step - 3]) > tol or \
        np.abs(alpha[step - 3] / beta[step - 4]) > tol

    while check_convergence:

        alpha[0] = alpha[step - 1] * (a + n) * (b + n) * z / ((c + n) * (n + 1))
        beta[0] = beta[step - 1] + alpha[0]
        n = n + 1

        for k in range(step - 1):
            alpha[k + 1] = alpha[k] * (a + n) * (b + n) * z / ((c + n) * (n + 1))
            beta[k + 1] = beta[k] + alpha[k + 1]
            n = n + 1

        check_convergence = \
            np.abs(alpha[step - 1] / beta[step - 2]) > tol or \
            np.abs(alpha[step - 2] / beta[step - 3]) > tol or \
            np.abs(alpha[step - 3] / beta[step - 4]) > tol

    return beta[step - 1]


@vectorize(
    [complex128(complex128, complex128, float64, float64)],
    nopython=True
)
def hyp2f1_r01(a, b, z, tol=1e-15):
    """
    Compute the Gauss hypergeometric function using series summation for the special case c = a + b, and
    z real with 0 <= z <= 1.0
    :param a: complex
    :param b: complex
    :param z: float, 0 <= z <= 1
    :param tol: float, tolerance for stopping summation
    :return: hyp2f1(a,b,c,z)
    """

    # Define c
    c = a + b

    # Define step (after every step iterations convergence is checked)
    step = 64

    # Compute using hypergeometric function
    alpha = np.zeros(shape=(step,), dtype=np.complex128)
    beta = np.zeros(shape=(step,), dtype=np.complex128)

    if 0 <= z <= 0.9:
        n = 0
        alpha[0] = 1
        beta[0] = 1
        for k in range(step - 1):
            alpha[k + 1] = alpha[k] * (a + n) * (b + n) * z / ((c + n) * (n + 1))
            beta[k + 1] = beta[k] + alpha[k + 1]
            n = n + 1

        check_convergence = \
            np.abs(alpha[step - 1] / beta[step - 2]) > tol or \
            np.abs(alpha[step - 2] / beta[step - 3]) > tol or \
            np.abs(alpha[step - 3] / beta[step - 4]) > tol

        while check_convergence:

            alpha[0] = alpha[step - 1] * (a + n) * (b + n) * z / ((c + n) * (n + 1))
            beta[0] = beta[step - 1] + alpha[0]
            n = n + 1

            for k in range(step - 1):
                alpha[k + 1] = alpha[k] * (a + n) * (b + n) * z / ((c + n) * (n + 1))
                beta[k + 1] = beta[k] + alpha[k + 1]
                n = n + 1

            check_convergence = \
                np.abs(alpha[step - 1] / beta[step - 2]) > tol or \
                np.abs(alpha[step - 2] / beta[step - 3]) > tol or \
                np.abs(alpha[step - 3] / beta[step - 4]) > tol

        return beta[step - 1]

    else:
        zz = 1.0 - z
        logzz = np.log(zz)

        n = 0
        psi1 = digamma(1.0)
        psi2 = digamma(a)
        psi3 = digamma(b)
        alpha[0] = 1.0
        beta[0] = 2 * psi1 - psi2 - psi3 - logzz

        for k in range(step - 1):
            alpha[k + 1] = alpha[k] * (a + n) * (b + n) * zz / ((n + 1) ** 2.0)
            psi1 = psi1 + 1.0 / (n + 1)
            psi2 = psi2 + 1.0 / (n + a)
            psi3 = psi3 + 1.0 / (n + b)
            beta[k + 1] = beta[k] + alpha[k + 1] * (2 * psi1 - psi2 - psi3 - logzz)
            n = n + 1

        check_convergence = \
            np.abs(alpha[step - 1] / beta[step - 2]) > tol or \
            np.abs(alpha[step - 2] / beta[step - 3]) > tol or \
            np.abs(alpha[step - 3] / beta[step - 4]) > tol

        while check_convergence:

            alpha[0] = alpha[step - 1] * (a + n) * (b + n) * zz / ((n + 1) ** 2.0)
            psi1 = psi1 + 1.0 / (n + 1)
            psi2 = psi2 + 1.0 / (n + a)
            psi3 = psi3 + 1.0 / (n + b)
            beta[0] = beta[step - 1] + alpha[0] * (2 * psi1 - psi2 - psi3 - logzz)
            n = n + 1

            for k in range(step - 1):
                alpha[k + 1] = alpha[k] * (a + n) * (b + n) * zz / ((n + 1) ** 2.0)
                psi1 = psi1 + 1.0 / (n + 1)
                psi2 = psi2 + 1.0 / (n + a)
                psi3 = psi3 + 1.0 / (n + b)
                beta[k + 1] = beta[k] + alpha[k + 1] * (2 * psi1 - psi2 - psi3 - logzz)
                n = n + 1

            check_convergence = \
                np.abs(alpha[step - 1] / beta[step - 2]) > tol or \
                np.abs(alpha[step - 2] / beta[step - 3]) > tol or \
                np.abs(alpha[step - 3] / beta[step - 4]) > tol

        return beta[step - 1] * gamma(c) / (gamma(a) * gamma(b))


@vectorize([complex128(complex128, float64, float64)], nopython=True)
def legendre_q(nu, z, tol=1e-15):
    """
    Computes legendre Q function using relation to hypergeometric function
    Q_nu(z) = f * hyp2f1((1+nu)/2, (2+nu)/2, nu + 3/2, 1/z^2)
    where, f = sqrt(pi) * gamma(nu + 1) / (2^ (nu+1) * gamma(nu + 3/2)), and |z| > 1

    :param nu: complex, but not a negative integer
    :param z: float, |z| > 1
    :param tol: float, tolerance for convergence checking
    :return: Q_nu(z)
    """

    # Evaluate a, b, z1 for hypergeometric function
    a = 0.5 * nu + 0.5
    b = 0.5 * nu + 1.0
    c = a + b
    z1 = z ** (-2.0)

    # Compute legendreQ function
    leg = hyp2f1_r01(a, b, z1, tol) * (np.pi ** 0.5) * gamma(1.0 + nu) / (gamma(1.5 + nu) * ((2 * z) ** (1.0 + nu)))
    # leg = hyp2f1(a, b, c, z1, tol) * (np.pi ** 0.5) * gamma(1.0 + nu) / (gamma(1.5 + nu) * ((2 * z) ** (1.0 + nu)))
    return leg


@vectorize([complex128(complex128, float64, float64)], nopython=True)
def legendre_q_v1(nu, z, tol=1e-15):
    """
    Computes legendre Q function using relation to hypergeometric function
    Q_(nu - 1/2)(z) = f * hyp2f1(1/2, 1/2 + nu, 1 + nu, e^(-2 eta)),
    where f = sqrt(pi) * e^(-eta(nu + 0.5)) * gamma(nu + 1/2) / gamma(nu + 1), z = cosh(eta), and |z| > 1

    :param nu: complex, but not a negative integer
    :param z: float, |z| > 1
    :param tol: float, tolerance for convergence checking
    :return: Q_nu(z)
    """

    nu = nu + 0.5

    # Compute eta
    eta = np.arccosh(z)

    # Evaluate a, b, z1 for hypergeometric function
    a = 0.5
    b = 0.5 + nu
    c = a + b
    z1 = np.exp(-2.0 * eta)

    # Compute legendreQ function
    leg = \
        hyp2f1_r01(a, b, z1, tol) * (np.pi ** 0.5) \
        * gamma(0.5 + nu) * np.exp(-1.0 * (0.5 + nu) * eta) / gamma(1.0 + nu)
    # leg = hyp2f1(a, b, c, z1, tol) * (np.pi ** 0.5) \
    #       * gamma(0.5 + nu) * np.exp(-1.0 * (0.5 + nu) * eta) / gamma(1.0 + nu)
    return leg


if __name__ == "__main__":
    """
    Test Gamma function
    """
    print("------------------------------------------------------------------------------------------------------")
    print("Testing Gamma function implementation...")
    print("------------------------------------------------------------------------------------------------------")
    print("\n")

    num = 100000
    _z_real_arr = np.random.uniform(-100, 100, size=(num,))
    _z_imag_arr = np.random.uniform(-100, 100, size=(num,))
    j = complex(0, 1)
    _z_arr = np.zeros(shape=(num,), dtype=np.complex64) + _z_real_arr + j * _z_imag_arr

    t_start = time.time()
    _x1 = gamma(_z_arr)
    t_end = time.time()
    print("Time for computation (average): ", (t_end - t_start) / num, " s")
    print("Computed x (method 1) = ", _x1[0])
    print("\n")

    t_start = time.time()
    _x2 = sp.gamma(_z_arr)
    t_end = time.time()
    print("Time for computation (average): ", (t_end - t_start) / num, " s")
    print("Computed x (Scipy) = ", _x2[0])
    print("\nAll Close check : ", np.allclose(_x1, _x2, atol=0.0, rtol=1e-12))
    print("\n")

    """
    Test Digamma function
    """
    print("------------------------------------------------------------------------------------------------------")
    print("Testing Digamma function implementation...")
    print("------------------------------------------------------------------------------------------------------")
    print("\n")

    num = 100000
    _z_real_arr = np.random.uniform(-100, 100, size=(num,))
    _z_imag_arr = np.random.uniform(-100, 100, size=(num,))
    j = complex(0, 1)
    _z_arr = np.zeros(shape=(num,), dtype=np.complex64) + _z_real_arr + j * _z_imag_arr

    t_start = time.time()
    _x1 = digamma(_z_arr)
    t_end = time.time()
    print("Time for computation (average): ", (t_end - t_start) / num, " s")
    print("Computed x (method 1) = ", _x1[0])
    print("\n")

    t_start = time.time()
    _x2 = sp.digamma(_z_arr)
    t_end = time.time()
    print("Time for computation (average): ", (t_end - t_start) / num, " s")
    print("Computed x (Scipy) = ", _x2[0])
    print("\nAll Close check : ", np.allclose(_x1, _x2, atol=0.0, rtol=1e-12))
    print("\n")

    """
    Gauss' Hypergeometric function test0 (special case c = a + b, 0 <= z <= 1.0)
    """
    print("------------------------------------------------------------------------------------------------------")
    print("Testing Gauss' Hypergeometric function implementation (special case c = a + b, 0 <= z <= 1.0)...")
    print("------------------------------------------------------------------------------------------------------")
    print("\n")

    num = 100000
    _a_arr = np.random.uniform(-10, 10, size=(num,)) + complex(0, 1) * np.random.uniform(-10, 10, size=(num,))
    _b_arr = np.random.uniform(-10, 10, size=(num,)) + complex(0, 1) * np.random.uniform(-10, 10, size=(num,))
    _c_arr = _a_arr + _b_arr
    _z_arr = np.random.uniform(0.0, 1.0, size=(num,))

    t_start = time.time()
    _x1 = hyp2f1(_a_arr, _b_arr, _c_arr, _z_arr, 1e-15)
    t_end = time.time()
    print("Time for computation (average): ", (t_end - t_start) / num, " s")

    t_start = time.time()
    _x2 = hyp2f1_r01(_a_arr, _b_arr, _z_arr, 1e-15)
    t_end = time.time()
    print("Time for computation (average): ", (t_end - t_start) / num, " s")

    rtol = 1e-10
    close_percent = 100 * np.count_nonzero(np.isclose(_x1, _x2, atol=0.0, rtol=rtol)) / num
    print("\nAll Close check (method 1 vs method 2) : % elements close = ", close_percent)
    print("\n")

    _a_arr = np.random.uniform(-10, 10, size=(num,))
    _b_arr = np.random.uniform(-10, 10, size=(num,))
    _c_arr = _a_arr + _b_arr
    _z_arr = np.random.uniform(0.9, 1.0, size=(num,))

    _x1 = hyp2f1(_a_arr, _b_arr, _c_arr, _z_arr, 1e-15)
    _x2 = hyp2f1_r01(_a_arr, _b_arr, _z_arr, 1e-15)
    _x3 = sp.hyp2f1(_a_arr, _b_arr, _c_arr, _z_arr)
    print("Compare against Scipy")

    rtol = 1e-13
    close_percent = 100 * np.count_nonzero(np.isclose(_x1, _x3, atol=0.0, rtol=rtol)) / num
    print("\nAll Close check (method 1 vs Scipy) : % elements close = ", close_percent)

    close_percent = 100 * np.count_nonzero(np.isclose(_x2, _x3, atol=0.0, rtol=rtol)) / num
    print("\nAll Close check (method 2 vs Scipy) : % elements close = ", close_percent)
    print("\n")

    """
    Test LegendreQ function
    """
    print("------------------------------------------------------------------------------------------------------")
    print("Testing Legendre function implementation...")
    print("------------------------------------------------------------------------------------------------------")
    print("\n")
    num = 10
    _nu = 50 + 50j
    _z = 1.0001

    _nu_arr = np.zeros(shape=(num,), dtype=np.complex128) + _nu
    _z_arr = np.zeros(shape=(num,), dtype=np.float64) + _z

    t_start = time.time()
    _x1 = legendre_q(_nu_arr, _z_arr, 1e-15)
    t_end = time.time()
    print("Time for computation (average): ", (t_end - t_start) / num, " s")
    print("Computed x (method 1) = ", _x1[0])

    print("\n")
    t_start = time.time()
    _x2 = legendre_q_v1(_nu_arr, _z_arr, 1e-15)
    t_end = time.time()
    print("Time for computation (average): ", (t_end - t_start) / num, " s")
    print("Computed x (method 2) = ", _x2[0])

    # print("\n")
    # _x3 = -1.0 + 0.5 * _z * np.log(np.abs((1.0 + _z) / (1.0 - _z)))
    # print("Analytical x = ", complex(_x3))

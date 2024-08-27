# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:11:15 2019

@author: rahul
"""
from . import TypeChecker
import copy
import time
import numpy as np


"""
TODO:
1. Add exception handling to the methods
"""


def conjugate_gradient(linear_operator, rhs, x0, niter, printobj=False):

    TypeChecker.check_int_positive(x=niter)

    # Get rhs norm
    fac = np.linalg.norm(x=rhs)
    print("Norm of rhs = ", fac)
    if fac < 1e-15:
        raise ValueError("Norm of rhs < 1e-15. Trivial solution of zero. Scale up the problem.")

    # Scale rhs
    rhs_new = rhs / fac
    x = x0 / fac

    # Define temporary variables
    y = x * 0
    matrix_times_p = x * 0

    # Calculate initial residual, and residual norm
    linear_operator(x, y)
    r = rhs_new - y
    r_norm = np.linalg.norm(x=r)
    if r_norm < 1e-12:
        return x0, [r_norm]
    r_norm_sq = r_norm ** 2

    # Initialize p
    p = copy.deepcopy(r)

    # Initialize residual array, iteration array
    residual = [r_norm]
    if printobj:
        linear_operator(x, y)
        objective = [np.real(0.5 * np.vdot(x, y) - np.vdot(x, rhs_new))]

    # Run CG iterations
    for num_iter in range(niter):

        t1 = time.time()

        if printobj:
            print(
                "Beginning iteration : ", num_iter,
                " , Residual : ", residual[num_iter],
                " , Objective : ", objective[num_iter]
            )
        else:
            print(
                "Beginning iteration : ", num_iter,
                " , Residual : ", residual[num_iter]
            )

        # Compute A*p and alpha
        linear_operator(p, matrix_times_p)
        alpha = r_norm_sq / np.vdot(p, matrix_times_p)

        # Update x0, residual
        x += alpha * p
        r -= alpha * matrix_times_p

        # Calculate beta
        r_norm_new = np.linalg.norm(x=r)
        r_norm_new_sq = r_norm_new ** 2
        beta = r_norm_new_sq / r_norm_sq

        # Check convergence
        if r_norm_new < 1e-12:
            break

        # Update p, residual norm
        p = r + beta * p
        r_norm_sq = r_norm_new_sq

        # Update residual array, iteration array
        residual.append(r_norm_new)
        if printobj:
            linear_operator(x, y)
            objective.append(np.real(0.5 * np.vdot(x, y) - np.vdot(x, rhs_new)))

        t2 = time.time()
        print("Iteration took ", t2 - t1, " s\n")

    # Remove the effect of the scaling
    x = x * fac

    return x, residual


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i,\t residual = %s' % (self.niter, str(rk)))


if __name__ == "__main__":

    # Create a trial problem
    dim = 30

    a = np.zeros((dim, dim), dtype=np.complex64)
    a += np.random.rand(dim, dim).astype(dtype=np.complex64) + 1j * np.random.rand(dim, dim).astype(dtype=np.complex64)
    q, _ = np.linalg.qr(a)
    d = np.random.uniform(low=1, high=2, size=(dim,)).astype(dtype=np.complex64)

    mat = np.dot(np.diag(d), q)
    mat = np.dot(np.conjugate(np.transpose(q)), mat)
    vector_true = np.random.rand(dim).astype(dtype=np.complex64) + 1j * np.random.rand(dim).astype(dtype=np.complex64)
    b = np.dot(mat, vector_true)
    x0 = np.zeros((dim,), dtype=np.complex64)

    # The operator
    def linop(x, y, add_flag=False):
        if not add_flag:
            y *= 0
        y += np.dot(mat, x)

    # Solve
    x0, res = conjugate_gradient(linear_operator=linop, rhs=b, x0=x0, niter=10, printobj=True)

    # Print solution
    print("\nTrue solution:")
    print(vector_true)

    print("\nCG solution:")
    print(x0)

    print("\nDifference:")
    print(vector_true - x0)

    print("\nResiduals:")
    print(res)

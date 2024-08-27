# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 12:11:15 2018

@author: rahul
"""
import numpy as np


# Type checking function
def check(x, expected_type, f=(lambda y: (True, True, ""))):

    str1 = ", ".join([str(types) for types in expected_type])
    if not isinstance(x, expected_type):
        raise TypeError("Object type : " + str(type(x)) + " , Expected types : " + str1)

    type_check, value_check, msg = f(x)

    if not type_check:
        raise TypeError(msg)
    if not value_check:
        raise ValueError(msg)

    return True


# Check if int > 0
def check_int_positive(x):

    str1 = ", ".join([str(types) for types in (int,)])
    if not isinstance(x, int):
        raise TypeError("Object type : " + str(type(x)) + " , Expected type : " + str1)

    if x <= 0:
        raise ValueError("Value of x :" + str(x) + " , Expected value : x > 0")

    return True


# Check if int >= 0
def check_int_nonnegative(x):

    str1 = ", ".join([str(types) for types in (int,)])
    if not isinstance(x, int):
        raise TypeError("Object type : " + str(type(x)) + " , Expected type : " + str1)

    if x < 0:
        raise ValueError("Value of x :" + str(x) + " , Expected value : x >= 0")

    return True


# Check if lb <= int <= ub
def check_int_bounds(x, lb, ub):

    str1 = ", ".join([str(types) for types in (int,)])
    if not isinstance(x, int):
        raise TypeError("Object type : " + str(type(x)) + " , Expected type : " + str1)

    if not lb <= x <= ub:
        raise ValueError("Value of x :" + str(x) + " , Expected value : " + str(lb) + " <= x <= " + str(ub))

    return True


# Check if lb <= int
def check_int_lower_bound(x, lb):

    str1 = ", ".join([str(types) for types in (int,)])
    if not isinstance(x, int):
        raise TypeError("Object type : " + str(type(x)) + " , Expected type : " + str1)

    if not lb <= x:
        raise ValueError("Value of x :" + str(x) + " , Expected value : " + str(lb) + " <= x")

    return True


# Check if int <= ub
def check_int_upper_bound(x, ub):

    str1 = ", ".join([str(types) for types in (int,)])
    if not isinstance(x, int):
        raise TypeError("Object type : " + str(type(x)) + " , Expected type : " + str1)

    if not x <= ub:
        raise ValueError("Value of x :" + str(x) + " , Expected value : " + " x <= " + str(ub))

    return True


# Check if float > 0
def check_float_positive(x):

    str1 = ", ".join([str(types) for types in (float, int)])
    if not isinstance(x, (float, int)):
        raise TypeError("Object type : " + str(type(x)) + " , Expected types : " + str1)

    if x <= 0:
        raise ValueError("Value of x :" + str(x) + " , Expected value : x > 0")

    return True


# Check if lb <= float <= ub
def check_float_bounds(x, lb, ub):

    str1 = ", ".join([str(types) for types in (float, int)])
    if not isinstance(x, (float, int)):
        raise TypeError("Object type : " + str(type(x)) + " , Expected types : " + str1)

    if not lb <= x <= ub:
        raise ValueError("Value of x :" + str(x) + " , Expected value : " + str(lb) + " <= x <= " + str(ub))

    return True


# Check if lb <= float
def check_float_lower_bound(x, lb):

    str1 = ", ".join([str(types) for types in (float, int)])
    if not isinstance(x, (float, int)):
        raise TypeError("Object type : " + str(type(x)) + " , Expected type : " + str1)

    if not lb <= x:
        raise ValueError("Value of x :" + str(x) + " , Expected value : " + str(lb) + " <= x")

    return True


# Check if lb < float
def check_float_strict_lower_bound(x, lb):

    str1 = ", ".join([str(types) for types in (float, int)])
    if not isinstance(x, (float, int)):
        raise TypeError("Object type : " + str(type(x)) + " , Expected type : " + str1)

    if not lb < x:
        raise ValueError("Value of x :" + str(x) + " , Expected value : " + str(lb) + " < x")

    return True


# Check if float <= ub
def check_float_upper_bound(x, ub):

    str1 = ", ".join([str(types) for types in (float, int)])
    if not isinstance(x, (float, int)):
        raise TypeError("Object type : " + str(type(x)) + " , Expected type : " + str1)

    if not x <= ub:
        raise ValueError("Value of x :" + str(x) + " , Expected value : " + " x <= " + str(ub))

    return True


# Check if float < ub
def check_float_strict_upper_bound(x, ub):

    str1 = ", ".join([str(types) for types in (float, int)])
    if not isinstance(x, (float, int)):
        raise TypeError("Object type : " + str(type(x)) + " , Expected type : " + str1)

    if not x < ub:
        raise ValueError("Value of x :" + str(x) + " , Expected value : " + " x < " + str(ub))

    return True


# Check np.ndarray objects
def check_ndarray(x, shape, dtypes, nan_inf=True, lb=None, ub=None):

    str1 = ", ".join([str(types) for types in (np.ndarray,)])
    if not isinstance(x, (np.ndarray,)):
        raise TypeError("Object type : " + str(type(x)) + " , Expected type : " + str1)

    if x.shape != shape:
        raise TypeError("Numpy array shape : " + str(x.shape) + " , Expected shape : " + str(shape))

    if x.dtype not in dtypes:
        raise TypeError("Numpy array dtype : " + str(x.dtype) + " , Expected dtypes : " + str(dtypes))

    if nan_inf is True:
        flag = np.all(np.isfinite(x))
        if not flag:
            raise ValueError("Nan or Inf values detected in numpy array")

    if lb is not None:
        flag = np.all(x >= lb)
        if not flag:
            raise ValueError("Value check condition x >= " + str(lb) + " failed")

    if ub is not None:
        flag = np.all(x <= ub)
        if not flag:
            raise ValueError("Value check condition x <= " + str(ub) + " failed")

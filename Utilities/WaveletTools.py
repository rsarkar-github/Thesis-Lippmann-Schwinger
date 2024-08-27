# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 12:11:15 2019

@author: rahul
"""
from . import TypeChecker
import math
import numpy as np


def ricker_time(freq_peak=10.0, nt=250, dt=0.004, delay=0.05):

    TypeChecker.check_float_positive(x=freq_peak)
    TypeChecker.check_int_positive(x=nt)
    TypeChecker.check_float_positive(x=dt)
    TypeChecker.check(x=delay, expected_type=(float, int))

    t = np.arange(0.0, nt * dt, dt)
    y = (1.0 - 2.0 * (math.pi ** 2) * (freq_peak ** 2) * ((t - delay) ** 2)) \
        * np.exp(-(math.pi ** 2) * (freq_peak ** 2) * ((t - delay) ** 2))
    return t, y


def ricker_wavelet_coefficient(omega, omega_peak, scale=1e6):

    TypeChecker.check_float_positive(x=omega)
    TypeChecker.check_float_positive(x=omega_peak)
    TypeChecker.check_float_positive(x=scale)

    amp = 2.0 * (omega ** 2.0) / (np.sqrt(math.pi) * (omega_peak ** 3.0)) * np.exp(-(omega / omega_peak) ** 2)
    return scale * np.complex64(amp)


def gaussian_spectrum_coefficient(omega, omega_mean, omega_std, scale=1e6):

    TypeChecker.check_float_positive(x=omega)
    TypeChecker.check_float_positive(x=omega_mean)
    TypeChecker.check_float_positive(x=omega_std)
    TypeChecker.check_float_positive(x=scale)

    f = (2 * math.pi * omega_std * omega_std) ** (-0.5)
    f1 = (omega - omega_mean) * (omega - omega_mean) / (2 * omega_std * omega_std)

    amp = scale * f * np.exp(-f1)
    return amp


def apply_spectrum_taper(value, omega, omega_low, omega_high, omega1, omega2):

    TypeChecker.check(x=value, expected_type=(np.complex64, np.complex128, float, int))
    TypeChecker.check_float_positive(x=omega)
    TypeChecker.check_float_positive(x=omega_low)
    TypeChecker.check_float_lower_bound(x=omega_high, lb=omega_low)
    TypeChecker.check_float_bounds(x=omega1, lb=omega_low, ub=omega_high)
    TypeChecker.check_float_bounds(x=omega2, lb=omega1, ub=omega_high)

    if omega <= omega_low or omega >= omega_high:
        return 0
    if omega_low < omega < omega1:
        f = (omega - omega_low) / (omega1 - omega_low)
        return value * f
    if omega1 <= omega <= omega2:
        return value
    if omega2 < omega < omega_high:
        f = (omega_high - omega) / (omega_high - omega2)
        return value * f

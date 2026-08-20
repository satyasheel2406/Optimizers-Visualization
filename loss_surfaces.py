"""
loss_surfaces.py
-----------------
The elongated-bowl test surfaces used in Part A:
    L(x, y) = x^2 + c*y^2 ,  grad L = [2x, 2c*y]
Global minimum is always at the origin. Increasing c increases the
condition number of the Hessian (diag(2, 2c) -> condition number = c),
which is what makes the bowl look narrower and causes SGD to zig-zag.
"""
import numpy as np

SURFACES = {
    "L1: x^2 + 10y^2  (mild)": 10.0,
    "L2: x^2 + 50y^2  (default)": 50.0,
    "L3: x^2 + 100y^2 (steep)": 100.0,
    "L4: x^2 + 1000y^2 (extreme)": 1000.0,
}


def loss(theta, c):
    x, y = theta
    return x ** 2 + c * y ** 2


def grad(theta, c):
    x, y = theta
    return np.array([2 * x, 2 * c * y])


def condition_number(c):
    return c

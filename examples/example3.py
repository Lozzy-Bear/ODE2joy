import numpy as np
import matplotlib.pyplot as plt


def rk2_midpoint(f, x0, ti, tf, dt):
    t = np.arange(ti, tf, dt)
    n = len(t)
    x = np.empty_like(t)
    x[0] = x0
    for idx in range(n-1):
        m1 = f(x[idx], t[idx])
        m2 = f(t[idx] + 0.5 * dt, x[idx] + 0.5 * dt * m1)
        x[idx + 1] = x[idx] + m2 * dt
    return t, x


def rk2_huens(f, x0, ti, tf, dt):
    t = np.arange(ti, tf, dt)
    n = len(t)
    x = np.empty_like(t)
    x[0] = x0
    for idx in range(n-1):
        m1 = f(x[idx], t[idx])
        m2 = f(t[idx] + dt, x[idx] + dt * m1)
        x[idx + 1] = x[idx] + (m2 + m1) / 2 * dt
    return t, x


def rk4(f, x0, ti, tf, dt):
    # todo
    t = np.arange(ti, tf, dt)
    n = len(t)
    x = np.empty_like(t)
    x[0] = x0
    for idx in range(n-1):
        m1 = f(x[idx], t[idx])
        m2 = f(t[idx] + 0.5 * dt, x[idx] + 0.5 * dt * m1)
        x[idx + 1] = x[idx] + m2 * dt
    return t, x

import numpy as np


def mv_euler_method(f, r0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    n = len(t)
    x = np.empty_like(t)
    y = np.empty_like(t)
    x[0] = x0
    y[0] = y0
    for idx in range(n-1):
        kx = fx(x[idx], y[idx])
        ky = fy(x[idx], y[idx], t[idx])
        x[idx + 1] = x[idx] + kx*dt
        y[idx + 1] = y[idx] + ky*dt
    return t, x, y
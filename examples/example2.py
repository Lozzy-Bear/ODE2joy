import numpy as np


def harmonic_oscillator(t, r):
    omega = 1
    x = r[0]
    v = r[1]
    fx = v
    fy = -(omega ** 2) * x
    k = np.array([fx, fy])
    return k


def euler_method_mv(f, r0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    n = len(t)
    r = np.empty_like((r, t))
    r[0] = x0
    for idx in range(n-1):
        kx = fx(x[idx], y[idx])
        ky = fy(x[idx], y[idx], t[idx])
        x[idx + 1] = x[idx] + kx*dt
        y[idx + 1] = y[idx] + ky*dt

    return t, r


if __name__ == '__main__':

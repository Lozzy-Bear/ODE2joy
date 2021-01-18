import numpy as np
import matplotlib.pyplot as plt


def euler_method(x0, ti, tf, dt, f):
    t = np.arange(ti, tf, dt)
    n = len(t)
    x = np.empty_like(t)
    x[0] = x0
    for idx in range(n-1):
        m = f(x[idx], t[idx])
        x[idx + 1] = x[idx] + m*dt
    return t, x


def mv_euler_method(x0, y0, ti, tf, dt, fx, fy):
    t = np.arange(ti, tf, dt)
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


if __name__ == '__main__':
    x0 = 1
    y0 = 1
    ti = 0
    tf = 10
    dt = 0.01
    fx = lambda a, b: a * b - a
    fy = lambda a, b, c: -a * b + np.sin(2*c)
    t, x, y = mv_euler_method(x0, y0, ti, tf, dt, fx, fy)
    plt.plot(t, x)
    plt.plot(t, y)
    plt.show()

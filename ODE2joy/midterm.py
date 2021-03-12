import numpy as np
import matplotlib.pyplot as plt


def relaxation(target):
    m = np.zeros(3)
    err = 1e16
    n = 0
    m[1] = 1.0
    m[2] = np.sqrt(np.exp(-(m[1]+1)**2)/2)
    while err > target:
        m[0] = m[1]
        m[1] = m[2]
        m[2] = np.sqrt(np.exp(-(m[1]+1)**2)/2)
        err = np.abs((m[1] - m[2])**2 / (2*m[1] - m[0] - m[2]))
        n += 1
        print(m[2], n)
    return m[2], n+2


relaxation(1e-12)
# Roots iterations
# 1) 0.38796156772789014
# 2) 0.2698761181274636
# 3) 0.31573050296885413
# 4) 0.29755774603845797


def secant(f, r0, x0, target, p):
    # Multivarbiale Newton-Raphson method
    x, r = rk4(f, r0, x0, xf, dx)
    err = target + 1
    while err > target:
        df = jacobian(f, x[:, -1], p) # feed last values for rk4
        dx = np.linalg.solve(df, f(x0)) # LU decomp
        x0 -= dx
        err = np.max(np.abs(dx))

    return x0

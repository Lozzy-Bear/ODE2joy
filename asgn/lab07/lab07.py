import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.constants as const


def diode_bridge(v):
    resistor = [1000, 4000, 3000, 2000]
    current = 3e-9
    vt = 0.05
    vp = 5.0
    f1 = (v[0]-vp)/resistor[0] + v[0]/resistor[1] + current*np.exp((v[0]-v[1])/vt - 1)
    f2 = (vp-v[1])/resistor[2] - v[1]/resistor[3] + current*np.exp((v[0]-v[1])/vt - 1)

    return np.array([f1, f2])


def newton_mv(f, x0, target, p):
    # Multivarbiale Newton-Raphson method
    err = target + 1
    while err > target:
        df = jacobian(f, x0, p)
        dx = np.linalg.solve(df, f(x0))
        #dx = la.lu_solve(la.lu_factor(df), f(x0))
        x0 -= dx
        err = np.max(np.abs(dx))

    return x0


def func(x):
    f1 = 1 * (x[0] - x[0] * x[1])
    f2 = -1 * (x[1] - x[0] * x[1])

    return np.array([f1, f2], dtype=np.float)


def jdf(f, x, p):
    J = np.zeros([len(x), len(x)], dtype=np.float)

    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()

        x1[i] += p
        x2[i] -= p

        f1 = f(x1)
        f2 = f(x2)

        J[:, i] = (f1 - f2) / (2 * p)

    return J


def jacobian(f, x, p):
    df = np.zeros((len(x), len(x)))
    fu = f(x)
    for i in range(len(x)):
            xtemp = x.copy()
            xtemp[i] += p
            df[:, i] = (f(xtemp) - fu) / p
    return df


def test_system(x):
    f1 = np.exp(-1*np.exp(x[0]+x[1])) - x[1]*(1+x[0]**2)
    f2 = x[0]*np.cos(x[1]) - x[1]*np.sin(x[0]) + 0.5
    return np.array([f1, f2])


def schrodinger(r, x):
    en = 14e3 * const.electron_volt
    m = const.electron_mass
    kpsi = r[1]
    kphi = 2 * m / const.hbar**2 * (potential_well(x) - en) * r[0]
    return np.array([kpsi, kphi])


def potential_well(x):
    b1 = 0
    b2 = 1e-11
    if b1 <= x <= b2:
        v = 0
    else:
        v = np.inf
    return v


def energies(n, m, l):
    return n**2 * np.pi**2 * const.hbar**2 / 2 / m / l**2


def rk4(f, r0, t0, tf, dt):
    """
    Multivaribale 4th-order Runge-Kutta

    Parameters
    ----------
    f : function np
        a function or array of functions handle that defines the ODE
    r0 : float np.array
        an array containing the initial condition of each dependent variable
    t0 : float
        the initial time
    tf : float
        the final time
    dt : float
        the time step

    Returns
    -------
    t : float np.array
        an array of time values corresponding to the solution
    r : float np.array
        a two dimensional array containing the solution for each variable at each time step
    """

    # Initialize variables
    t = np.arange(t0, tf, dt)
    r = np.zeros((len(r0), len(t)))
    r[:, 0] = r0

    # Iterate method
    for idx in range(len(t) - 1):
        k1 = f(r[:, idx], t[idx])
        k2 = f(r[:, idx] + 0.5*dt*k1, t[idx] + 0.5*dt)
        k3 = f(r[:, idx] + 0.5*dt*k2, t[idx] + 0.5*dt)
        k4 = f(r[:, idx] + dt*k3, t[idx] + dt)
        k = k1/6 + k2/3 + k3/3 + k4/6
        r[:, idx + 1] = r[:, idx] + k * dt

    return t, r


if __name__ == '__main__':
    # v0 = np.array([1., 0.])
    # x = newton_mv(diode_bridge, v0, 0.01, 1e-10)
    # print(x[0]-x[1])
    #
    # x = np.array([1, 2], dtype=np.float)
    # print(jdf(func, x, 1e-10))
    # print(jacobian(func, x, 1e-10))

    # Question 2
    #print(newton_mv(test_system, np.array([np.pi, np.pi]), 1e-4, 1e-10))

    # Question 3
    x, r = rk4(schrodinger, np.array([0, 1]), 0, 1e-11, 1e-11/1000)
    plt.figure()
    plt.plot(x, r[0])
    plt.show()

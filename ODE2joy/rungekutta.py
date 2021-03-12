import numpy as np
import matplotlib.pyplot as plt


def rk1_eulers(f, r0, t0, tf, dt):
    """
    Multivaribale Eulers Method
    General Form
        - dx/dt = fx(x,y,...,t), dy/dt = fy(x,y,...,t)
        - dr/dt = fr(r,t) where r = (x, y,...)
        - fr(r,t) = (fx(r,t), fy(r,t), ...)

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
        k = f(r[:, idx], t[idx])
        r[:, idx + 1] = r[:, idx] + k * dt

    return t, r


def rk2_midpoint(f, r0, t0, tf, dt):
    """
    Multivaribale Midpoint method 2nd-order Runge-Kutta

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
        r[:, idx + 1] = r[:, idx] + k2 * dt

    return t, r


def rk2_huens(f, r0, t0, tf, dt):
    """
    Multivaribale Huen's method 2nd-order Runge-Kutta

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
        k2 = f(r[:, idx] + dt*k1, t[idx] + dt)
        k = (k1 + k2) / 2
        r[:, idx + 1] = r[:, idx] + k * dt

    return t, r


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
        print('eawe')
        print(k1, k2, k3, k4)
        k = k1/6 + k2/3 + k3/3 + k4/6
        r[:, idx + 1] = r[:, idx] + k * dt

    return t, r


def rk4_adaptive(f, r0, t0, tf, tol):
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
    tol : float
        target accuracy

    Returns
    -------
    t : float np.array
        an array of time values corresponding to the solution
    r : float np.array
        a two dimensional array containing the solution for each variable at each time step
    dt_steps : float np.array
        adaptive steps taken
    """

    def rk4_slope(f, r, t, dt):
        k1 = f(r, t)
        k2 = f(r + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(r + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(r + dt * k3, t + dt)

        k = k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
        r = r + k * dt

        return r

    # Initialize variables
    dt = (tf - t0) / 1e5
    t = np.arange(t0, tf, dt)
    dt_steps = np.zeros_like(t)
    r = np.zeros((len(r0), len(t)))
    r[:, 0] = r0

    # Iterate method
    idx = 0
    while t[idx] < tf:
        # Two steps of dt
        r_temp = rk4_slope(f, r[:, idx], t[idx], dt)
        x1 = rk4_slope(f, r_temp, t[idx] + dt, dt)
        # One step of 2dt
        x2 = rk4_slope(f, r[:, idx], t[idx], 2 * dt)
        # Check the accuracy
        a = np.sqrt(x1[0] ** 2 + x1[1] ** 2 + x1[2] ** 2)
        b = np.sqrt(x2[0] ** 2 + x2[1] ** 2 + x2[2] ** 2)
        rho = 30 * tol * dt / np.abs(a - b)
        # Decrease dt
        if rho < 1:
            dt = dt * rho ** 0.25
            r_temp = rk4_slope(f, r[:, idx], t[idx], dt)
            x1 = rk4_slope(f, r_temp, t[idx] + dt, dt)
            r[:, idx + 1] = x1
            t[idx + 1] = t[idx] + 2 * dt
            dt_steps[idx] = dt
        # Increase dt
        else:
            r[:, idx + 1] = x1
            t[idx + 1] = t[idx] + 2 * dt
            dt_steps[idx] = dt
            if dt * rho ** 0.25 > 2 * dt:
                dt = 2 * dt
            else:
                dt = dt * rho ** 0.25
        idx += 1

    return t[0:idx], r[:, 0:idx], dt_steps[0:idx]


def leapfrog(f, r0, t0, tf, dt):
    """
    Leapfrog method

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
    t = np.arange(t0, tf, dt)
    r = np.zeros((len(r0), len(t)))
    m = np.zeros_like(r)  # midpoint steps tracking
    r[:, 0] = r0
    m[:, 0] = r[:, 0] + 0.5 * dt * f(r[:, 0], t[0])
    for i in range(len(t) - 1):
        r[:, i + 1] = r[:, i] + dt * f(m[:, i], t[i] + 0.5 * dt)
        m[:, i + 1] = m[:, i] + dt * f(r[:, i + 1], t[i] + dt)
    return t, r


def verlet(f, r0, t0, tf, dt):
    """
    Verlet method is: specific to the problem, requires dx/dt=v and dv/dt only be a function of x.

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
        ri : float np.array
            combined xi, v, vi (optional on half interval velocity)

    Notes
    -----
        xi : float np.array
            Positions at half step intervals
        vi : float np.array
            Velocity at half step intervals. (optionally returned)
        v : float np.array
            Velocity at off set 1/2 step intervals. (normally returned)
    """

    t = np.arange(t0, tf, dt)
    r = np.zeros((len(r0)+1, len(t)))
    r[0, 0] = r0[0]
    k = f(r[0:2, 0], t[0])
    r[1, 0] = r[1, 0] + 0.5 * dt * k[1]
    r[2, 0] = r0[1]
    for i in range(len(t) - 1):
        r[0, i+1] = r[0, i] + dt * r[1, i]
        k = f(r[0:2, i+1], t[i] + dt)
        r[1, i+1] = r[1, i] + dt * k[1]
        r[2, i+1] = r[1, i] + 0.5 * dt * k[1]

    return t, r

def harmonic(r, t):
    """
    Generic simple harmonic oscillator
    """
    k = 2
    m = 1
    x = r[0]
    v = r[1]
    fx = v
    fy = -k * x**3/m
    return np.array([fx, fy])


if __name__ == '__main__':
    print(rk4(harmonic, np.array([1, 0]), 0, 2, 1))






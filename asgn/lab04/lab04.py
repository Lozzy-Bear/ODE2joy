import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import ode


def rk4_adaptive(f, r0, t0, tf, tol=10e-5):
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
        
    def rk4_slope(f, r, t, dt):
        k1 = f(r, t)
        k2 = f(r + 0.5*dt*k1, t + 0.5*dt)
        k3 = f(r + 0.5*dt*k2, t + 0.5*dt)
        k4 = f(r + dt*k3, t + dt)
        k = k1/6 + k2/3 + k3/3 + k4/6
        r = r + k * dt

        return r

    # Initialize variables
    dt = (tf - t0) / 10e5
    t = np.arange(t0, tf, dt)
    r = np.zeros((len(r0), len(t)))
    r[:, 0] = r0

    # Iterate method
    idx = 0
    while t[idx] < tf:
        # Two steps of dt
        r_temp = rk4_slope(f, r[:, idx], t[idx], dt)
        x1 = rk4_slope(f, r_temp, t[idx]+dt, dt)
        # One step of 2dt
        x2 = rk4_slope(f, r[:, idx], t[idx], 2*dt)
        # Check the accuracy
        rho = tol / (1/30 * np.abs(x1[0] - x2[0]))
        # Decrease dt
        if rho < 1:
            dt = dt * rho**0.25
            r_temp = rk4_slope(f, r[:, idx], t[idx], dt)
            x1 = rk4_slope(f, r_temp, t[idx] + dt, dt)
            r[:, idx + 1] = x1
            t[idx + 1] = t[idx] + 2 * dt
        # Increase dt
        else:
            r[:, idx+1] = x1
            t[idx+1] = t[idx] + 2*dt
            if dt*rho**0.25 > 2*dt:
                dt = 2*dt
            else:
                dt = dt*rho**0.25
        idx += 1
    
    return t[0:idx], r[:, 0:idx]


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


def non_linear_ode(r, t):
    """
    A non-linear ODE analytically solveable by x(t)=sin(sqrt(k)*t^2)
    """
    kappa = 1.0

    x = r[0]
    v = r[1]

    fx = v
    fv = v/t - 4*kappa*x*t**2
    return np.array([fx, fv])


def orbital_burn(r, t):
    mu = 398_600  # [km^3/s^2]
    thrust = 10  # [kN]
    g = 0.00981  # [km/s^]
    isp = 300   # [s]

    m = r[6]

    alpha = -mu/np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)**3
    beta = thrust/m/np.sqrt(r[3]**2 + r[4]**2 + r[5]**2)

    fm = np.array([-thrust/g/isp])
    fr = r[3:5+1]
    fv = alpha * r[0:2+1] + beta * r[3:5+1]

    return np.concatenate((fr, fv, fm))


def orbital_motion(r, t):
    mu = 398_600    # [km^3/s^2]
    thrust = 0  # [N]
    g = 0.00981  # [km/s^]
    isp = 300   # [s]

    m = r[6]
    alpha = -mu/np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)**3
    beta = thrust/(m * np.sqrt(r[3]**2 + r[4]**2 + r[5]**2))

    fm = np.array([-thrust/(g*isp)])
    fr = r[3:5+1]
    fv = alpha * r[0:2+1] + beta * r[3:5+1]

    return np.concatenate((fr, fv, fm))


if __name__ == '__main__':
    f = non_linear_ode
    t0 = 0.1
    tf = 10
    dt = 0.01
    kappa = 1
    ta = np.arange(t0, tf, dt)
    xa = np.sin(np.sqrt(kappa)*ta**2)
    va = 2 * np.sqrt(kappa)*ta*np.cos(np.sqrt(kappa)*ta**2)
    r0 = [xa[0], va[0]]
    t, r = rk4_adaptive(f, r0, t0, tf, 10e-8)
    t2, r2 = rk4(f, r0, t0, tf, dt)
    plt.figure()
    plt.plot(ta, va, '--k')
    plt.plot(t, r[1], 'r')
    plt.plot(t2, r2[1], 'b')
    plt.title('Analytical vs. Adaptive RK4 Solution')
    plt.xlabel('Time [s]')
    plt.ylabel('v(t)')
    plt.legend(['Analytical', 'Adaptive RK4', 'Discrete Step RK4'])
    plt.savefig('q2a_nonlinear.png')

    plt.figure()
    v = 2 * np.sqrt(kappa)*t*np.cos(np.sqrt(kappa)*t**2)
    v2 = 2 * np.sqrt(kappa) * t2 * np.cos(np.sqrt(kappa) * t2 ** 2)
    plt.semilogy(t2, np.abs(r2[1] - v2), 'b')
    plt.semilogy(t, np.abs(r[1] - v), 'r')
    plt.title('Log of Absolute Error of Adaptive RK4 vs. Analytical')
    plt.xlabel('Time [s]')
    plt.ylabel('Log of Absolute Error,  log|εv(t)|')

    plt.savefig('q2b_nonlinear.png')
    plt.show()

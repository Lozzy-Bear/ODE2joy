import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import ode



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
        # a = x1[0]
        # b = x2[0]
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


def set_axes_equal(ax):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def fit_func(x, a, b, c):
    return a*x**2 + b*x + c


if __name__ == '__main__':
    # Question 2
    # f = non_linear_ode
    # t0 = 0.1
    # tf = 10
    # dt = 0.01
    # kappa = 1
    # ta = np.arange(t0, tf, dt)
    # xa = np.sin(np.sqrt(kappa)*ta**2)
    # va = 2 * np.sqrt(kappa)*ta*np.cos(np.sqrt(kappa)*ta**2)
    # r0 = [xa[0], va[0]]
    # t, r, steps = rk4_adaptive(f, r0, t0, tf, 10e-8)
    # t1, r1, steps1 = rk4_adaptive(f, r0, t0, tf, 10e-4)
    # t2, r2 = rk4(f, r0, t0, tf, dt)
    # plt.figure(figsize=[12, 5])
    # plt.subplot(121)
    # plt.plot(ta, xa, '--k')
    # plt.plot(t, r[0], 'r')
    # plt.title('Adaptive RK4 Solution')
    # plt.xlabel('Time [s]')
    # plt.ylabel('x(t)')
    # plt.legend(['Analytical', 'Adaptive RK4'], loc='lower left')
    # plt.subplot(122)
    # plt.plot(t, steps, '--k')
    # plt.title('Time Steps Taken vs. Time')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Time Step [s]')
    # plt.savefig('q2a_nonlinear.png')
    #
    # plt.figure()
    # plt.semilogy(t, np.abs(np.sin(np.sqrt(kappa)*t**2) - r[0]), 'b',
    #              label=f'Adaptive, tol=10e-8, steps={len(t)}')
    # plt.semilogy(t1, np.abs(np.sin(np.sqrt(kappa)*t1**2) - r1[0]), '--b',
    #              label=f'Adaptive, tol=10e-4, steps={len(t1)}')
    # plt.semilogy(t2, np.abs(np.sin(np.sqrt(kappa)*t2**2) - r2[0]), 'r',
    #              label=f'Fixed, dt=0.01, steps={len(t2)}')
    # plt.title('Log of Absolute Error of Various Methods')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Log of Absolute Error,  log|εv(t)|')
    # plt.legend(loc='lower right')
    #
    # plt.savefig('q2b_nonlinear.png')
    # plt.show()

    # Question 3
    f = orbital_motion
    r0 = [500+6378, 0, 0, 0, 10, 0, 2000]
    t0 = 0
    tf = 700*60
    tol = 1e-7
    #t, r, steps = rk4_adaptive(f, r0, t0, tf, tol)
    t1, r1, steps1 = rk4_adaptive(orbital_motion, r0, t0, tf, tol)
    r0 = r1[:, -1]
    r0[5] = 2
    t2, r2, steps2 = rk4_adaptive(orbital_motion, r0, t1[-1], t1[-1] + tf*5, tol)
    t = np.append(t1, t2)
    r = np.append(r1, r2, axis=1)
    steps = np.append(steps1, steps2)

    print(len(t))

    fig = plt.figure(figsize=[12, 10])

    ax1 = fig.add_subplot(2, 2, 1)
    earth = plt.Circle((0, 0), 6378, color='g')
    ax1.add_patch(earth)
    ax1.plot(r[0], r[1], 'k')
    ax1.axis('equal')
    ax1.set_title('Molniya Orbital Motion')
    ax1.set_xlabel('[km]')
    ax1.set_ylabel('[km]')
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(np.sqrt(r[0]**2 + r[1]**2) - 6378, steps)
    ax2.set_title('Time step vs. Altitude')
    ax2.set_xlabel('Altitude [km]')
    ax2.set_ylabel('Time Step [s]')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(t, np.sqrt(r[0]**2 + r[1]**2) - 6378)
    ax3.set_title('Altitude vs. Time')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Altitude [km]')
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    ax4 = fig.add_subplot(2, 2, 2, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6378 * np.outer(np.cos(u), np.sin(v))
    y = 6378 * np.outer(np.sin(u), np.sin(v))
    z = 6378 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax4.plot(r[0], r[1], r[2], 'k')
    ax4.plot_surface(x, y, z, color='g')
    ax4.set_title('Molniya Orbital Motion 3D')
    ax4.set_xlabel('[km]')
    ax4.set_ylabel('[km]')
    ax4.set_zlabel('[km]')
    set_axes_equal(ax4)

    plt.savefig('q3c_plane_change.png')
    plt.show()

    # Question 3.d
    # f = orbital_motion
    # t0 = 0
    # tf = 5*700*60
    # tol = 1e-7
    #
    # fig = plt.figure(figsize=[12, 5])
    # ax1 = fig.add_subplot(1, 2, 1)
    # earth = plt.Circle((0, 0), 6378, color='g')
    # ax1.add_patch(earth)
    # ax1.axis('equal')
    # ax1.set_title('Molniya Orbital Motion')
    # ax1.set_xlabel('[km]')
    # ax1.set_ylabel('[km]')
    # plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    #
    # ax2 = fig.add_subplot(1, 2, 2)
    # #ax2.axis('equal')
    # ax2.set_title('Molniya Orbital Motion')
    # ax2.set_xlabel('Pergigee Velocity [km/s]')
    # ax2.set_ylabel('Apogee Altitude [km]')
    # plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    #
    # vels = np.arange(8.0, 10.5 + 0.1, 0.1)
    # alts = np.zeros_like(vels)
    # for i in range(len(vels)):
    #     r0 = [500 + 6378, 0, 0, 0, vels[i], 0, 2000]
    #     t, r, steps = rk4_adaptive(f, r0, t0, tf, tol)
    #     result = np.min(r[0])
    #     alts[i] = np.abs(result) - 6378
    #     ax1.plot(r[0], r[1])
    #     ax1.scatter(result, 0)
    #
    # ax2.plot(vels, alts, 'b', label='')
    # plt.savefig('q3d_pergiapo.png')
    # plt.show()

import matplotlib.pyplot as plt
import numpy as np


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



def euler_method_mv(f, r0, t0, tf, dt):
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


def lorenz(r, t):
    """
    The classically famous set of Lorenz equations ODE
    """
    sigma = 10
    rho = 28
    beta = 8 / 3

    x = r[0]
    y = r[1]
    z = r[2]

    fx = sigma * (y - x)
    fy = x * (rho - z) - y
    fz = x * y - beta * z

    return np.array([fx, fy, fz])


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
    # Question 2
    # f = lorenz
    # r0 = [0, 1, 0]
    # t, r = rk4(f, r0, 0, 50, 0.001)
    # te, re = euler_method_mv(f, r0, 0, 50, 0.001)
    # plt.figure()
    # for i in range(len(r0)):
    #     plt.plot(t, r[i])
    #     plt.plot(te, re[i], '--')
    # plt.title('Lorenz')
    # plt.ylabel('Amplitude')
    # plt.xlabel('Time [s]')
    # plt.legend(('x(t) Euler', 'x(t) RK4', 'y(t) Euler', 'y(t) RK4', 'z(t) Euler', 'z(t) RK4',))
    # plt.savefig('q2_lorenz.png')
    #
    # plt.figure()
    # for i in range(len(r0)):
    #     plt.plot(t, r[i] - re[i])
    # plt.title('Lorenz')
    # plt.ylabel('Amplitude')
    # plt.xlabel('Time [s]')
    # plt.legend(('Delta x(t)', 'Delta y(t)', 'Delta z(t)'))
    # plt.savefig('q2_delta_lorenz.png')
    #
    # plt.figure()
    # plt.plot(t, np.abs(np.sqrt(r[0]**2 + r[1]**2 + r[2]**2) - np.sqrt(re[0]**2 + re[1]**2 + re[2]**2)))
    # plt.title('Lorenz Magnitude Difference')
    # plt.ylabel('Amplitude')
    # plt.xlabel('Time [s]')
    # plt.savefig('q2_sqr_lorenz.png')
    # plt.show()

    # Question 3
    # f = non_linear_ode
    # t0 = 0.1
    # tf = 30
    # dt = 0.001
    # kappa = 1
    # ta = np.arange(t0, tf, dt)
    # xa = np.sin(np.sqrt(kappa)*ta**2)
    # va = 2 * np.sqrt(kappa)*ta*np.cos(np.sqrt(kappa)*ta**2)
    # r0 = [xa[0], va[0]]
    # t, r = rk4(f, r0, t0, tf, dt)
    # plt.figure()
    # plt.plot(ta, va, '--')
    # plt.plot(t, r[1])
    # plt.title('Analytical vs. RK4 Solution for dt=0.001')
    # plt.xlabel('Time [s]')
    # plt.ylabel('v(t)')
    # plt.legend(['Analytical', 'RK4'])
    # plt.savefig('q3b_nonlinear.png')
    #
    # f = non_linear_ode
    # t0 = 0.1
    # tf = 30
    # dt = 0.01
    # kappa = 1
    # ta = np.arange(t0, tf, dt)
    # xa = np.sin(np.sqrt(kappa)*ta**2)
    # va = 2 * np.sqrt(kappa)*ta*np.cos(np.sqrt(kappa)*ta**2)
    # r0 = [xa[0], va[0]]
    # t, r = rk4(f, r0, t0, tf, dt)
    # plt.figure()
    # plt.plot(t, np.abs(r[1] - va))
    # plt.title('Absolute Error of RK4 vs. Analytical for dt=0.01')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Absolute Error,  |εv(t)|')
    # plt.savefig('q3c_nonlinear.png')
    # plt.show()

    # Question 4.c
    # f = orbital_motion
    # r0 = [480+6378, 0, 0, 0, 7.7102, 0, 2000]
    # t, r = rk4(f, r0, 0, 100*60, 0.1)
    # plt.figure()
    # plt.plot(r[0], r[1])
    # plt.show()

    # Question 4.d
    # t_burn, r_burn = rk4(orbital_burn, [480+6378, 0, 0, 0, 7.7102, 0, 2000], 0, 261.11, 0.005)
    # t_coast, r_coast = rk4(orbital_motion, r_burn[:, -1], 0, 250*60, 0.5)
    # plt.figure()
    # plt.plot(r_burn[0], r_burn[1], 'r')
    # plt.plot(r_coast[0], r_coast[1], 'b')
    # plt.show()
    # print(f'Mass used: {2000 - r_coast[6, -1]}')
    # print(f'Velocity change: {np.sqrt(r_burn[3, -1]**2 + r_burn[4, -1]**2 + r_burn[5, -1]**2) - 7.7102}')
    # print(f'New apogee: {np.max(np.sqrt(r_coast[0]**2 + r_coast[1]**2))}')

    # Question 4.e
    t, r = rk4(orbital_motion, [480+6378, 0, 0, 0, 7.7102, 0, 2000], 0, 100 * 60, 0.1)
    t_burn, r_burn = rk4(orbital_burn, [480+6378, 0, 0, 0, 7.7102, 0, 2000], 0, 261.11, 0.005)
    t_coast, r_coast = rk4(orbital_motion, r_burn[:, -1], 0, 250*60, 0.5)
    t_burn2, r_burn2 = rk4(orbital_burn, r_coast[:, 17336], 0, 118.88, 0.005)
    t_coast2, r_coast2 = rk4(orbital_motion, r_burn2[:, -1], 0, 600 * 60, 0.5)
    plt.figure()
    plt.plot(r[0], r[1], 'k')
    plt.plot(r_burn[0], r_burn[1], 'r')
    plt.plot(r_coast[0, 0:17336], r_coast[1, 0:17336], 'b')
    plt.plot(r_burn2[0], r_burn2[1], 'm')
    plt.plot(r_coast2[0], r_coast2[1], 'g')
    plt.axis('equal')
    plt.savefig('q4e_hohmann.png')
    plt.show()
    print(f'Mass used burn 1: {2000 - r_burn[6, -1]}')
    print(f'Mass used burn 1: {2000 - r_burn2[6, -1]}')
    print(f'Velocity change burn 1: {np.sqrt(r_burn[3, -1]**2 + r_burn[4, -1]**2 + r_burn[5, -1]**2) - 7.7102}')
    print(f'Velocity change burn 2: {np.sqrt(r_burn2[3, -1]**2 + r_burn2[4, -1]**2 + r_burn2[5, -1]**2) - np.sqrt(r_coast[3, 17336]**2 + r_coast[4, 17336]**2 + r_coast[5, 17336]**2)}')
    print(f'New apogee: {np.argmax(np.sqrt(r_coast[0]**2 + r_coast[1]**2))}')



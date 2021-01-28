import matplotlib.pyplot as plt
import numpy as np


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


def harmonic_oscillator(r, t):
    """
    Generic simple harmonic oscillator
    """
    omega = 1
    x = r[0]
    v = r[1]
    fx = v
    fy = -(omega ** 2) * x
    return np.array([fx, fy])


def simple_ode(r, t):
    """
    Simple ODE example
    """
    x = r[0]
    y = r[1]
    fx = x * y - x
    fy = -x * y + np.sin(2 * t)
    return np.array([fx, fy])


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


def sir_model(r, t):
    """
    The epidemic spread of a contagious disease has been modelled with substantial success with a set of coupled
    non-linear ODEs in three dependent variables that describe:
        - s(t) number susceptible
        - i(t) number infected
        - r(t) number recovered
        Note: vaccines lower s(t) and/or lower r(t) if it is both preventive and curative
        Note: r(t) is equally for dead people and immune people
        - beta is the disease spread rate
        - gamma how quickly infected person recovers/dies
        - n = s + i + r the constant total population
    """
    n = 1_000_000
    beta = 8.0  # 8.0 per month
    gamma = 8.6  # 4.3 per month

    susceptible = r[0]
    infected = r[1]
    recovered = r[2]

    fs = -beta / n * infected * susceptible
    fi = beta / n * infected * susceptible - gamma * infected
    fr = gamma * infected

    return np.array([fs, fi, fr])


def sir_model_vaccine(r, t):
    """
    The epidemic spread of a contagious disease has been modelled with substantial success with a set of coupled
    non-linear ODEs in three dependent variables that describe:
        - s(t) number susceptible
        - i(t) number infected
        - k(t) number treated
        - r(t) number recovered
        - beta is the disease spread rate
        - alpha is the vaccine distribution rate
        - gamma_i how quickly infected person recovers/dies
        - gamma_k how quickly infected person recovers from treatment
        - f portion of infected able to get treated
        - sigma transmission reduction from treatment.
        - n = s + i + r the constant total population
    """
    n = 1_000_000 # Total initial population
    beta = 8.0  # 8.0 per month
    gamma_untreated = 4.3  # 4.3 per month
    gamma_treated = 4.3  # Recovery rate of treated people
    sigma = 0.5  # Transmission reduction due of the anti-viral treatment
    tv = 1.5  # Time when vaccine is introduced
    ta = 1.5

    susceptible = r[0]
    infected_untreated = r[1]
    infected_treated = r[2]
    recovered = r[3]

    # Before vaccine introduction
    f0 = 0
    alpha = 0

    # Point after anti-viral treatment is introduced.
    if t >= ta:
        f0 = 0.75  # Fraction of Infected Untreated people who will get antiviral treatment
    # Point after vaccine in introduced
    if t >= tv:
        alpha = 1.0

    fs = -beta * susceptible * (infected_untreated + sigma * infected_treated) / n - alpha * susceptible #* (1 - (infected_untreated + infected_treated + recovered) / n)
    fiu = beta * susceptible * (1 - f0) * (infected_untreated + sigma * infected_treated) / n - gamma_untreated * infected_untreated
    fit = beta * susceptible * f0 * (infected_untreated + sigma * infected_treated) / n - gamma_treated * infected_treated
    fr = gamma_untreated * infected_untreated + gamma_treated * infected_treated + alpha * susceptible #* (1 - (infected_untreated + infected_treated + recovered) / n)

    return np.array([fs, fiu, fit, fr])


def sir_model_zombie(r, t):
    """
    Zombie model -- RAAHAhhahhahhaGaahaha bRaInS!
    """
    n = 1_000_000
    beta = 1.0
    alpha = 4.0
    gamma = 2.0

    susceptible = r[0]
    infected = r[1]
    recovered = r[2]

    fs = -beta / n * infected * susceptible
    fi = (beta - alpha) / n * infected * susceptible + gamma * recovered
    fr = alpha / n * susceptible * infected - gamma * recovered

    return np.array([fs, fi, fr])


if __name__ == '__main__':
    # Question 1
    # f = simple_ode
    # r0 = [1, 1]
    # t, r = euler_method_mv(f, r0, 0, 10, 0.001)
    # plt.figure()
    # for i in range(len(r0)):
    #     plt.plot(t, r[i])
    # plt.title('Simple ODE')
    # plt.ylabel('Amplitude')
    # plt.xlabel('Time [s]')
    # plt.legend(('x(t)', 'y(t)'))
    # plt.savefig('q1_simpleode.png')
    # plt.show()

    # Question 2.a
    # f = lorenz
    # r0 = [0, 1, 0]
    # t, r = euler_method_mv(f, r0, 0, 50, 0.001)
    # plt.figure()
    # for i in range(len(r0)):
    #     plt.plot(t, r[i])
    # plt.title('Lorenz')
    # plt.ylabel('Amplitude')
    # plt.xlabel('Time [s]')
    # plt.legend(('x(t)', 'y(t)', 'z(t)'))
    # plt.savefig('q2a_lorenz.png')
    # plt.show()

    # Question 2.b
    # f = lorenz
    # r0 = [0, 2, 20]
    # t, r = euler_method_mv(f, r0, 0, 35, 0.001)
    # cm = plt.get_cmap('bone')
    # fig = plt.figure()
    # ax1 = fig.add_subplot(131)
    # ax2 = fig.add_subplot(132)
    # ax3 = fig.add_subplot(133)
    # ax1.set_prop_cycle(color=[cm(1. * i / (len(t) - 1)) for i in range(len(t) - 1)])
    # ax2.set_prop_cycle(color=[cm(1. * i / (len(t) - 1)) for i in range(len(t) - 1)])
    # ax3.set_prop_cycle(color=[cm(1. * i / (len(t) - 1)) for i in range(len(t) - 1)])
    # for i in range(len(t) - 1):
    #     ax1.plot(r[1, i:i + 2], r[0, i:i + 2])
    #     ax2.plot(r[0, i:i + 2], r[2, i:i + 2])
    #     ax3.plot(r[1, i:i + 2], r[2, i:i + 2])
    # ax1.set_xlabel('y vs. x')
    # ax2.set_xlabel('x vs. z')
    # ax3.set_xlabel('y vs. z')
    # fig.suptitle('Lorenz Phase Space -- Wiki')
    # plt.savefig('q2b_lorenz_attractor')
    # plt.show()

    # Question 3
    # n = 1_000_000
    # r0 = [0.99*n, 0.01*n, 0]
    # t, r = euler_method_mv(sir_model, r0, 0, 3, 0.001)
    # plt.figure()
    # plt.plot(t, r[0], 'b')
    # plt.plot(t, r[1], 'r')
    # plt.plot(t, r[2], '--k')
    # plt.legend(('Susceptible', 'Infected', 'Recovered'))
    # plt.title('SIR Contagious Disease Model Q3.B')
    # plt.ylabel('Population')
    # plt.xlabel('Time [months]')
    # plt.savefig('q3c_sir_model.png')
    # plt.show()

    # Question 3.D
    # n = 1_000_000
    # r0 = [0.99*n, 0.01*n, 0, 0]
    # t, r = euler_method_mv(sir_model_vaccine, r0, 0, 3, 0.001)
    # plt.figure()
    # plt.plot(t, r[0], 'b')
    # plt.plot(t, r[1], 'r')
    # plt.plot(t, r[2], 'y')
    # plt.plot(t, r[3], '--k')
    # plt.plot(t, r[0] + r[1] + r[2] + r[3])
    # plt.legend(('Susceptible', 'Infected', 'Treated', 'Recovered'))
    # plt.title('SIR Contagious Disease Model with Vaccine and Treatment Q3.D')
    # plt.ylabel('Population')
    # plt.xlabel('Time [months]')
    # plt.savefig('q3d_sir_model_vt.png')
    # plt.show()

    # Question 4
    n = 1_000_000
    r0 = [0.99*n, 0.01*n, 0]
    t, r = euler_method_mv(sir_model_zombie, r0, 0, 100, 0.001)
    plt.figure()
    plt.plot(t, r[0], 'b')
    plt.plot(t, r[1], 'r')
    plt.plot(t, r[2], '--k')
    plt.legend(('Susceptible', 'Zombies', 'Recovered'))
    plt.title('SIR Zombie Q4.B')
    plt.ylabel('Population')
    plt.xlabel('Time [months]')
    plt.savefig('q4b_sir_model.png')
    plt.show()

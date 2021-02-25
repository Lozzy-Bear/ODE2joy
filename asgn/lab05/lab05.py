import numpy as np
import matplotlib.pyplot as plt


# class example Leap Frog
def harmonic_osc(r, t):
    w = 10  # sqrt(k/m) = sqrt(100/1)
    x = r[0]
    v = r[1]
    kx = v
    kv = -w ** 2 * x
    k = [kx, kv]
    return k


def orbital_motion(r, t):
    mu = 398_600  # [km^3/s^2]
    thrust = 0  # [N]
    g = 0.00981  # [km/s^]
    isp = 300  # [s]

    m = r[6]
    alpha = -mu / np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2) ** 3
    beta = thrust / (m * np.sqrt(r[3] ** 2 + r[4] ** 2 + r[5] ** 2))

    fm = np.array([-thrust / (g * isp)])
    fr = r[3:5 + 1]
    fv = alpha * r[0:2 + 1] + beta * r[3:5 + 1]

    return np.concatenate((fr, fv, fm))


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


def leapfrog(f, r0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    r = np.zeros((len(r0), len(t)))
    m = np.zeros_like(r)  # midpoint steps tracking
    r[:, 0] = r0
    m[:, 0] = r[:, 0] + 0.5 * dt * f(r[:, 0], t[0])
    for i in range(len(t) - 1):
        r[:, i + 1] = r[:, i] + dt * f(m[:, i], t[i] + 0.5 * dt)
        m[:, i + 1] = m[:, i] + dt * f(r[:, i + 1], t[i] + dt)
    return t, r


def crystalline(r, t):
    k = 1
    m = 1
    n = int(len(r)/2)
    x = np.zeros(n+2)
    x[1:-1] = r[:n]
    fx = r[n:]
    fv = k / m * (x[2:] + x[:-2] - 2*x[1:-1])

    return np.concatenate((fx, fv))


def crystalline_atomic(r, t):
    k = 10
    m = 1
    a = 1000
    n = int(len(r)/2)
    x = np.zeros(n+2)
    x[1:-1] = r[:n]
    fx = r[n:]
    fv = k / m * (x[2:] + x[:-2] - 2*x[1:-1]) + a / m * (x[2:] - x[1:-1])**5 + a / m * (x[:-2] - x[1:-1])**5

    return np.concatenate((fx, fv))


def total_energy(k, m, x, v):
    kinetic = np.sum(0.5 * m * v**2, axis=0)
    potential = np.zeros_like(x)
    potential[0, :] = 0.5 * k * x[0, :] ** 2
    potential[-1, :] = 0.5 * k * x[-1, :] ** 2
    for i in range(1, x.shape[0]-1, 1):
        potential[i, :] = 0.5 * k * (x[i+1, :] - x[i, :])**2
    potential = np.sum(potential, axis=0)
    return kinetic + potential


def plot_it(t, r, n, w0, name):
    plt.figure(figsize=[10, 12])
    plt.subplot(311)
    plt.plot(t, r[0, :])
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    
    plt.subplot(312)
    plt.imshow(r[:n, :], cmap="gray", origin="lower", aspect='auto', extent=(0, len(t)*dt, 0, n))
    plt.xlabel("Time")
    plt.ylabel("Mass Number")
    
    fft = np.fft.fft(r[0])
    freq = np.fft.fftfreq(len(r[0]), dt)
    fft = np.fft.fftshift(fft)
    freq = np.fft.fftshift(freq)
    fft = np.abs(fft)
    freq = freq * 2 * np.pi
    plt.subplot(313)
    plt.plot(freq, fft, 'm')
    top = np.max(np.abs(fft))
    for l in range(0, masses+1, 1):
        w = 2*w0*np.sin(l*np.pi/2/(masses+1))
        plt.plot([w, w], [0, top], 'k')
    plt.xlim(0, 2)
    plt.xlabel('Frequency, ω [rad/s]')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(name)
    return


if __name__ == '__main__':
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
    #rc('text', usetex=True)
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)       # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Question 1
    # t, r = leapfrog(orbital_motion, [500+6378, 0, 0, 0, 10, 0, 2000], 0, 700*60, 1)
    # t2, r2 = leapfrog(orbital_motion, [2000+6378, 0, 0, 0, 6.89755, 0, 2000], 0, 130*60, 1)
    # fig = plt.figure(figsize=[6, 5])
    # ax = fig.add_subplot(1, 1, 1)
    # earth = plt.Circle((0, 0), 6378, color='g')
    # ax.add_patch(earth)
    # plt.title('Leapfrog Orbital Motion')
    # plt.xlabel('Distance [km]')
    # plt.ylabel('Distance [km]')
    # plt.plot(r[0], r[1], '--m', label='Molynia')
    # plt.plot(r2[0], r2[1], '--c', label='LEO 2000 km')
    # plt.grid()
    # plt.tick_params(which='major', length=4, color='b')
    # plt.tick_params(which='minor', length=4, color='r')
    # plt.legend(loc='best')
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.savefig('q1_leapfrog_orbit.png')
    # plt.show()

    # Question 2
    # masses = 100
    # t0 = 0
    # tf = 1000
    # dt = 0.01
    # k = 1
    # m = 1
    # # Part A
    # x0 = np.zeros(masses)
    # v0 = np.zeros_like(x0)
    # r0 = np.concatenate((x0, v0))
    # r0[0] = 0.5
    # t, r = leapfrog(crystalline, r0, t0, tf, dt)
    # plot_it(t, r, masses, np.sqrt(k/m), 'q2a.png')
    # # Part B
    # x0 = np.random.uniform(-0.2, 0.2, masses)
    # v0 = np.zeros_like(x0)
    # r0 = np.concatenate((x0, v0))
    # t, r = leapfrog(crystalline, r0, t0, tf, dt)
    # plot_it(t, r, masses, np.sqrt(k/m), 'q2b.png')
    # # Part C
    # x0 = np.ones(masses) * 0.2
    # v0 = np.zeros_like(x0)
    # r0 = np.concatenate((x0, v0))
    # t, r = leapfrog(crystalline, r0, t0, tf, dt)
    # plot_it(t, r, masses, np.sqrt(k/m), 'q2c.png')
    # # Part D
    # x0 = np.zeros(masses)
    # v0 = np.ones(masses)*0.1
    # r0 = np.concatenate((x0, v0))
    # t, r = leapfrog(crystalline, r0, t0, tf, dt)
    # plot_it(t, r, masses, np.sqrt(k/m), 'q2d.png')

    # Question 2f
    masses = 100
    x0 = np.zeros(masses)
    v0 = np.zeros_like(x0)
    r0 = np.concatenate((x0, v0))
    r0[0] = 0.5
    t0 = 0
    tf = 10_000
    dt = 0.1
    k = 1
    m = 1
    tl, rl = leapfrog(crystalline, r0, t0, tf, dt)
    tr, rr = rk4(crystalline, r0, t0, tf, dt)
    el = total_energy(k, m, rl[:masses, :], rl[masses:, :])
    er = total_energy(k, m, rr[:masses, :], rr[masses:, :])
    plt.figure()
    plt.plot(tl, el, '--m', label='Leapfrog Method')
    plt.plot(tr, er, '--c', label='Runge-Kutta 4 Method')
    plt.xlabel('Time [s]')
    plt.ylabel('Energy [J]')
    plt.title('Energy Conserved')
    plt.legend(loc='best')
    plt.savefig('q2f_energy.png')
    plt.show()

    # Question 2g
    # masses = 100
    # x0 = np.zeros(masses)
    # v0 = np.zeros_like(x0)
    # r0 = np.concatenate((x0, v0))
    # r0[0] = 0.5
    # t0 = 0
    # tf = 1000
    # dt = 0.01
    # k = 10
    # m = 1
    # t, r = leapfrog(crystalline_atomic, r0, t0, tf, dt)
    # plot_it(t, r, masses, np.sqrt(k/m), 'q2g_k10a1000.png')

    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import scipy.constants as const


def gaussian_pulse_1d(l, x0, sigma, dt, tf, kappa):
    n = 1000
    alpha = l / n
    t = np.arange(0, tf, dt)
    x = np.arange(0, l, alpha)
    hbar = const.physical_constants["Planck constant over 2 pi in eV s"][0]
    mass = const.physical_constants["electron mass energy equivalent in MeV"][0] * 1e6 / const.speed_of_light ** 2
    beta = -1j * hbar * dt / (4 * mass * alpha**2)

    # Initialize matrices A and B
    A = np.eye(n, k=-1, dtype=complex) * -beta + \
        np.eye(n, k=0, dtype=complex) * (1 + 2 * beta) + \
        np.eye(n, k=1, dtype=complex) * -beta
    B = np.eye(n, k=-1, dtype=complex) * beta + \
        np.eye(n, k=0, dtype=complex) * (1 - 2 * beta) + \
        np.eye(n, k=1, dtype=complex) * beta

    # Initialize psi and set initial pulse(s)
    psi = np.zeros((len(t), len(x)), dtype=complex)
    for i in range(len(x0)):
        psi[0, :] += np.exp(-1 * (x - x0[i])**2 / (2 * sigma**2)) * np.exp(1j * kappa * x)

    for i in range(0, len(t)-1, 1):
        b = np.matmul(B, psi[i, :])
        psi[i + 1, :] = np.linalg.solve(A, b)

    return x, t, psi


def gaussian_pulse_2d():
    return


if __name__ == '__main__':
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)       # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labelsa
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Question 5a
    l = 1.0e-8              # [m]
    x0 = np.array([l/2])    # [m]
    sigma = 1.0e-10         # [m]
    dt = 1.0e-18            # [s]
    tf = 3.0e-15            # [s]
    kappa = 5.0e10          # [m^-1]

    x, t, p = gaussian_pulse_1d(l, x0, sigma, dt, tf, kappa)
    print("wave processed")

    with imageio.get_writer('pulse1d_5a.mp4', fps=100, mode='I') as writer:
        for i in range(len(t)):
            fig = plt.figure(1)
            plt.plot(x, np.real(p[i, :]), 'k')
            plt.title(f'Gaussian Pulse\nTime = {t[i]*1.0e-15:.4f} [fs]')
            plt.xlabel("Distance [m]")
            plt.ylabel("|Ψ(x)|")
            plt.ylim([-1.0, 1.0])
            canvas = FigureCanvas(fig)
            canvas.draw()
            writer.append_data(np.asarray(canvas.buffer_rgba()))
            plt.close(1)

        writer.close()
    print("movie processed")

    # Question 5b
    # l = 1.0e-8              # [m]
    # x0 = np.array([l/2])    # [m]
    # sigma = 1.0e-10         # [m]
    # dt = 1.0e-18            # [s]
    # tf = 3.0e-15            # [s]
    # kappa = 0.0             # [m^-1]
    #
    # x, t, p = gaussian_pulse_1d(l, x0, sigma, dt, tf, kappa)
    # print("wave processed")
    #
    # with imageio.get_writer('pulse1d_5b.mp4', fps=100, mode='I') as writer:
    #     for i in range(len(t)):
    #         fig = plt.figure(1)
    #         plt.plot(x, np.real(p[i, :]), 'k')
    #         plt.title(f'Gaussian Pulse\nTime = {t[i]*1.0e-15:.4f} [fs]')
    #         plt.xlabel("Distance [m]")
    #         plt.ylabel("|Ψ(x)|²")
    #         plt.ylim([-1.0, 1.0])
    #         canvas = FigureCanvas(fig)
    #         canvas.draw()
    #         writer.append_data(np.asarray(canvas.buffer_rgba()))
    #         plt.close(1)
    #
    #     writer.close()
    # print("movie processed")

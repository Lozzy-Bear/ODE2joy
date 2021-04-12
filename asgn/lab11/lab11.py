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
        psi[0, :] += np.exp(-1 * (x - x0[i])**2 / (2 * sigma**2)) * np.exp(1j * kappa[i] * x)

    for i in range(0, len(t)-1, 1):
        b = np.matmul(B, psi[i, :])
        psi[i + 1, :] = np.linalg.solve(A, b)

    return x, t, psi


def gaussian_pulse_2d_movie(l, x0, y0, sigma, dt, tf, kx, ky):
    # Initialize constants
    n = 1000
    alpha = l / n
    t = np.arange(0, tf, dt)
    x = np.arange(0, l, alpha)
    y = np.arange(0, l, alpha)
    gx, gy = np.meshgrid(x, y)
    hbar = const.physical_constants["Planck constant over 2 pi in eV s"][0]
    mass = const.physical_constants["electron mass energy equivalent in MeV"][0] * 1e6 / const.speed_of_light ** 2
    beta = -1j * hbar * dt / (4 * mass * alpha ** 2)

    # Initialize matrices A and B
    A = np.eye(n, k=-1, dtype=complex) * -beta + \
        np.eye(n, k=0, dtype=complex) * (1 - 4 * beta) + \
        np.eye(n, k=1, dtype=complex) * -beta
    A = np.kron(np.eye(n), A)
    A += np.eye(A.shape[0], k=-n, dtype=complex) * -beta + \
         np.eye(A.shape[0], k=n, dtype=complex) * -beta
    B = np.eye(n, k=-1, dtype=complex) * beta + \
        np.eye(n, k=0, dtype=complex) * (1 + 4 * beta) + \
        np.eye(n, k=1, dtype=complex) * beta
    B = np.kron(np.eye(n), B)
    B += np.eye(B.shape[0], k=-n, dtype=complex) * beta + \
         np.eye(B.shape[0], k=n, dtype=complex) * beta

    # Initialize psi and set initial pulse(s)
    psi = np.zeros((len(x), len(y)), dtype=complex)
    for i in range(len(x0)):
        psi[:, :] += np.exp(-1 * ((gx - x0[i]) ** 2 / (2 * sigma ** 2) + (gy - y0[i]) ** 2 / (2 * sigma ** 2))) * \
                        np.exp(1j * (kx[i] * gx + ky[i] * gy))
    psi = psi.flatten()

    # Simulate and make .mp4
    with imageio.get_writer('pulse2d.mp4', fps=100, mode='I') as writer:
        for i in range(0, len(t) - 1, 1):
            b = np.matmul(B, psi)
            psi = np.linalg.solve(A, b)

            fig = plt.figure(1)
            ax = plt.axes(projection='3d')
            ax.view_init(elev=45.0, azim=-45.0)
            ax.plot_surface(gx, gy, np.real(psi.reshape([n, n])), cmap='inferno', vmin=-1.0, vmax=1.0,
                            rstride=1, cstride=1, alpha=None, antialiased=True)
            ax.set_title(f'Gaussian Pulse\nTime = {t[i] * 1.0e15:.4f} [fs]')
            ax.set_xlabel("Distance [m]")
            ax.set_ylabel("Distance [m]")
            ax.set_zlabel("|Ψ(x)|")
            ax.set_xlim([0.0, l])
            ax.set_ylim([0.0, l])
            ax.set_zlim([0.0, 1.0])

            # plt.imshow(np.real(psi.reshape([n, n])), vmin=-1.0, vmax=1.0)
            # plt.xlabel('x')
            # plt.ylabel('y')

            canvas = FigureCanvas(fig)
            canvas.draw()
            writer.append_data(np.asarray(canvas.buffer_rgba()))
            plt.close(1)

            try:
                print(f'Step: {i}/{(len(t)-1)}, Time: {t[i] * 1.0e15:.4f}/{tf * 1.0e15:.4f} [fs]')
            except KeyboardInterrupt:
                print("user close")
                break

        writer.close()
        print("successful close")

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

    # Question 5
    # l = 1.0e-8              # [m]
    # x0 = np.array([l/4, l*3/4])    # [m]
    # sigma = 1.0e-10         # [m]
    # dt = 1.0e-18            # [s]
    # tf = 1.0e-17            # [s]
    # kappa = np.array([-5.0e10, 5.0e10])          # [m^-1]
    #
    # x, t, p = gaussian_pulse_1d(l, x0, sigma, dt, tf, kappa)
    # print("wave processed")
    #
    # with imageio.get_writer('collide_pulse_1d_multi.mp4', fps=100, mode='I') as writer:
    #     for i in range(len(t)):
    #         fig = plt.figure(1, figsize=[12, 7])
    #         plt.suptitle(f'Time = {t[i]*1.0e15:.4f} [fs]')
    #
    #         plt.subplot(121)
    #         plt.plot(x, np.real(p[i, :]), 'r', label='Re[Ψ(x)]')
    #         plt.plot(x, np.imag(p[i, :]), 'm', label='Im[Ψ(x)]')
    #         plt.title(f'Real and Imaginary')
    #         plt.xlabel("Distance [m]")
    #         plt.ylabel("Ψ(x)")
    #         plt.ylim([-1.0, 1.0])
    #
    #         plt.subplot(122)
    #         plt.plot(x, np.sqrt(np.real(p[i, :])**2 + np.imag(p[i, :])**2), 'k')
    #         plt.title(f'Magnitude')
    #         plt.xlabel("Distance [m]")
    #         plt.ylabel("|Ψ(x)|")
    #         plt.ylim([-1.0, 1.0])
    #
    #         plt.tight_layout()
    #
    #         canvas = FigureCanvas(fig)
    #         canvas.draw()
    #         writer.append_data(np.asarray(canvas.buffer_rgba()))
    #         plt.close(1)
    #
    #     writer.close()
    # print("movie processed")


    # Bonus
    l = 1.0e-8              # [m]
    sigma = 1.0e-10         # [m]
    dt = 1.0e-18            # [s]
    tf = 3.0e-15            # [s]
    speed = 5.0e10
    x0 = np.array([l/4, l/4, l*3/4, l*3/4])    # [m]
    y0 = np.array([l/4, l*3/4, l/4, l*3/4])    # [m]
    kx = np.array([1.0, 0.0, 0.0, -1.0]) * speed  # [m^-1]
    ky = np.array([0.0, -1.0, 1.0, 0.0]) * speed  # [m^-1]
    # x0 = np.array([l/2])
    # y0 = np.array([l/2])
    # kx = np.array([0.0])
    # ky = np.array([0.0])

    gaussian_pulse_2d_movie(l, x0, y0, sigma, dt, tf, kx, ky)

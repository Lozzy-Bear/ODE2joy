import numpy as np
import matplotlib.pyplot as plt


def magnetization_relaxation(const, temp, target):
    u = 700
    m = np.zeros(3)
    err = 1e16
    n = 0
    m[1] = 1
    m[2] = u * np.tanh(const * m[1] / temp)
    while err > target:
        m[0] = m[1]
        m[1] = m[2]
        m[2] = u * np.tanh(const * m[1] / temp)
        err = np.abs((m[1] - m[2])**2 / (2*m[1] - m[0] - m[2]))
        n += 1
    return m[2], n+2


def binary_search__root(f, x1, x2, target):
    while True:
        xp = 0.5 * (x1 + x2)
        err = np.abs(x1 - x2)
        if err < target:
            return xp
        elif (f(x1) * f(x2)) > 0 and err > 1e16:  # return nan if the error has exploded, initial bracket is poorly set.
            print(f"binary search non-convergence: likely bracket ({x1}, {x2}) does not bound root")
            return np.nan
        if f(xp) * f(x1) > 0:
            x1 = xp
        else:
            x2 = xp


def lagrange(r):
    g = 6.674e-11
    me = 5.974e24
    mm = 7.348e22
    rm = 3.844e8
    w = 2.662e-6
    return g*me/r**2 - g*mm/(rm-r)**2 - r*w**2


if __name__ == '__main__':
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
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
    # consts = np.arange(0.5, 1.5+0.1, 0.1)
    # temps = np.arange(1, 1200+1, 1)
    # dm = 1
    # mags = np.zeros((len(consts), len(temps)))
    # itrs = np.zeros_like(mags)
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[12, 8])
    # ax1.set_title('Magnetization strength, dm = 1.0')
    # ax2.set_title('Relaxation iterations, dm = 1.0')
    # for i, c in enumerate(consts):
    #     for j, t in enumerate(temps):
    #         mags[i, j], itrs[i, j] = magnetization_relaxation(c, t, dm)
    #     ax1.plot(temps, mags[i, :], label=f'j/kB = {np.around(c, 1)}')
    #     ax2.semilogy(temps, itrs[i, :], label=f'j/kB = {np.around(c, 1)}')
    #
    # ax1.set_xlabel('Temperature [K]')
    # ax1.set_ylabel('Magnetization [T]')
    # ax2.set_xlabel('Temperature [K]')
    # ax2.set_ylabel('Iterations')
    # ax1.legend(loc='best')
    # ax2.legend(loc='best')
    # plt.tight_layout()
    # plt.savefig('q1abc.png')
    #
    # consts = np.arange(0.5, 1.5+0.1, 0.1)
    # temps = np.arange(1, 1200+1, 1)
    # dm = 5
    # mags = np.zeros((len(consts), len(temps)))
    # itrs = np.zeros_like(mags)
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[12, 8])
    # ax1.set_title('Magnetization strength, dm = 5.0')
    # ax2.set_title('Relaxation iterations, dm = 5.0')
    # for i, c in enumerate(consts):
    #     for j, t in enumerate(temps):
    #         mags[i, j], itrs[i, j] = magnetization_relaxation(c, t, dm)
    #     ax1.plot(temps, mags[i, :], label=f'j/kB = {np.around(c, 1)}')
    #     ax2.semilogy(temps, itrs[i, :], label=f'j/kB = {np.around(c, 1)}')
    #
    # ax1.set_xlabel('Temperature [K]')
    # ax1.set_ylabel('Magnetization [T]')
    # ax2.set_xlabel('Temperature [K]')
    # ax2.set_ylabel('Iterations')
    # ax1.legend(loc='best')
    # ax2.legend(loc='best')
    # plt.tight_layout()
    # plt.savefig('q1d.png')
    #
    # consts = np.arange(0.5, 1.5+0.1, 0.1)
    # temps = np.arange(1, 1200+1, 1)
    # dm = 0.1
    # mags = np.zeros((len(consts), len(temps)))
    # itrs = np.zeros_like(mags)
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[12, 8])
    # ax1.set_title('Magnetization strength, dm = 0.1')
    # ax2.set_title('Relaxation iterations, dm = 0.1')
    # for i, c in enumerate(consts):
    #     for j, t in enumerate(temps):
    #         mags[i, j], itrs[i, j] = magnetization_relaxation(c, t, dm)
    #     ax1.plot(temps, mags[i, :], label=f'j/kB = {np.around(c, 1)}')
    #     ax2.semilogy(temps, itrs[i, :], label=f'j/kB = {np.around(c, 1)}')
    #
    # ax1.set_xlabel('Temperature [K]')
    # ax1.set_ylabel('Magnetization [T]')
    # ax2.set_xlabel('Temperature [K]')
    # ax2.set_ylabel('Iterations')
    # ax1.legend(loc='best')
    # ax2.legend(loc='best')
    # plt.tight_layout()
    # plt.savefig('q1d_dm.png')
    # plt.show()

    # Question 2
    # Solution should be 323,050 km from Earth or 61,350 km from Moon
    root = binary_search__root(lagrange, 7e6, 4e9, 1e3)
    print(f'L1 = {root/1000} [km]')


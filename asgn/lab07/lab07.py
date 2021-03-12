import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const


def newton_raphson(f, x0, target, p):
    # Newton-Raphson method
    err = target + 1
    while err > target:
        fu = f(x0)
        df = (f(x0+p) - fu) / p
        dx = fu / df
        x0 -= dx
        err = np.abs(dx)

    return x0


def newton_mv(f, x0, target, p):
    """
    Multivarbiale Newton-Raphson method

    Parameters
    ----------
        f : function
            function handle
        x0 : float np.array
            n-length array of values
        target : float
            target error value
        p : float
            perturbation amount for Jacobian

    Returns
    -------
        x0 : float np.array
            roots of the given function f
    """
    err = target + 1
    while err > target:
        df = jacobian(f, x0, p)
        dx = np.linalg.solve(df, f(x0))
        x0 -= dx
        err = np.max(np.abs(dx))

    return x0


def jacobian(f, x, p):
    """
    Calculate the multivariable Jacobian

    Parameters
    ----------
        f : function
            function handle
        x : float np.array
            n-length array of values
        p : float
            perturbation amount

    Returns
    -------
        df : np.array
            solved jacobian (nxn) matrix
    """
    df = np.zeros((len(x), len(x)))
    fu = f(x)
    for i in range(len(x)):
            xtemp = x.copy()
            xtemp[i] += p
            df[:, i] = (f(xtemp) - fu) / p
    return df


def test_system(x):
    f1 = np.exp(-1 * np.exp(x[0] + x[1])) - x[1] * (1 + x[0] ** 2)
    f2 = x[0] * np.cos(x[1]) + x[1] * np.sin(x[0]) - 0.5
    return np.array([f1, f2])


def rk4(f, r0, t0, tf, dt):
    # Initialize variables
    t = np.arange(t0, tf, dt)
    r = np.zeros((len(r0), len(t)), dtype=np.float)
    r[:, 0] = r0
    # Iterate method
    for idx in range(len(t) - 1):
        k1 = f(r[:, idx], t[idx])
        k2 = f(r[:, idx] + 0.5 * dt * k1, t[idx] + 0.5 * dt)
        k3 = f(r[:, idx] + 0.5 * dt * k2, t[idx] + 0.5 * dt)
        k4 = f(r[:, idx] + dt * k3, t[idx] + dt)
        k = k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
        r[:, idx + 1] = r[:, idx] + k * dt

    return t, r


def integrate(x, y):
    """
    Finds the are under the function x, f(x)
    Parameters
    ----------
        x : float np.array
            x-values
        y : float np.array
            f(x)-values

    Returns
    -------
        s : float
            area under the function
    """
    s = 0
    for i in range(len(x)-1):
        s += 0.5 * (x[i+1] - x[i])*(y[i+1] + y[i])
    return s


class Schrodinger:
    """
    Schrodinger wave equation class with multiple potential options.
    """
    def __init__(self, en, l=1e-11, vo=50.0):
        self.en = en
        self.l = l
        self.vo = vo
        self.hbar = const.physical_constants["Planck constant over 2 pi in eV s"][0]
        self.m = const.physical_constants["electron mass energy equivalent in MeV"][0] * 1e6 / const.speed_of_light**2

    def schrodinger(self, r, x):
        k1 = r[1]
        k2 = 2 * self.m / self.hbar ** 2 * (self.potential_quadratic(x) - self.en) * r[0]
        return np.array([k1, k2])

    def potential_well(self, x):
        if 0 <= x <= self.l:
            v = 0
        else:
            v = np.inf
        return v

    def energies_well(self, n):
        return 0.5 / self.m * (n * np.pi * self.hbar / self.l) ** 2

    def potential_quadratic(self, x):
        if -10*self.l <= x <= 10*self.l:
            v = self.vo * x ** 2 / self.l**2
        else:
            v = np.inf
        return v

    def energies_quadratic(self, n):
        return self.hbar / self.l * np.sqrt(2 * self.vo / self.m) * (n + 0.5)

    def potential_harmonic(self, x):
        k = (1e16)**2 * self.m
        return 0.5 * k * x ** 2

    def energies_harmonic(self, n):
        omega = 1e16
        return (2*n + 1) * self.hbar / 2 * omega


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

    # Question 2
    print(newton_mv(test_system, np.array([np.pi, np.pi]), 1e-4, 1e-10))

    # Question 3
    # f = Schrodinger(0, 1e-11)
    # plt.figure()
    # for en in [11e3, 14e3, 17e3]:
    #     f.en = en
    #     x, r = rk4(f.schrodinger, np.array([0, 1]), 0, 1e-11, 1e-11/1000)
    #     plt.plot(x, r[0], label=f"{en/1000} [keV]")
    # plt.xlabel("Distance [m]")
    # plt.ylabel("Ψ(x)")
    # plt.title("Schrodinger, Various Energies")
    # plt.legend(loc='best')
    # #plt.savefig('q3.png')
    # plt.show()

    # Question 4 and 5
    # enrg = [0, 4.0e3, 15.0e3, 40.0e3, 65.0e3]
    # plt.figure(figsize=[10, 7])
    # for n in range(1, 5, 1):
    #     pertb = 1e-10
    #     target = 0.01
    #     f = Schrodinger(enrg[n], 1e-11)
    #     err = target + 1
    #     while err > target:
    #         _, r = rk4(f.schrodinger, np.array([0, 1]), 0, 1e-11, 1e-11 / 1000)
    #         fu = r[0, -1]
    #         f.en = f.en + pertb
    #         _, r = rk4(f.schrodinger, np.array([0, 1]), 0, 1e-11, 1e-11 / 1000)
    #         df = (r[0, -1] - fu) / pertb
    #         dx = fu / df
    #         f.en = f.en - pertb - dx
    #         err = np.abs(dx)
    #     en = f.energies_well(n)
    #     print(f"solved: {f.en/1000} [keV]\nanalytical: {en/1000} [keV]")
    #
    #     x, r = rk4(f.schrodinger, np.array([0, 1.0]), 0, 1e-11, 1e-11/1000)
    #     y = r[0]**2
    #     y /= integrate(x, y)
    #     print("Check integrate = 1: ", integrate(x, y))
    #     plt.plot(x, y, label=f"Energy Level {n}: {f.en/1000:.03f} [keV]")
    #
    # plt.xlabel("Distance [m]")
    # plt.ylabel("|Ψ(x)|²")
    # plt.title("Infinite Potential Well")
    # plt.legend(loc='upper right')
    # #plt.savefig('q5.png')
    # plt.show()

    # Question 6
    # enrg = np.array([130, 410, 690, 960], dtype=np.float)
    # l = 10 * 1e-11
    # plt.figure(figsize=[10, 7])
    # for n in range(0, 4, 1):
    #     pertb = 1e-10
    #     target = 1e-6
    #     f = Schrodinger(en=enrg[n], l=1e-11, vo=50.0)
    #     err = target + 1
    #     while err > target:
    #         _, r = rk4(f.schrodinger, np.array([0, 1]), -l, l, l/2_000)
    #         fu = r[0, -1]
    #         f.en = f.en + pertb
    #         _, r = rk4(f.schrodinger, np.array([0, 1]), -l, l, l/2_000)
    #         df = (r[0, -1] - fu) / pertb
    #         dx = fu / df
    #         f.en = f.en - pertb - dx
    #         err = np.abs(dx)
    #     en = f.energies_quadratic(n)
    #     print(f"n: {n}, solved: {f.en} [eV], analytical: {en} [eV]")
    #
    #     # Question 5
    #     x, r = rk4(f.schrodinger, np.array([0, 1.0]), -l, l, l/2_000)
    #     y = r[0]**2
    #     y /= integrate(x, y)
    #     print("Check integrate = 1: ", integrate(x, y))
    #     plt.plot(x, y, label=f"Energy Level {n}: {f.en:.03f} [eV]")
    #
    # plt.xlabel("Distance [m]")
    # plt.ylabel("|Ψ(x)|²")
    # plt.title("Quadratic Potential")
    # plt.legend(loc='upper right')
    # plt.savefig('q6.png')
    # plt.show()

    # Bonus
    l = 78 * 1e-11
    plt.figure(figsize=[10, 7])
    data_x = np.zeros(21)
    data_y = np.zeros((21, len(np.arange(-l, l, l/2_000))))
    for n in range(0, 21, 1):
        pertb = 1e-10
        target = 1e-6
        f = Schrodinger(en=0.0, l=1e-11)
        f.en = f.energies_harmonic(n)
        x, r = rk4(f.schrodinger, np.array([0, 1.0]), -l, l, l/2_000)
        y = r[0]**2
        y /= integrate(x, y)
        data_x[n] = np.max(y)
        data_y[n, :] = y
        plt.plot(x, y, label=f"Energy Level {n}: {f.en:.03f} [eV]")
    plt.xlabel("Distance [m]")
    plt.ylabel("|Ψ(x)|²")
    plt.title("Harmonic Potential")


    plt.figure()
    plt.imshow(data_y, cmap="gist_heat", origin="lower",
               extent=(x[0], x[-1], 0, np.max(np.max(data_y))),
               aspect='auto', interpolation='none')
    plt.axis('off')
    plt.text(35e-11, 1e9, "|Ψₙ|²", c='w')
    plt.text(-35e-11, 4.8e9, "Eₙ=ħω(n+1/2)", c='w')
    for n in range(0, 21, 1):
        top = (0.142e9 + n*0.245e9)
        plt.plot([-60e-11, -40e-11], [top, top], 'w')
    plt.savefig('qbonus.png')
    plt.show()

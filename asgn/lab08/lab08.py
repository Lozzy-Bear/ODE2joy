import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const


def qr_decomposition(a):
    """
    Gram-Schmidt QR Decomposition method
    """
    n = a.shape[0]
    q = np.zeros_like(a, dtype=np.float)
    r = np.zeros_like(a, dtype=np.float)
    u = np.zeros_like(a, dtype=np.float)

    # Set matrix Q
    for i in range(n):
        ui = np.copy(a[:, i])
        for j in range(0, i):
            qj = q[:, j]
            p = np.dot(qj, ui) * qj
            ui = ui - p
        u[:, i] = ui
        q[:, i] = ui / np.linalg.norm(ui)

    # Set matrix R
    for i in range(n):
        # Set the diagonal
        r[i, i] = np.linalg.norm(u[:, i])
        for j in range(i+1, n):
            # Set the upper right
            r[i, j] = np.dot(q[:, i], a[:, j])

    return q, r


def qr_algorithm(a, target):
    n = a.shape[0]
    v = np.identity(n, dtype=np.float)
    m = np.ones_like(a, dtype=np.float)
    np.fill_diagonal(m, 0.0)
    while True:
        q, r = qr_decomposition(a)
        a = np.matmul(r, q)
        v = np.matmul(v, q)
        err = np.max(np.max(np.abs(a * m), axis=1) / np.max(np.abs(np.diag(a))))
        if err < target:
            break

    return a * np.identity(n), v


def lattice(k, m):
    # Todo: allow any k and m
    n = len(m)
    kk, mm = np.meshgrid(k, m)
    km = kk / mm
    a = np.eye(n, k=-1)*1.0 + np.eye(n, k=0)*-2.0 + np.eye(n, k=1)*1.0
    return a * km


def infinite_well(n, l, a):
    h = np.zeros((n, n), dtype=np.float)
    hbar = const.physical_constants["Planck constant over 2 pi in eV s"][0]
    m = const.physical_constants["electron mass energy equivalent in MeV"][0] * 1e6 / const.speed_of_light ** 2
    c = hbar**2 / 2.0 / m * np.pi**2 / l**2
    for i in range(1, n+1, 1):
        for j in range(1, n+1, 1):
            if i == j:  # m == n
                h[i-1, j-1] = c *j**2 + a/2
            else:  # m != n
                if (i + j) & 1:  # and m, n are one is odd on is even
                    h[i-1, j-1] = -(2 * l / np.pi) ** 2 * (i * j / (i ** 2 - j ** 2) ** 2) * (2 / l**2 * a)
                else:  # and m, n are both even or both odd
                    h[i-1, j-1] = 0.0
    return h


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

    # A = np.array([[1, 2, 3], [-1, 0, -3], [0, -2, 3]], dtype=np.float)
    # print(np.eye(2))
    # print(~np.eye(2))
    # Q, R = qr_decomposition(A)
    # print(Q)
    # print(R)
    # print(np.matmul(Q, R))
    # print(np.allclose(np.matmul(Q, R), A))
    # A = np.array([[1, 4, 8, 4], [4, 2, 3, 7], [8, 3, 6, 9], [4, 7, 9, 2]], dtype=float)
    # d, v = qr_algorithm(A, 1e-7)
    # print(np.matmul(A,v))
    # print(np.matmul(v,d))
    # print(np.allclose(np.matmul(A, v), np.matmul(v, d)))

    # Question 3
    # k = np.array([1, 2, 3, 4])
    # m = np.array([4, 3, 2, 1])
    # n = 100
    # k = np.ones(n)
    # m = np.ones(n)
    # a = lattice(k, m)
    # eval, evec = qr_algorithm(a, 1e-3)
    # w = 2 * 1.0 * np.sin(np.arange(1, n + 1, 1) * np.pi / 2 / (n + 1))
    # eval = np.diag(eval)
    # print('spring eigens')
    # print(eval)
    # print('spring freq')
    # print(np.sqrt(-eval))
    # print('spring modes')
    # print(w)
    # plt.figure()
    # plt.scatter(np.arange(1, n + 1, 1), w, marker='s', c='k', label='Analytical')
    # plt.scatter(np.arange(n, 0, -1), np.sqrt(-eval), marker='.', c='c', label='Computed')
    # plt.legend(loc='lower right')
    # plt.title(f'Spring Mass System, {n} Masses')
    # plt.xlabel('Modes [n]')
    # plt.ylabel('Frequencies [ω]')
    # plt.savefig(f'q3_{n}.png')
    # plt.show()

    # Question 4a
    # a = infinite_well(100, 0.5e-9, 10)
    # eval, evec = qr_algorithm(a, 1e-8)
    # eval = np.diag(eval)
    # print('energy levels\n', eval)

    # 10 energy levels
    # [155.55532632 126.85138369 101.28548221  78.72935895  59.18525691
    #  42.65507421  29.14419735  18.66289135  11.18109281   5.83637687]

    # 100 energy levels
    # [1.50462200e+04 1.47468864e+04 1.44505747e+04 1.41572711e+04
    #  1.38669758e+04 1.35796888e+04 1.32954100e+04 1.30141394e+04
    #  1.27358771e+04 1.24606231e+04 1.21883772e+04 1.19191396e+04
    #  1.16529103e+04 1.13896892e+04 1.11294763e+04 1.08722717e+04
    #  1.06180753e+04 1.03668872e+04 1.01187073e+04 9.87353563e+03
    #  9.63137222e+03 9.39221704e+03 9.15607011e+03 8.92293142e+03
    #  8.69280097e+03 8.46567876e+03 8.24156480e+03 8.02045907e+03
    #  7.80236159e+03 7.58727235e+03 7.37519135e+03 7.16611859e+03
    #  6.96005407e+03 6.75699779e+03 6.55694976e+03 6.35990997e+03
    #  6.16587842e+03 5.97485511e+03 5.78684004e+03 5.60183322e+03
    #  5.41983463e+03 5.24084429e+03 5.06486219e+03 4.89188834e+03
    #  4.72192272e+03 4.55496535e+03 4.39101622e+03 4.23007533e+03
    #  4.07214268e+03 3.91721828e+03 3.76530212e+03 3.61639420e+03
    #  3.47049452e+03 3.32760309e+03 3.18771990e+03 3.05084495e+03
    #  2.91697825e+03 2.78611978e+03 2.65826957e+03 2.53342759e+03
    #  2.41159386e+03 2.29276838e+03 2.17695114e+03 2.06414215e+03
    #  1.95434140e+03 1.84754890e+03 1.74376464e+03 1.64298863e+03
    #  1.54522087e+03 1.45046136e+03 1.35871010e+03 1.26996709e+03
    #  1.18423233e+03 1.10150583e+03 1.02178759e+03 9.45077601e+02
    #  8.71375878e+02 8.00682421e+02 7.32997235e+02 6.68320326e+02
    #  6.06651700e+02 5.47991366e+02 4.92339337e+02 4.39695628e+02
    #  3.90060258e+02 3.43433255e+02 2.99814655e+02 2.59204508e+02
    #  2.21602887e+02 1.87009899e+02 1.55425704e+02 1.26850551e+02
    #  1.01284851e+02 7.87293071e+01 5.91852043e+01 4.26550651e+01
    #  2.91441886e+01 1.86628895e+01 1.11810915e+01 5.83637647e+00]

    # Question 4b

    # Question 4c
    # x = np.arange(0, 0.5e-9, 0.1e-11)
    # plt.figure(figsize=[12, 8])
    # for i in range(0, 4+1, 1):
    #     psi_n = evec[:, -1-i]
    #     psi = 0
    #     for n in range(len(psi_n)):
    #         psi += psi_n[n] * np.sin(n * np.pi * x / 0.5e-9)
    #     plt.plot(x, psi, label=f'Energy level {i} = {eval[-1-i]:.2f} [eV]')
    # plt.legend(loc='best')
    # plt.title("Wave function")
    # plt.xlabel("Distance [m]")
    # plt.ylabel("Ψ(x)")
    # plt.savefig('q4c.png')
    # plt.show()



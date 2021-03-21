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
    #k = np.array([1, 2, 3, 4])
    #m = np.array([4, 3, 2, 1])
    # n = 10
    # k = np.ones(10)
    # m = np.ones(10)
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
    # plt.scatter(np.arange(1, n + 1, 1)+0.25, w, c='b')
    # plt.scatter(np.arange(n, 0, -1), np.sqrt(-eval), c='k')
    # plt.savefig('q3.png')
    # plt.show()

    # Question 4a
    a = infinite_well(10, 0.5e-9, 10)
    eval, evec = qr_algorithm(a, 1e-8)
    eval = np.diag(eval)
    print('energy levels\n', eval)
    # Question 4b

    # Question 4c
    # psi[]
    # for x in range(len(eval)):
    #     psi[x] =


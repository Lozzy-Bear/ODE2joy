import numpy as np
import matplotlib.pyplot as plt
import imageio


def svd(a):
    """
    Singular value decomposition
    """
    a = a.astype(complex)  # Cast problem as complex valued
    s, v = np.linalg.eig(np.matmul(a.T, a))
    s = np.sqrt(s)
    idx = np.argsort(s)[::-1]  # Sort in descending order
    s = s[idx]
    v = v[:, idx]
    u = np.zeros((a.shape[0], a.shape[0]), dtype=np.complex)
    for i in range(v.shape[0]):
        u[:, i] = 1./s[i] * np.matmul(a, v[:, i])

    # Remove complex value if solution is purely real
    if ~(np.any(np.iscomplex(u)) or np.any(np.iscomplex(s)) or np.any(np.iscomplex(v))):
        u = np.real(u)
        s = np.real(s)
        v = np.real(v)

    return u, np.diag(s), v


def svd_reconstruct(u, s, v, n):
    """
    Singular value decomposition reconstruct
    """
    a = np.zeros((u.shape[0], v.shape[0]), dtype=s.dtype)
    for i in range(n):
        a += np.outer(u[:, i], v[:, i]) * s[i, i]

    if ~np.any(np.iscomplex(a)):
        a = np.abs(a)

    return a


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

    # Question 1 & 2
    # #testa = np.array([[4., 4.], [-3., 3.]])
    # #testa = np.array([[2., 2.], [-3., 3.]])
    # testa = np.array([[1., 2., 1.], [2., 3., 2.], [1., 2., 1.]])
    # testu, tests, testv = svd(testa)
    # print('a:', testa)
    # print('u:', testu)
    # print('s:', tests)
    # print('v.T:', testv.T)
    # #print('usv.T:', testu @ tests @ testv.T)
    # print('recon:', svd_reconstruct(testu, tests, testv, 3))

    # a: [[2.  2.]
    #     [-3.  3.]]
    # u: [[0.  1.]
    #     [-1.  0.]]
    # s: [[4.24264069 0.]
    #     [0.         2.82842712]]
    # v.T: [[0.70710678 - 0.70710678]
    #       [0.70710678  0.70710678]]
    # usv.T: [[2.  2.]
    #         [-3.  3.]]

    # a: [[1. 2. 1.]
    #     [2. 3. 2.]
    #     [1. 2. 1.]]
    # u: [[0.45440135 + 0.00000000e+00j  0.54177432 + 0.00000000e+00j 0. + 6.78014817e-08j]
    #     [0.76618459 + 0.00000000e+00j - 0.64262055 + 0.00000000e+00j 0. + 3.39007409e-08j]
    #     [0.45440135 + 0.00000000e+00j 0.54177432 + 0.00000000e+00j   0. + 6.78014817e-08j]]
    # s: [[5.37228132 + 0.00000000e+00j 0. + 0.00000000e+00j 0. + 0.00000000e+00j]
    #     [0. + 0.00000000e+00j 0.37228132 + 0.00000000e+00j 0. + 0.00000000e+00j]
    #     [0. + 0.00000000e+00j 0.         + 0.00000000e+00j 0. + 1.30996904e-08j]]
    # v.T: [[4.54401349e-01 + 0.j  7.66184591e-01 + 0.j  4.54401349e-01 + 0.j]
    #       [-5.41774320e-01 + 0.j  6.42620551e-01 + 0.j - 5.41774320e-01 + 0.j]
    #       [7.07106781e-01 + 0.j - 1.26552158e-15 + 0.j - 7.07106781e-01 + 0.j]]
    # recon: [[1. 2. 1.]
    #         [2. 3. 2.]
    #         [1. 2. 1.]]

    # Question 3
    # acolor = imageio.imread('kath.jpg')
    # a = np.sum(acolor, axis=2)
    # a = a * 255. / np.max(a)
    # print(a.shape)
    # plt.figure()
    # plt.imshow(a, cmap='gray')
    # plt.title('Original')
    # plt.savefig('q3b_original.png')
    # u, s, v = svd(a)
    #
    # #terms = [512, 256, 128, 64, 32, 16, 8, 4]
    # terms = [128, 64, 32, 16, 8, 4]
    # for n in terms:
    #     an = svd_reconstruct(u, s, v, n)
    #     an = np.abs(an)
    #     plt.figure(n)
    #     plt.imshow(an, cmap='gray')
    #     plt.title(f'Terms Used = {n}')
    #     plt.savefig(f'q3b_terms_{n}.png')
    #
    # plt.show()



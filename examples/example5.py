import numpy as np
import matplotlib.pyplot as plt


def nonlineq2(x):
    return np.exp(1 - x**2)


def nonlineq1(x):
    return 2 - np.exp(-x)


def invnonlineq2(x):
    return np.sqrt(1 - np.log(x))


def relaxation_solve(f, x0, n):
    x = np.zeros(n)
    x[0] = x0

    for i in range(0, n-1, 1):
        x[i+1] = f(x[i])

    plt.figure()
    plt.plot(x)
    plt.show()

    return x[-1]


if __name__ == '__main__':
    #s = relaxation_solve(nonlineq1, 1, 10)
    #s = relaxation_solve(nonlineq2, 0.99, 10)
    s = relaxation_solve(invnonlineq2, 0.5, 10)
    print(s)

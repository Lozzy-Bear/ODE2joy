"""
Assignment 1 due Jan 21, 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def logistic_map(ri, rf, dr, xi, n):
    timer = time.time()
    r = np.arange(ri, rf+dr, dr)
    print(r)
    n = np.arange(n)
    x = np.zeros((len(n), len(r)))
    x[0, :] = xi
    for j in n[0:-1]:
        x[j + 1, :] = r[:] * x[j, :] * (1 - x[j, :])
    print(f'Processing time: {time.time() - timer} [s]')

    plt.figure(2)
    legend = []
    for i in range(x.shape[1]):
        plt.plot(n, x[:, i])
        legend.append(f'r = {r[i]}')
        plt.legend(f'r = {i}')
    plt.title('Slow Logistics Map Q1.B)')
    plt.xlabel('Iterations')
    plt.ylabel('x[n] Value')
    plt.legend(legend)

    return x, n


def slow_logistic_map(ri, rf, dr, xi, n):
    """
    This needs to be slow as heck. We want to plot 1 by 1 and plot (r, x)

    This seems to match what was demonstrated by Lucas.
    Parameters
    ----------
    ri
    rf
    dr
    xi
    n

    Returns
    -------

    """
    timer = time.time()
    r = np.arange(ri, rf+dr, dr)
    print(r)
    n = np.arange(n)
    #x = np.zeros((len(n), len(r)))
    #x[0, :] = xi
    x = xi
    for i in range(len(r)):
        for j in n[0:-1]:
            #x[j + 1, i] = r[i] * x[j, i] * (1 - x[j, i])
            x = r[i] * x * (1 - x)
        for j in n[0:-1]:
            x = r[i] * x * (1 - x)
            plt.scatter(r[i], x, c='red')
    print(f'Processing time: {time.time() - timer} [s]')

    #plt.figure(1)
    #legend = []
    #for i in range(x.shape[1]):
    #    plt.plot(r, x[:, i])
    #    legend.append(f'r = {r[i]}')
    #    plt.legend(f'r = {i}')
    plt.title('Slow Logistics Map Q1.A)')
    plt.xlabel('r value')
    plt.ylabel('x Value')
    #plt.legend(legend)

    return x, n


if __name__ == '__main__':
    dr = 0.5
    x, n = slow_logistic_map(1, 4, dr, 0.5, 1000)
    plt.show()
    #a, b = logistic_map(1, 4, dr, 0.5, 1000)
    #plt.show()
    """
    Values 4, 3.5, 3
    """

    """"
    Logistical mapping:
    Using logical arguments to set index points based on an argument from another vector.
    This is np.where() and np.ma.mask() stuff
    """
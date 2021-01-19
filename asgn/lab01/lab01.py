"""
Assignment 1 due Jan 21, 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def slow_logistic_map(dr):
    # Start time over operation
    timer = time.time()

    # Initialize variables
    r = np.arange(1, 4+dr, dr)

    # Set up figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Logistics Map Q1.A')
    ax.set_xlabel('[r] value')
    ax.set_ylabel('[x] value')
    for i in range(len(r)):
        x = 0.5
        # Run over first 1000 iterations
        for j in range(0, 999, 1):
            x = r[i] * x * (1 - x)
        # Iterate another 1000 times and plot points one by one -- very slow
        for j in range(0, 999, 1):
            x = r[i] * x * (1 - x)
            ax.scatter(r[i], x, c='black', marker='o')

    # Print total processing time and append to the plot
    timer = time.time() - timer
    ax.text(1, 0.9, f'Processing time: {timer:.4f} [s]', style='italic', bbox={'facecolor': 'tan', 'alpha': 0.5, 'pad': 10})
    print(f'Processing time: {timer:.4f} [s]')

    # Save and show the plot
    fig.savefig('q1a_slow_logistic_map.png')
    plt.show()

    return


def arr_logistic_map(dr):
    timer = time.time()
    r = np.arange(1, 4+dr, dr)
    x = np.zeros((2000, len(r)))
    x[0, :] = 0.5

    for j in range(0, 2000-1, 1):
        x[j + 1, :] = r[:] * x[j, :] * (1 - x[j, :])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Logistics Map Q1.D')
    ax.set_xlabel('[r] value')
    ax.set_ylabel('[x] value')
    ax.scatter(np.repeat(r[np.newaxis, :], x.shape[0]-1000, axis=0), x[999:-1, :], c='black', marker='o')

    # Print total processing time and append to the plot
    timer = time.time() - timer
    #ax.text(1, 0.9, f'Processing time: {timer:.4f} [s]', style='italic', bbox={'facecolor': 'tan', 'alpha': 0.5, 'pad': 10})
    print(f'Processing time: {timer:.4f} [s]')

    # Over plot r - 1 / r to verify limit of fixed point regime
    ax.plot(r, (r - 1)/r, c='red')
    # Save and show the plot
    fig.savefig('q1d_arr_logistic_map.png')
    plt.show()

    return


def leg_logistic_map(r, x0, n):
    x = np.zeros((n, len(r)))
    x[0, :] = x0

    for j in range(0, n-1, 1):
        x[j + 1, :] = r[:] * x[j, :] * (1 - x[j, :])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Logistics Map Q1.E')
    ax.set_xlabel('[r] value')
    ax.set_ylabel('[x] value')
    legend = []
    for i, rr in enumerate(r):
        ax.scatter(np.repeat(rr, x.shape[0]), x[:, i])
        legend.append(f'r = {rr}')
    ax.legend(legend)

    # Save and show the plot
    fig.savefig('q1e_leg_logistic_map.png')
    plt.show()

    return


def tent_map(du):
    n = 2000
    x0 = 0.5
    u = np.arange(1.0, 2.0+du, du)
    # u = np.linspace(1, 2, 10)
    print(u[0], u[-1], u)
    x = np.zeros((n, len(u)))
    x[0, :] = x0

    for j in range(0, n-1, 1):
        x[j + 1, :] = u[:] * np.where(x[j, :] < 0.5, x[j, :], 1 - x[j, :])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Tent Map Q2.A')
    ax.set_xlabel('[u] value')
    ax.set_ylabel('[x] value')
    ax.scatter(np.repeat(u[np.newaxis, :], x.shape[0] - 1000, axis=0), x[999:-1, :], c='black', marker='o')

    # Save and show the plot
    fig.savefig('q2a_tent_map.png')
    plt.show()

    return


if __name__ == '__main__':
    dr = 0.25
    du = 0.1
    #slow_logistic_map(dr)
    #arr_logistic_map(dr)
    #leg_logistic_map(np.array([1.5, 2.9, 3.5, 3.8]), 0.5, 100)
    tent_map(du)
    """
    Values 4, 3.5, 3
    
    Eliminating plotting calls the times are:
    Processing time: 0.2153 [s]
    Processing time: 0.1694 [s]
    
    Which goes to show in this low itr count the plotting is the real issue.
    At 200,000 itr
    Processing time: 2.5594 [s]
    Processing time: 0.9987 [s]
    """

    """"
    Logistical mapping:
    Using logical arguments to set index points based on an argument from another vector.
    This is np.where() and np.ma.mask() stuff
    """

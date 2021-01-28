"""
Assignment 1 due Jan 21, 2021
"""

import numpy as np
import matplotlib.pyplot as plt
import time


def slow_logistic_map(dr):
    """
    Produces a plot and times a slow implementation of the Logistic Map recurrence relation to demonstrate the
    efficacy of array based computation by comparison.

    Parameters
    ----------
        dr : float
            step increment of the constant r

    Returns
    -------
        None
    """

    # Start timer
    timer = time.time()

    # Initialize variables
    r = np.arange(1, 4+dr, dr)

    # Set up figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Logistics Map Q1.A')
    ax.set_xlabel('[r] value')
    ax.set_ylabel('[x] value')

    # Perform slow Logistic Map iterations
    for i in range(len(r)):
        x = 0.5
        # Run over first 1000 iterations to stabilize
        for j in range(0, 999, 1):
            x = r[i] * x * (1 - x)
        # Iterate another 1000 times and plot points one by one -- very slow
        for j in range(0, 999, 1):
            x = r[i] * x * (1 - x)
            ax.scatter(r[i], x, c='black', marker='o', s=0.25)

    # Print total processing time and append to the plot
    timer = time.time() - timer
    ax.text(1, 0.9, f'Processing time: {timer:.4f} [s]', style='italic', bbox={'facecolor': 'tan', 'alpha': 0.5, 'pad': 10})
    print(f'Processing time: {timer:.4f} [s]')

    # Save and show the plot
    fig.savefig('q1a_slow_logistic_map.png')
    plt.show()

    return


def arr_logistic_map(dr):
    """
    Produces a plot and times an Array-based Logistic Map recurrence relation to demonstrate the
    efficacy of array based computation over for-loops.

    Parameters
    ----------
        dr : float
            step increment of the constant r

    Returns
    -------
        None
    """
    # Start timer
    timer = time.time()

    # Initialize variables
    r = np.arange(1, 4+dr, dr)
    x = np.zeros((2000, len(r)))
    x[0, :] = 0.5

    # Perform slow Logistic Map iterations
    for j in range(0, 2000-1, 1):
        x[j + 1, :] = r[:] * x[j, :] * (1 - x[j, :])

    # Plot the result
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Logistics Map Q1.D')
    ax.set_xlabel('[r] value')
    ax.set_ylabel('[x] value')
    ax.scatter(np.repeat(r[np.newaxis, :], x.shape[0]-1000, axis=0), x[999:-1, :], c='black', marker='.', s=0.0001)

    # Print total processing time and append to the plot
    timer = time.time() - timer
    #ax.text(1, 0.9, f'Processing time: {timer:.4f} [s]', style='italic', bbox={'facecolor': 'tan', 'alpha': 0.5, 'pad': 10})
    print(f'Processing time: {timer:.4f} [s]')

    # Over plot r - 1 / r to verify limit of fixed point regime
    ax.plot(r, (r - 1)/r, c='red')

    # Save and show the plot
    fig.savefig('q1d_hd_arr_logistic_map.png')
    plt.show()

    return


def leg_logistic_map(r, x0, n):
    """
    Produces a Logistic Map plot for given array of constants, starting values, and iterations.

    Parameters
    ----------
        r : np.array(floats)
            An array of constants
        x0 : float
            The initial seed value for x
        n   : int
            The number of iterations to perform

    Returns
    -------

    """

    # Initialize variables
    x = np.zeros((n, len(r)))
    x[0, :] = x0

    # Perform iterations
    for j in range(0, n-1, 1):
        x[j + 1, :] = r[:] * x[j, :] * (1 - x[j, :])

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Logistics Map Negative Seed')
    ax.set_xlabel('[r] value')
    ax.set_ylabel('[x] value')

    # Produce a legend
    legend = []
    for i, rr in enumerate(r):
        ax.scatter(np.repeat(rr, x.shape[0]), x[:, i], s=3.5)
        legend.append(f'r = {rr}')
    ax.legend(legend)

    # Save and show the plot
    fig.savefig('q1e_leg_logistic_map.png')
    plt.show()

    return


def tent_map(du):
    """
    Creates a Tent Map plot

    Parameters
    ----------
        du : float
            constants step size

    Returns
    -------
        None

    """

    # Initialize variable
    n = 2000
    x0 = 0.5
    u = np.arange(1.0, 2.0+du, du)
    u[-1] = 2.0
    x = np.zeros((n, len(u)))
    x[0, :] = x0

    # Perform iterations
    for j in range(0, n-1, 1):
        x[j + 1, :] = u[:] * np.where(x[j, :] < 0.5, x[j, :], 1 - x[j, :])

    # Plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Tent Map Q2.A')
    ax.set_xlabel('[u] value')
    ax.set_ylabel('[x] value')
    ax.scatter(np.repeat(u[np.newaxis, :], x.shape[0] - 1000, axis=0), x[999:-1, :], c='black', marker='o', s=0.0001)

    # Save and show the plot
    fig.savefig('q2a_tent_map.png')
    plt.show()

    return


def bifurcation_tent_map():
    """
    Generates a bifurcation tent map to match https://commons.wikimedia.org/wiki/File:TentMap BifurcationDiagram.png

    Bifurcation diagram for the tent map. The horizontal axis is the μ parameter, the vertical axis is the x variable.
    The image was created by forming a 1000 x 1000 array representing increments of 0.001 in μ and x. A starting value
    of x=0.25 was used, and the map was iterated 1000 times in order to stabilize the values of x.
    100,000 x-values were then calculated for each value of μ and for each x value, the corresponding (x,μ) pixel in
    the image was incremented by one. All values in a column (corresponding to a particular value of μ) were then
    multiplied by the number of non-zero pixels in that column, in order to even out the intensities. Finally, values
    above 180,000 were set to 180,000, and then the entire image was normalized to 0-255.

    Parameters
    ----------
        None

    Returns
    -------
        None

    """

    # Initialize variable and pre-allocate image
    px = 1000
    n = 101000
    max_intensity = 180000.0
    ubins = np.linspace(1.0, 2.0, px)
    xbins = np.linspace(0.0, 1.0, px+1)
    x = np.zeros((n, len(ubins)))
    x[0, :] = 0.25
    img = np.zeros((px, px))

    # Perform iterations
    for j in range(0, n-1, 1):
        x[j + 1, :] = ubins[:] * np.where(x[j, :] < 0.5, x[j, :], 1 - x[j, :])

    # Even out intensities; multiply by non-zero pixels in column (u)
    x = x[999:-1, :]
    for i in range(img.shape[0]):
        img[i, :] = np.histogram(x[:, i], bins=xbins, density=False)[0]
        img[i, :] *= np.count_nonzero(img[i, :])

    # Set the maximum value to 180,000 and normalize to 0 - 255
    img = np.where(img > max_intensity, max_intensity, img)
    img = np.round(255 * img / np.amax(img))

    # Plot the result
    plt.figure()
    plt.imshow(np.flip(img.T, axis=0), cmap='Greys', vmin=0, vmax=255, extent=[1.0, 2.0, 0.0, 1.0])
    plt.title('Bifurcation Tent Map Q2.C')
    plt.xlabel('[u] value')
    plt.ylabel('[x] value')

    # Save and show the plot
    #plt.savefig('q2c_bifurcation_tent_map.png')
    plt.show()

    return


if __name__ == '__main__':
    dr = 0.001
    #du = 0.001
    #slow_logistic_map(dr)
    #arr_logistic_map(dr)
    #leg_logistic_map(np.array([1.5, 2.9, 3.5, 4.8]), -0.5, 3)
    #tent_map(du)
    bifurcation_tent_map()
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

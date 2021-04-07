import numpy as np
import matplotlib.pyplot as plt


def laplace_box():
    n = 100
    v = 1.0
    target = 1e-6

    # Initialize grids
    phi0 = np.zeros((n, n))
    phi1 = np.zeros_like(phi0)

    # Set boundary conditions
    phi0[0, :] = v
    phi1[0, :] = v
    phi0[-1, :] = 2 * v
    phi1[-1, :] = 2 * v

    delta = 1.0
    while delta > target:
        # Loop over internal grid points
        for i in range(1, n-1, 1):
            for j in range(1, n-1, 1):
                phi1[i, j] = 0.25 * (phi0[i+1, j] + phi0[i-1, j] + phi0[i, j+1] + phi0[i, j-1])

        delta = np.max(np.abs(phi1 - phi0))
        phi0 = phi1

    plt.figure()
    plt.imshow(phi0)
    plt.colorbar()
    plt.show()
    return


def poisson_box():
    n = 100
    v = 1.0
    l = 1.0
    eps0 = 1.0
    a = l/n
    target = 1e-8

    # Initialize grids
    phi0 = np.zeros((n, n))
    phi1 = np.zeros_like(phi0)
    rho = np.zeros_like(phi0)

    # Set boundary conditions
    # phi0[0, :] = v
    # phi1[0, :] = v
    # phi0[-1, :] = 2 * v
    # phi1[-1, :] = 2 * v
    # Set up bound charges rho
    rho[int(n/5):int(2*n/5), int(3*n/5):int(4*n/5)] = 100.0
    rho[int(3*n/5):int(4*n/5), int(n/5):int(2*n/5)] = -100.0

    delta = 1.0
    while delta > target:
        # Loop over internal grid points
        for i in range(1, n-1, 1):
            for j in range(1, n-1, 1):
                phi1[i, j] = 0.25 * (phi0[i+1, j] + phi0[i-1, j] + phi0[i, j+1] + phi0[i, j-1]) + \
                             0.25 * a**2 * rho[i, j] / eps0

        delta = np.max(np.abs(phi1 - phi0))
        phi0 = phi1
        print(delta)
    plt.figure()
    plt.imshow(phi0, interpolation='bilinear')
    plt.colorbar()
    plt.show()
    return


def heat_box(t0, tf, dt):
    l = 0.01
    d = 4.26e-6
    n = 100
    a = l/n
    t = np.arange(t0, tf, dt)

    temp0 = np.ones((n, len(t))) * 20.0
    temp0[0, :] = 50.0
    temp0[-1, :] = 0.0

    for i in range(len(t)):
        temp0[1:-1, i] = temp0[1:-1, i] + dt*d/a**2 * (temp0[1:-1, i] + temp0[0:-2, i] - 2*temp0[1:-1, i])

    plt.figure()
    plt.imshow(temp0)
    plt.show()

    return


if __name__ == '__main__':

    # These examples are not working 100% correctly.
    # DO NOT RELY ON THIS CODE.

    # laplace_box()
    # poisson_box()
    # heat_box(0, 1, 0.1)



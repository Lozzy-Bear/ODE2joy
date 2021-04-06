import numpy as np
import matplotlib.pyplot as plt
import time
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from numba import njit

def gaussSeidel( ):
    n = 100
    V = 12.0  # volts of plate potential
    q = 1.0e-5  # coulombs/m of bound charge between plates
    L = 0.1  # meters, so 10 cm by 10 cm box
    w = 0.8
    eps0 = 8.8541878128e-12  # F/m or C^2/N/m^2
    # normalize charge density in terms of permittivity of free space
    # eps0 = 1.0
    a = L/n  # grid spacing, m per grid spacing
    target = 1.0e-3

    phi1 = np.zeros( (n, n) )
    rho = np.zeros( (n, n) )

    # left and right sides of plates, in meters
    xl = int( 0.035 / a )  # [m]/[m/grid] -> [grid]
    xr = int( 0.065 / a )  # [m]/[m/grid] -> [grid]
    # top and bottom of plates
    yb = int ( 0.045 / a )
    yt = int ( 0.055 / a )
    # left plate
    phi1[ xl, yb:yt + 1 ] = V
    # right plate
    phi1[ xr, yb:yt + 1] = -V

    # bound charge between plates
    rho[ xl+1:xr, yb:yt ] = q

    delta = target * 2.0

    while delta > target:
        delta = 0.0
        # exclude edges as they are grounded
        for i in range( 1, n-1 ):
            for j in range( 1, n-1 ):
                current = phi1[i, j]
                phi1[i, j] = (1.0 + w) * 0.25 * ( phi1[i+1, j] + phi1[i-1, j] + phi1[i, j+1] + phi1[i, j-1] + a**2 / eps0 * rho[i, j] ) - w * current
                # put as little code into the branch as possible
                # so failing a branch isn't as big as a hit to the cache
                if i == xl and yb <= j <= yt:
                    phi1[i, j] = V
                if i == xr and yb <= j <= yt:
                    phi1[i, j] = -V

                diff = np.abs( phi1[i, j] - current )
                delta = diff if diff > delta else delta

    return


def capacitor(w):
    ts = time.time()
    n = 100
    v = 5
    l = 1.0
    eps0 = 1.0
    a = l/n
    target = 1e-3
    m = 1  # spacing between bound charge and plates

    # Initialize grids
    phi = np.zeros((n, n))
    rho = np.zeros_like(phi)

    # Set boundary conditions
    phi[int(2*n/5):int(3*n/5), int(3*n/5)] = -v
    phi[int(2*n/5):int(3*n/5), int(2*n/5)] = v

    # Set up bound charges rho
    rho[int(2*n/5):int(3*n/5), int(2*n/5+m):int(3*n/5-m)] = -v*1000

    delta = 1.0
    while delta > target:
        delta = 0.0
        # Loop over internal grid points
        for i in range(1, n-1, 1):
            for j in range(1, n-1, 1):
                temp_phi = phi[i, j]
                phi[i, j] = (1 + w) * 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] + a**2 * rho[i, j] / eps0) - w * phi[i, j]
                # Keep plate charge boundary conditions
                phi[int(2*n/5):int(3*n/5), int(3*n/5)] = -v
                phi[int(2*n/5):int(3*n/5), int(2*n/5)] = v
                # Check if solution is changing by less than target
                temp_delta = np.abs(phi[i, j] - temp_phi)
                if temp_delta > delta:
                    delta = temp_delta

    tf = time.time() - ts
    plt.figure(figsize=[16, 8])
    plt.suptitle(f'Capacitor\nw = {w}, Time = {tf:.2f}')
    plt.subplot(121)
    plt.imshow(phi, origin='lower left', interpolation='bilinear', cmap='inferno',
               vmax=np.max(np.abs(phi)), vmin=-1*np.max(np.abs(phi)))
    plt.axis('off')
    plt.colorbar(label='Voltage [V]')

    plt.subplot(122)
    dx, dy, mag = gradient2d(phi, a)
    plt.quiver(-1 * dx, -1 * dy, mag, cmap='inferno', pivot='mid')
    plt.axis('off')
    plt.colorbar(label='Magnitude')

    plt.savefig(f'q1d_{v}.png')
    plt.show()
    return


def gradient2d(arr, a):
    dx = np.zeros_like(arr)
    dy = np.zeros_like(arr)

    for i in range(1, arr.shape[1]-1, 1):
        dx[:, i] = (arr[:, i-1] - arr[:, i+1]) / (2 * a)
    for j in range(1, arr.shape[0]-1, 1):
        dy[j, :] = (arr[j - 1, :] - arr[j + 1, :]) / (2 * a)

    mag = np.sqrt(dx**2 + dy**2)

    return dx, dy, mag


def diffu(arr, min_temp, bound_temp):
    #m = 0.555e-4 / (bound_temp - min_temp)
    m = 1.053e-7
    d = arr * m
    return d


def heat_plate(bound_temp, min_temp, d, dt):
    l = 0.01
    n = 100
    a = l/n
    cadence = 100

    temp0 = np.ones((n, n)) * bound_temp
    temp0[1:-1, 1:-1] = min_temp
    frames = np.zeros((cadence * 60, n, n))
    times = np.zeros(cadence * 60)

    cnt = 0
    idx = 0

    while not np.allclose(min_temp, bound_temp, rtol=0.001):
        temp0[1:n-1, 1:n-1] = temp0[1:n-1, 1:n-1] + diffu(temp0[1:n-1, 1:n-1], min_temp, bound_temp) * \
                             dt/(a**2) * (temp0[1:n-1, 2:n] + temp0[1:n-1, 0:n-2] + temp0[2:n, 1:n-1] + temp0[0:n-2, 1:n-1] - 4*temp0[1:n-1, 1:n-1])
        min_temp = np.min(temp0)
        if cnt % int(1/(cadence * dt)) == 0:
            frames[idx, :, :] = temp0
            times[idx] = cnt*dt
            idx += 1
            print(min_temp, cnt*dt, idx, cnt)
        cnt += 1

    frames[idx, :, :] = temp0
    times[idx] = cnt * dt

    return frames[0:idx+1, :, :], times[0:idx+1]


if __name__ == '__main__':
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['DejaVu Serif']})
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14
    plt.rc('font', size=MEDIUM_SIZE)       # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labelsa
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Question 1
    # print(timeit.timeit(gaussSeidel, number=1))
    # print(timeit.timeit(capacitor, number=1))

    # capacitor(0.8)

    # Question 1d

    # Question 2a
    # d = 4.26e-6
    # bound_temp = 800.0
    # min_temp = 273.0
    # dt = 1e-4
    # frames, times = heat_plate(bound_temp, min_temp, d, dt)
    # print('processed steel')
    #
    # with imageio.get_writer('steel.mp4', fps=100, mode='I') as writer:
    #     for i in range(len(times)):
    #         fig = plt.figure(1)
    #         plt.imshow(frames[i, :, :], origin='lower left', cmap='inferno', vmin=0.0, vmax=bound_temp)
    #         plt.colorbar(label='Temperature [K]')
    #         plt.axis('off')
    #         plt.title(f'Steel Plate\nTime = {times[i]:.2f} [s], Min Temp = {np.min(frames[i, :, :]):.2f} [K]')
    #         canvas = FigureCanvas(fig)
    #         canvas.draw()
    #         writer.append_data(np.asarray(canvas.buffer_rgba()))
    #         plt.close(1)
    #
    #     writer.close()
    # print('movie steel')

    # Question 2b
    # d = 1.11e-4
    # bound_temp = 800
    # min_temp = 273
    # dt = 1e-5
    # frames, times = heat_plate(bound_temp, min_temp, d, dt)
    # print('processed copper')
    #
    # with imageio.get_writer('copper.mp4', fps=100, mode='I') as writer:
    #     for i in range(len(times)):
    #         fig = plt.figure(1)
    #         plt.imshow(frames[i, :, :], origin='lower left', cmap='inferno', vmin=0.0, vmax=bound_temp)
    #         plt.colorbar(label='Temperature [K]')
    #         plt.axis('off')
    #         plt.title(f'Copper Plate\nTime = {times[i]:.2f} [s], Min Temp = {np.min(frames[i, :, :]):.2f} [K]')
    #         canvas = FigureCanvas(fig)
    #         canvas.draw()
    #         writer.append_data(np.asarray(canvas.buffer_rgba()))
    #         plt.close(1)
    #
    #     writer.close()
    # print('movie copper')

    # Question 2c
    d = 1.11e-4
    bound_temp = 800
    min_temp = 273
    dt = 1e-5
    frames, times = heat_plate(bound_temp, min_temp, d, dt)

    with imageio.get_writer('variable.mp4', fps=100, mode='I') as writer:
        for i in range(len(times)):
            fig = plt.figure(1)
            plt.imshow(frames[i, :, :], origin='lower left', cmap='inferno', vmin=0.0, vmax=bound_temp)
            plt.colorbar(label='Temperature [K]')
            plt.axis('off')
            plt.title(f'Variable Plate\nTime = {times[i]:.2f} [s], Min Temp = {np.min(frames[i, :, :]):.2f} [K]')
            canvas = FigureCanvas(fig)
            canvas.draw()
            writer.append_data(np.asarray(canvas.buffer_rgba()))
            plt.close(1)

        writer.close()

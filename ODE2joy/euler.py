import numpy as np


def mv_euler_method(f, r0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    n = len(t)
    x = np.empty_like(t)
    y = np.empty_like(t)
    x[0] = x0
    y[0] = y0
    for idx in range(n-1):
        kx = fx(x[idx], y[idx])
        ky = fy(x[idx], y[idx], t[idx])
        x[idx + 1] = x[idx] + kx*dt
        y[idx + 1] = y[idx] + ky*dt
    return t, x, y


def arr_logistic_map(ri, rf, dr, xi, n):
    timer = time.time()
    r = np.arange(ri, rf+dr, dr)
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
    #for i in range(x.shape[1]):
    #    plt.plot(r, x[:, i])
    #    legend.append(f'r = {r[i]}')
    #    plt.legend(f'r = {i}')

    #plt.legend(legend)
    return x, n

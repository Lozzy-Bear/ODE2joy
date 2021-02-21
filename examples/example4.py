import numpy as np

# class example Leap Frog
def harmonic_osc(r, t):
    w = 10  # sqrt(k/m) = sqrt(100/1)
    x = r[0]
    v = r[1]
    kx = v
    kv = -w**2 * x
    k = [kx, kv]
    return k


def leapfrog(f, r0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    r = np.zeros((len(r0), len(t)))
    m = np.zeros_like(r)  # midpoint steps tracking
    r[:, 0] = r0
    k = f(r[:, 0], t[0])
    m[:, 0] = r0 + 0.5*dt*k  # may need transpose k
    for i in range(1, len(t)+1, 1):
        k = f(r[:, i], t[i] + 0.5*dt)
        r[:, i+1] = r[:, i] + dt*k
        k = f(r[:, i+1], t[i] + dt)
        m[:, i+1] = m[:, i] + dt*k
    return r, t

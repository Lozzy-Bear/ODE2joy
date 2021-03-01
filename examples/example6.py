import numpy as np


def diode_bridge(v):
    resistor = [1000, 4000, 3000, 2000]
    current0 = 3e-9
    vt = 0.05
    vp = 5.0

    f1 = (v[0]-vp)/resistor[0] + v[0]/resistor[1] + current0*np.exp((v[0]-v[1])/vt - 1)
    f2 = (vp-v[1])/resistor[2] - v[1]/resistor[3] + current0*np.exp((v[0]-v[1])/vt - 1)

    return [f1, f2]

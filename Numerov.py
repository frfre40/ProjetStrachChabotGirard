import numpy as np
import matplotlib.pyplot as plt


def Numerov(psi0, xi, xf, dx, potentiel):
    psi = np.array([psi0], float)

    for i in range(1, len(np.arange(xi, xf, dx))) - 2:
        A = 2 * (1 - 5 * dx**2/12* potentiel[i]) * psi[i]
        B = (1 + dx**2/12*potentiel[i - 1]) * psi[i - 1]
        C = (1 + dx**2/12*potentiel[i + 1])
    



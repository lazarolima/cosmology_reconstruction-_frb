from scipy.integrate import quad
import numpy as np

# Modelo cosmológico teórico
# Constants
Omega_b = 0.0408
c = 2.998e+8
m_p = 1.672e-27
pi = np.pi
G_n = 6.674e-11
Omega_m = 0.315
H_today = 74.03

def H_padrao(z):
    return H_today * np.sqrt(Omega_m * (1 + z) ** 3 + 1 - Omega_m)

# Fator
factor = (1.0504e-42) * 3 * c * Omega_b * H_today ** 2 / (8 * pi * G_n * m_p)

# Integrando da integral
def I(z):
    f_IGM = 0.83
    xe = 0.875
    H_th = H_padrao(z)
    return xe * f_IGM * factor * (1 + z) / H_th

# DM_IGM(z)
def DM_IGM(z):
    if np.isscalar(z):
        return quad(I, 0, z)[0]
    else:
        return np.array([quad(I, 0, zi)[0] for zi in z])
        
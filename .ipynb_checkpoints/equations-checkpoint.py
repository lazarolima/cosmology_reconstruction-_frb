from scipy.integrate import quad
import numpy as np

# Include your fiducial models for H(z) and DM_IGM(z)
class FiducialModel:
    def __init__(self):
        # Constants
        self.Omega_b = 0.0408 
        self.c = 2.998e+8
        self.m_p = 1.672e-27
        self.pi = np.pi
        self.G_n = 6.674e-11
        self.Omega_m = 0.315
        self.H_today = 74.03
        self.f_IGM_fid = 0.83
        self.xe_fid = 0.875
        
        # Precalculate factor
        self.factor = (1.0504e-42) * 3 * self.c * self.Omega_b * self.H_today ** 2 / (8 * self.pi * self.G_n * self.m_p)

    def H_padrao(self, z):
        return self.H_today * np.sqrt(self.Omega_m * (1 + z) ** 3 + 1 - self.Omega_m)

    def I(self, z):
        H_th = self.H_padrao(z)
        return self.xe_fid * self.f_IGM_fid * self.factor * (1 + z) / H_th

    def DM_IGM(self, z):
        if np.isscalar(z):
            return quad(self.I, 0, z)[0]
        else:
            return np.array([quad(self.I, 0, zi)[0] for zi in z])

# Include your parameterizations for baryon mass fraction f_IGM(z)
class Parameterization_f_IGM:

    @staticmethod
    def f_IGM_p2(z, f_IGM, alpha):
        return f_IGM + alpha * z / (1 + z)

    @staticmethod
    def f_IGM_p3(z, f_IGM, alpha):
        return f_IGM + alpha * z * np.exp(-z)

    @staticmethod
    def f_IGM_Linder(z, f_IGM, s):
        return f_IGM * (1 + s * (z - 3))
    
# Using the derivative of DM_IGM(z) via Gaussin Process
import gaussian_process as gp

# Create your class to include the parameterization for the Hubble parameter

class H_Model:           

    def __init__(self):
        self.factor = fiducial_model.factor
        fiducial_model = FiducialModel()
        Parameterization = Parameterization_f_IGM()
        mean_deriv1 = gp.pred_new()
        self.mean_deriv1 = np.asarray(self.mean_deriv1)

    def H_p1(self, z, f_IGM):
        result = self.factor * (1 + z) * f_IGM * 0.875 / self.mean_deriv1
        return result

    def H_p2(self, z, f_IGM, alpha):
        fIGM2 = Parameterization.f_IGM_p2(z, f_IGM, alpha)
        result = self.factor * (1 + z) * fIGM2 * 0.875 / self.mean_deriv1
        return result

    def H_p3(self, z, f_IGM, alpha):
        fIGM3 = Parameterization.f_IGM_p3(z, f_IGM, alpha)
        result = self.factor * (1 + z) * fIGM3 * 0.875 / self.mean_deriv1
        return result  

    def H_p4(self, z, f_IGM, s):
        fIGM4 = Parameterization.f_IGM_Linder(z, f_IGM, s)
        result = self.factor * (1 + z) * fIGM4 * 0.875 / self.mean_deriv1
        return result      
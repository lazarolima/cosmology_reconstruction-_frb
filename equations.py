from scipy.integrate import quad
from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

# Include your fiducial models for H(z) and DM_IGM(z)
class FiducialModel:
    def __init__(self):
        # Constants
        self.Omega_b: float = 0.0408 
        self.c: float = 2.998e+8
        self.m_p: float = 1.672e-27
        self.pi: float = np.pi
        self.G_n: float = 6.674e-11
        self.Omega_m: float = 0.3
        self.H_today: float = 70.0
        self.f_IGM_fid: float = 0.83
        self.xe_fid: float = 0.875
        
        # Precalculate factor
        self.factor: float = 1.0504e-42 * 3 * self.c * self.Omega_b * self.H_today ** 2 / \
                            (8 * self.pi * self.G_n * self.m_p)

    def H_std(self, z):
        return self.H_today * np.sqrt(self.Omega_m * (1 + z) ** 3 + 1 - self.Omega_m)

    def I(self, z):
        H_th = self.H_std(z)
        return self.xe_fid * self.f_IGM_fid * self.factor * (1 + z) / H_th

    def DM_IGM(self, z):
        if np.isscalar(z):
            return quad(self.I, 0, z)[0]
        else:
            return np.array([quad(self.I, 0, zi)[0] for zi in z])


# Include your parameterizations for baryons mass fraction f_IGM(z)
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

# Create your class to include the parameterization for the Hubble parameter

fiducial_model = FiducialModel()
Parameterization = Parameterization_f_IGM()

class H_Model:           

    def __init__(self):
        self.factor = fiducial_model.factor

        # Using the derivative of DM_IGM(z) via Gaussian Process
        from gaussian_process import GPReconstructionDMIGM
        gp = GPReconstructionDMIGM()
        __, __, mean_deriv, __ = gp.predict()
        mean_deriv = mean_deriv.flatten()
        self.mean_deriv = mean_deriv
        self.z_interp = gp.z_pred().flatten()

        # Carrega a função reconstruída da ANN
        self.func = np.load('data/ANN_DM_IGM_nodes[1, 4096, 2].npy')

    def deriv_ann(self):

        self.x = self.func[:, 0]
        self.y = self.func[:, 1]
        
        # Cria uma spline cúbica
        spl = InterpolatedUnivariateSpline(self.x, self.y, k=3)
        
        # Calcula a derivada
        derivative = spl.derivative()(self.x)
        
        return np.column_stack((self.x, derivative))

    def H_p(self, z, f_IGM, param, model_type, deriv_type):

        # Using the DM_IGM(z) via ANN (ReFANN)
        deriv = self.deriv_ann()
        self.z_interp_ann = deriv[:,0] 
        self.mean_deriv_ann = deriv[:,1]

        # Interpolation to represent the derivative of dDM_IGM(z)
        # Select interpolation based on deriv_type
        if deriv_type == 'GP':
            self.interp_mean_deriv = interp1d(self.z_interp, self.mean_deriv, kind='linear', fill_value="extrapolate")
        elif deriv_type == 'ANN':
            self.interp_mean_deriv = interp1d(self.z_interp_ann, self.mean_deriv_ann, kind='linear', fill_value="extrapolate")
        else:
            raise ValueError("Derivative type must be 'GP' or 'ANN'.")    

        # Select parameterization based on model_type
        if model_type == 'constant':
            fIGM = f_IGM
        elif model_type == 'p2':
            fIGM = Parameterization.f_IGM_p2(z, f_IGM, param)
        elif model_type == 'p3':
            fIGM = Parameterization.f_IGM_p3(z, f_IGM, param)
        elif model_type == 'p4':
            fIGM = Parameterization.f_IGM_Linder(z, f_IGM, param)
        else:
            raise ValueError("Model type must be 'constant', 'p2', 'p3', or 'p4'.")

        mean_deriv_interp = self.interp_mean_deriv(z)
        result = self.factor * (1 + z) * fIGM * 0.875 / mean_deriv_interp
        return result      
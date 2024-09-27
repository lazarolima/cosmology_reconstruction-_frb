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
    

class derivative_ann:
    def __init__(self):
    # Carrega a função reconstruída da ANN
        self.func = np.load('data/ANN_DM_IGM_bingo_nodes[1, 4096, 1].npy')

    def deriv_ann(self):

        self.x = self.func[:, 0]
        self.y = self.func[:, 1]
        
        # Cria uma spline cúbica
        spl = InterpolatedUnivariateSpline(self.x, self.y, k=3)
        
        # Calcula a derivada
        derivative = spl.derivative()(self.x)
        
        return np.column_stack((self.x, derivative))

# Create your class to include the parameterization for the Hubble parameter

fiducial_model = FiducialModel()
Parameterization = Parameterization_f_IGM()
derivative_dm_ann = derivative_ann()

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

    def H_p(self, z, f_IGM, param, model_type, deriv_type):

        # Using the dDM_IGM(z)/dz reconstructed via ANN (ReFANN)
        deriv = derivative_dm_ann.deriv_ann()
        self.z_interp_ann = deriv[:,0] 
        self.mean_deriv_ann = deriv[:,1]

        # Interpolation to represent the derivative of DM_IGM(z)
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
        xe = 0.875
        result = self.factor * (1 + z) * fIGM * xe / mean_deriv_interp
        return result    


# Include your models for DM_IGM(z) and DM_ext(z)
class DM_EXT_model:

    def __init__(self):
        # Constants 
        self.c: float = 2.998e+8
        self.m_p: float = 1.672e-27
        self.pi: float = np.pi
        self.G_n: float = 6.674e-11
        self.xe_fid: float = 0.875
        
        # Precalculate factor
        self.factor: float = 1.0504e-42 * 3 * self.c / (8 * self.pi * self.G_n * self.m_p)
        
    def H_std_new(self, z, Omega_m):
        return np.sqrt(Omega_m * (1 + z) ** 3 + 1 - Omega_m)

    def I(self, z, Omega_b, Omega_m, H_today, f_IGM, param, model_type):

        H_th = self.H_std_new(z, Omega_m)

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

        return self.xe_fid * fIGM * self.factor * Omega_b * H_today * (1 + z) / H_th
    

    def DM_IGM(self, z, Omega_b, Omega_m, H_today, f_IGM, param, model_type):

        integrand = self.I(z, Omega_b, Omega_m, H_today, f_IGM, param, model_type)
        
        if np.isscalar(z):
            # Single value of z
            return quad(integrand, 0, z)[0]
        else:
            # Array of z values
            return np.array([quad(integrand, 0, zi)[0] for zi in z])
        
    def DM_ext_th(self, z, f_IGM, DM_host_0, model_type, Omega_b=None, Omega_m=None, H_today=None, param=None):

        """# Set default values for cosmological parameters and f_IGM if not provided
        if f_IGM is None:
            f_IGM = 0.83
            model_type = 'constant'"""
        
        if Omega_b is None:
            Omega_b = 0.0408
        
        if Omega_m is None:
            Omega_m = 0.3
        
        if H_today is None:
            H_today = 70.0

        # Calculate the IGM contribution to the DM
        dm_igm_th = self.DM_IGM(z, Omega_b, Omega_m, H_today, f_IGM, param, model_type)

        # Return the total extragalactic DM: IGM contribution + host galaxy contribution
        return dm_igm_th + DM_host_0 / (1 + z)


  
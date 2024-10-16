from scipy.integrate import quad
from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import hyp2f1

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
        H_th = lambda z: self.H_std(z)
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

    # Insert here your dark energy parametrization 
    def func_de(self, z, omega_0, omega_a, param_type):
        if  param_type == 'constant':
            f_z = (1 + z) ** (3 * (1 + omega_0))
        elif param_type == 'CPL':
            f_z = (1 + z) ** (3 * (1 + omega_0 + omega_a)) * np.exp(- 3 * omega_a * z / (1 + z))
        elif param_type == 'BA':
            f_z = (1 + z) ** (3 * (1 + omega_0)) * (1 + z ** 2) ** (1.5 * omega_a)
        else:
            raise ValueError("Parameterization type must be 'constant', 'CPL', or 'BA'.")
        return f_z
        
    def H_std_new(self, z, Omega_m, cosmo_type, omega_0, omega_a, param_type):

        func_new = self.func_de(z, omega_0, omega_a, param_type)

        if cosmo_type == 'standard':
            H_z = np.sqrt(Omega_m * (1 + z) ** 3 + 1 - Omega_m)
        elif cosmo_type == 'non_standard':
            H_z = np.sqrt(Omega_m * (1 + z) ** 3 + (1 - Omega_m) * func_new)
        else:
            raise ValueError("Cosmology type must be 'standard' or 'non_standard'.")
        return H_z

    def I(self, z, Omega_b, Omega_m, H_today, f_IGM, param, model_type, cosmo_type, omega_0, omega_a, param_type):

        H_th = self.H_std_new(z, Omega_m, cosmo_type, omega_0, omega_a, param_type)

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
    

    def DM_IGM(self, z, Omega_b, Omega_m, H_today, f_IGM, param, model_type, cosmo_type, omega_0, omega_a, param_type):

        integrand = lambda z: self.I(z, Omega_b, Omega_m, H_today, f_IGM, param, model_type, cosmo_type, omega_0, omega_a, param_type)
        
        if np.isscalar(z):
            return quad(integrand, 0, z)[0]
        else:
            return np.array([quad(integrand, 0, zi)[0] for zi in z])

        
    #def DM_ext_th(self, z, f_IGM, DM_host_0, model_type, cosmo_type, param_type, omega_0=None, omega_a=None, Omega_b=None, Omega_m=None, H_today=None, param=None):
    def DM_ext_th(self, z, f_IGM, A, beta, model_type, cosmo_type, param_type, omega_0=None, omega_a=None, Omega_b=None, Omega_m=None, H_today=None, param=None):

        if omega_0 is  None:
            omega_0 = - 1

        if omega_a is None:
            omega_a = 0
        
        if Omega_b is None:
            Omega_b = 0.04897
        
        if Omega_m is None:
            Omega_m = 0.30966
            #Omega_m = 0.315
        
        if H_today is None:
            H_today = 73

        # Calculate the IGM contribution to the DM
        dm_igm_th = self.DM_IGM(z, Omega_b, Omega_m, H_today, f_IGM, param, model_type, cosmo_type, omega_0, omega_a, param_type)

        # Return the total extragalactic DM: IGM contribution + host galaxy contribution
        #return dm_igm_th + DM_host_0 / (1 + z)
        return dm_igm_th + A * (1 + z) ** beta
    

class Hubble:
    def __init__(self):
        pass 

    # Insert here your dark energy parametrization 
    def func_DE(self, z, omega_0, omega_a, param_type):
        if  param_type == 'constant':
            f_z = (1 + z) ** (3 * (1 + omega_0))
        elif param_type == 'CPL':
            f_z = (1 + z) ** (3 * (1 + omega_0 + omega_a)) * np.exp(- 3 * omega_a * z / (1 + z))
        elif param_type == 'BA':
            f_z = (1 + z) ** (3 * (1 + omega_0)) * (1 + z ** 2) ** (1.5 * omega_a)
        else:
            raise ValueError("Parameterization type must be 'constant', 'CPL', or 'BA'.")
        return f_z
        
    def H_func(self, z, H_0, Omega_m, cosmo_type, param_type, omega_0=None, omega_a=None):

        func_new = self.func_DE(z, omega_0, omega_a, param_type)

        if cosmo_type == 'standard':
            H_z = H_0 * np.sqrt(Omega_m * (1 + z) ** 3 + 1 - Omega_m)
        elif cosmo_type == 'non_standard':
            H_z = H_0 * np.sqrt(Omega_m * (1 + z) ** 3 + (1 - Omega_m) * func_new)
        else:
            raise ValueError("Cosmology type must be 'standard' or 'non_standard'.")
        return H_z


class Modulus_sne:

    def __init__(self):
        self.c: int = 299792 # km/s
        
    def Lumi_std(self, z, H_0, Omega_m, cosmo_type, param_type, omega_0, omega_a):

        hubble = Hubble()

        integrand = lambda z: 1 / hubble.H_func(z, H_0, Omega_m, cosmo_type, param_type, omega_0, omega_a)

        if np.isscalar(z):
            return self.c * (1 + z) * quad(integrand, 0, z)[0]
        else:
            return self.c * (1 + z) * np.array([quad(integrand, 0, zi)[0] for zi in z])
        
    # Distance modulis
    def Modulo_std(self, z, H_0, Omega_m, cosmo_type, param_type, omega_0=None, omega_a=None):

        lumi_std = self.Lumi_std(z, H_0, Omega_m, cosmo_type, param_type, omega_0, omega_a)

        return 5 * np.log10(lumi_std) + 25


  
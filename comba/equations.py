from scipy.integrate import quad
from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from joblib import Parallel, delayed
import pymp
import warnings
from scipy.integrate import IntegrationWarning
from scipy.stats import norm, lognorm
from sklearn.neighbors import KernelDensity

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
        
        # Precalculate factor (if H0=100h, we add a factor of 1e+4)
        self.factor: float = 1e+4 * 1.0504e-42 * 3 * self.c / (8 * self.pi * self.G_n * self.m_p)

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
        
    def H_std_new(self, z, h, Omega_mh2, cosmo_type, omega_0, omega_a, param_type):

        func_new = self.func_de(z, omega_0, omega_a, param_type)

        if cosmo_type == 'standard':
            H_z = 100 * np.sqrt(Omega_mh2 * ((1 + z) ** 3 - 1) + h ** 2)
        elif cosmo_type == 'non_standard':
            H_z = 100 * np.sqrt(Omega_mh2 * ((1 + z) ** 3 - func_new) + h ** 2 * func_new)
        else:
            raise ValueError("Cosmology type must be 'standard' or 'non_standard'.")
        return H_z

    def I(self, z, Omega_bh2, h, Omega_mh2, f_IGM, param, model_type, cosmo_type, omega_0, omega_a, param_type):

        H_th = self.H_std_new(z, h, Omega_mh2, cosmo_type, omega_0, omega_a, param_type)

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

        return self.xe_fid * fIGM * self.factor * Omega_bh2 * (1 + z) / H_th
    
    def DM_IGM(self, z, Omega_bh2, h, Omega_mh2, f_IGM, param, model_type, cosmo_type, omega_0, omega_a, param_type):

        integrand = lambda z: self.I(z, Omega_bh2, h, Omega_mh2, f_IGM, param, model_type, cosmo_type, omega_0, omega_a, param_type)
        
        if np.isscalar(z):
            return quad(integrand, 0, z)[0]
        else:
            return np.array([quad(integrand, 0, zi)[0] for zi in z])

    #def DM_ext_th(self, z, f_IGM, A, beta, model_type, cosmo_type, param_type, omega_0=None, omega_a=None, Omega_bh2=None, Omega_m=None, param=None):
    def DM_ext_th(self, z, f_IGM, Omega_bh2, h, DM_host_0, model_type, cosmo_type, param_type, omega_0=None, omega_a=None, Omega_mh2=None, param=None):

        if omega_0 is  None:
            omega_0 = - 1

        if omega_a is None:
            omega_a = 0
        
        if Omega_mh2 is None:
            Omega_mh2 = 0.1424
            #Omega_m = 0.315
        
        """if H_today is None:
            H_today = 73.04"""

        # Calculate the IGM contribution to the DM
        dm_igm_th = self.DM_IGM(z, Omega_bh2, h, Omega_mh2, f_IGM, param, model_type, cosmo_type, omega_0, omega_a, param_type)

        # Return the total extragalactic DM: IGM contribution + host galaxy contribution
        #return dm_igm_th + A * (1 + z) ** (beta - 1)
        return dm_igm_th + DM_host_0 / (1 + z)


class DM_EXT_model_Rec:

    def __init__(self):
        # Constants 
        self.c: float = 2.998e+8
        self.m_p: float = 1.672e-27
        self.pi: float = np.pi
        self.G_n: float = 6.674e-11
        self.xe_fid: float = 0.875
        
        # Precalculate factor (if H0=100h, we add a factor of 1e+4)
        self.factor: float = 1e+4 * 1.0504e-42 * 3 * self.c / (8 * self.pi * self.G_n * self.m_p)

    def I(self, z, Omega_bh2, f_IGM, param, model_type, rec_type):
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
        
        # Carregar os dados (ignorar a primeira linha com o cabeçalho)
        H_rec_gp = np.loadtxt('data/Hz35_rec_gapp.dat', skiprows=1)

        H_rec_ann = np.load('data/ANN_Hz35_nodes[1, 4096, 2].npy')

        # Separar as colunas em variáveis individuais
        z_val_gp, mean_gp = H_rec_gp[:, 0], H_rec_gp[:, 1]

        z_val_ann, mean_ann = H_rec_ann[:, 0], H_rec_ann[:, 1]
        
        # Criar a função de interpolação (interpolação linear)
        if rec_type ==  'GP':
            interp_func = interp1d(z_val_gp, mean_gp, kind='cubic', fill_value='extrapolate')
        elif  rec_type == 'ANN':
            interp_func = interp1d(z_val_ann, mean_ann, kind='cubic', fill_value='extrapolate')

        H_value = interp_func(z)
            
        return self.xe_fid * fIGM * self.factor * Omega_bh2 * (1 + z) / H_value
    
    def DM_IGM(self, z, Omega_bh2, f_IGM, param, model_type, rec_type):
        # Define a wrapper to extract only the integral result from quad
        def integrate(zi):
            result, _ = quad(self.I, 0, zi, args=(Omega_bh2, f_IGM, param, model_type, rec_type))
            return result

        if np.isscalar(z):
            # Single value of z
            return integrate(z)
        else:
            # Parallel integration for multiple values of z
            results = Parallel(n_jobs=-1)(
                delayed(integrate)(zi) for zi in z
            )
            return np.array(results)

    def DM_ext_th(self, z, f_IGM, Omega_bh2, A, beta, model_type, rec_type, param):
    # def DM_ext_th(self, z, f_IGM, Omega_bh2, DM_host_0, model_type, rec_type, param=None):

        # Calculate the IGM contribution to the DM
        dm_igm_th = self.DM_IGM(z, Omega_bh2, f_IGM, param, model_type, rec_type)

        # Return the total extragalactic DM: IGM contribution + host galaxy contribution
        return dm_igm_th + A * (1 + z) ** (beta - 1)
        # return dm_igm_th + DM_host_0 / (1 + z)

    def DM_ext_pdf(self, z, A, beta, sigma_igm, dm_ext_obs, dm_ext_th_values):
        
        # Definir uma função auxiliar para o cálculo da integral para cada valor de z_val
        def compute_pdf(z_val, dm_ext_th, sigma, dm_obs):
            def integrand(z_local):
                # Parte Gaussiana para DM_IGM
                pdf_igm = np.exp(-(dm_obs - dm_ext_th) ** 2 / (2 * sigma ** 2)) / sigma

                # Parte Lognormal para DM_HG
                mean = np.log(63.55)
                sigma_host = 50 
                pdf_host = np.exp(-(np.log(A * (1 + z_local) ** beta) - mean) ** 2 / (2 * (sigma_host / (1 + z_local)) ** 2)) / sigma_host

                return beta * pdf_igm * pdf_host / (2 * np.pi)

            # Executar a integração
            integral_result, _ = quad(integrand, -1, (dm_obs / A) ** (1 / beta), epsabs=1e-5, epsrel=1e-5)
            return integral_result

        # Paralelizar o cálculo usando todos os núcleos disponíveis
        pdf_values = Parallel(n_jobs=-1)(
            delayed(compute_pdf)(z_val, dm_ext_th, sigma, dm_obs)
            for z_val, dm_ext_th, sigma, dm_obs in zip(z, dm_ext_th_values, sigma_igm, dm_ext_obs)
        )

        # Retornar o resultado como um array numpy
        return np.array(pdf_values)


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
        
    def H_func(self, z, h, Omega_mh2, cosmo_type, param_type, omega_0=None, omega_a=None):

        func_new = self.func_DE(z, omega_0, omega_a, param_type)

        if cosmo_type == 'standard':
            H_z = 100 * np.sqrt(Omega_mh2 * ((1 + z) ** 3 - 1) + h ** 2)
        elif cosmo_type == 'non_standard':
            H_z = 100 * np.sqrt(Omega_mh2 * ((1 + z) ** 3 - func_new) + h ** 2 * func_new)
        else:
            raise ValueError("Cosmology type must be 'standard' or 'non_standard'.")
        return H_z


class Modulus_sne:

    def __init__(self):
        self.c: int = 299792 # km/s
    
    def Lumi_std(self, z, h, Omega_mh2, cosmo_type, param_type, omega_0, omega_a):
        # Instancia o objeto Hubble
        hubble = DM_EXT_model()

        # Função integrando a 1/H(z)
        def integrand(zi):
            return 1 / hubble.H_std_new(zi, h, Omega_mh2, cosmo_type, omega_0, omega_a, param_type)

        # Função que faz a integração
        def integrate(zi):
            result, _ = quad(integrand, 0, zi)
            return result

        if np.isscalar(z):
            # Caso escalar, não é necessário paralelizar
            return self.c * (1 + z) * integrate(z)
        else:
            # Paraleliza a integração para múltiplos valores de z
            results = Parallel(n_jobs=-1)(
                delayed(integrate)(zi) for zi in z
            )
            return self.c * (1 + z) * np.array(results)
        
    # Distance modulis
    def Modulo_std(self, z, h, Omega_mh2, cosmo_type, param_type, z_alt=None, omega_0=None, omega_a=None):

        if omega_0 is  None:
            omega_0 = - 1

        if omega_a is None:
            omega_a = 0

        lumi_std = self.Lumi_std(z, h, Omega_mh2, cosmo_type, param_type, omega_0, omega_a)

        if z_alt is not None:
            return 5 * np.log10(lumi_std) + 25 + 5 * np.log10((1 + z_alt)/(1 + z))
        elif z_alt is None:
            return 5 * np.log10(lumi_std) + 25 
        

class Cosmography:

    def __init__(self):
        # Constants 
        self.c: float = 2.998e+8
        self.m_p: float = 1.672e-27
        self.pi: float = np.pi
        self.G_n: float = 6.674e-11
        self.xe_fid: float = 0.875
        
        # Precalculate factor (if H0=100h, we add a factor of 1e+4)
        self.factor: float = 1e+4 * 1.0504e-42 * 3 * self.c / (8 * self.pi * self.G_n * self.m_p)
        # self.factor: float = 1.0504e-42 * 3 * self.c / (8 * self.pi * self.G_n * self.m_p)

    def DM_IGM_cmy(self, z, Omega_bh2, H0, f_IGM, q0, j0):

        dm_igm = (self.factor * Omega_bh2 * self.xe_fid * f_IGM / H0) * (z - 0.5 * q0 * z ** 2 + 
                     0.16 * (4 + 6 * q0 + 3 * q0 ** 3 - j0) * z ** 3) 
                    #  - 0.0416 * (18 * q0 + 42 * q0 ** 2 + 14 * q0 * j0 - 9 * q0 ** 3 - 18 * j0 - s0) * z ** 4)
        
        return dm_igm    
    
    def DM_ext_th_cmy(self, z, Omega_bh2, H0, f_IGM, q0, j0, A, beta):

        # Calculate the IGM contribution to the DM
        dm_igm_th_cmy = self.DM_IGM_cmy(z, Omega_bh2, H0, f_IGM, q0, j0)

        # Return the total extragalactic DM: IGM contribution + host galaxy contribution
        return dm_igm_th_cmy + A * (1 + z) ** (beta - 1)
    

    # def DM_ext_cmy_pdf(self, z, A, beta, Omega_bh2, H0, f_IGM, q0, j0, n_samples=66):
    #     """
    #     Gera amostras de DM_ext_th usando Monte Carlo e estima a PDF para cada valor de z.

    #     Args:
    #         z (array): Valores de redshift.
    #         A (float): Constante do modelo para DM_host.
    #         beta (float): Expoente do modelo para DM_host.
    #         Omega_bh2, H0, f_IGM, q0, j0 (float): Parâmetros cosmológicos.
    #         n_samples (int): Número de amostras por z.
    #         parallel (bool): Paralelizar o cálculo.

    #     Returns:
    #         array: Amostras de DM_ext_th para cada valor de z.
    #     """
    #     dm_igm_th_cmy = self.DM_IGM_cmy(z, Omega_bh2, H0, f_IGM, q0, j0)

    #     def compute_cmy_monte_carlo(z_val, dm_igm_th_cmy_val):
    #         np.random.seed(42)  # Seed para reprodutibilidade
    #         sigma_igm = 173.8 * z_val ** 0.4
    #         X_samples = np.random.normal(loc=dm_igm_th_cmy_val, scale=sigma_igm, size=n_samples)
    #         dm_host_mean = A * (1 + z_val) ** (beta - 1)
    #         sigma_host = 1.24
    #         Y_samples = np.random.lognormal(mean=np.log(dm_host_mean), sigma=np.log(1 + sigma_host / dm_host_mean), size=n_samples)
    #         Z_samples = X_samples + Y_samples
    #         return Z_samples

    #     # Gerar amostras usando paralelização ou loop
    #     # if parallel:
    #     #     pdf_cmy_values = Parallel(n_jobs=20)(
    #     #         delayed(compute_cmy_monte_carlo)(z_val, dm_igm_th_cmy_val)
    #     #         for z_val, dm_igm_th_cmy_val in zip(z, dm_igm_th_cmy)
    #     #     )
        
    #     pdf_cmy_values = [
    #         compute_cmy_monte_carlo(z_val, dm_igm_th_cmy_val)
    #         for z_val, dm_igm_th_cmy_val in zip(z, dm_igm_th_cmy)
    #     ]

    #     return np.array(pdf_cmy_values)
    
    # def DM_ext_cmy_pdf(self, z, A, beta, sigma_igm, dm_ext_obs, dm_ext_th_values, n_samples=66):
    
    #     # Função auxiliar para o cálculo da integral para cada valor de z_val
    #     def compute_cmy_monte_carlo(z_val, dm_ext_th, sigma, dm_obs):
    #         # Gerar amostras de X (normal) e Y (lognormal)
    #         np.random.RandomState(42)
    #         X_samples = np.random.normal(loc=dm_ext_th, scale=sigma, size=n_samples)  # DM_IGM ~ Normal
    #         dm_host_mean = A * (1 + z_val) ** beta
    #         sigma_host = 1.24
    #         Y_samples = np.random.lognormal(mean=dm_host_mean, sigma=sigma_host, size=n_samples)  # DM_HG ~ Lognormal

    #         # Calcular Z = X + Y para cada amostra
    #         Z_samples = X_samples + Y_samples / (1 + z_val)

    #         # Estimar a PDF de Z usando a média das amostras
    #         pdf_estimate = np.mean(np.exp(-0.5 * ((Z_samples - dm_obs)**2) / (sigma**2)))

    #         return pdf_estimate
        
    #     # Paralelizar o cálculo para cada valor de z
    #     pdf_cmy_values = [compute_cmy_monte_carlo(z_val, dm_ext_th, sigma, dm_obs)
    #                     for z_val, dm_ext_th, sigma, dm_obs in zip(z, dm_ext_th_values, sigma_igm, dm_ext_obs)]

    #     # Retornar o resultado como um array numpy
    #     return np.array(pdf_cmy_values)
            
    # def DM_ext_cmy_pdf(self, z, A, beta, sigma_igm, dm_ext_obs, dm_ext_th_values, n_samples=66, bandwidth=1.0):
    #     """
    #     Calcula a PDF de DM_ext usando o método de Monte Carlo.
        
    #     Parâmetros:
    #         - z: Array de valores de redshift
    #         - A, beta: Parâmetros da relação para DM_host = A * (1 + z)**beta
    #         - sigma_igm: Desvios padrão da Gaussiana DM_IGM
    #         - dm_ext_obs: Valores observados de DM_ext
    #         - dm_ext_th_values: Valores teóricos de DM_ext para DM_IGM
    #         - n_samples: Número de amostras Monte Carlo
    #         - bandwidth: Bandwidth para o estimador de densidade Kernel (KDE)
        
    #     Retorna:
    #         - pdf_cmy_values: Array com os valores da PDF de DM_ext
    #     """

    #     def compute_cmy_pdf(z_val, dm_ext_th, sigma, dm_obs):
    #         """
    #         Calcula a PDF para um único valor de redshift, com Monte Carlo.
    #         """
    #         # Amostras de DM_IGM
    #         dm_igm_samples = norm(loc=dm_ext_th, scale=sigma).rvs(size=n_samples)
            
    #         # Amostras de DM_host
    #         dm_host_mean = A * (1 + z_val) ** beta
    #         sigma_host = 1.24
    #         dm_host_samples = lognorm(s=sigma_host, scale=dm_host_mean).rvs(size=n_samples)
            
    #         # Calcula DM_ext = DM_IGM + DM_host / (1 + z)
    #         dm_ext_samples = dm_igm_samples + dm_host_samples / (1 + z_val)
            
    #         # Estima a PDF no ponto dm_obs usando KDE
    #         kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(dm_ext_samples[:, np.newaxis])
    #         log_pdf = kde.score_samples(np.array([dm_obs])[:, np.newaxis])
    #         return log_pdf[0]  # Retorna a densidade em dm_obs

    #     # # Cálculo da PDF para cada valor de entrada (sem paralelização)
    #     # pdf_cmy_values = [compute_cmy_pdf(z_val, dm_ext_th, sigma, dm_obs)
    #     #                 for z_val, dm_ext_th, sigma, dm_obs in zip(z, dm_ext_th_values, sigma_igm, dm_ext_obs)]

    #     # Paralelizar o cálculo da PDF com joblib
    #     pdf_cmy_values = Parallel(n_jobs=40)(
    #         delayed(compute_cmy_pdf)(z_val, dm_ext_th, sigma, dm_obs)
    #         for z_val, dm_ext_th, sigma, dm_obs in zip(z, dm_ext_th_values, sigma_igm, dm_ext_obs)
    #     )
        
    #     # Retorna a PDF como um array
    #     return np.array(pdf_cmy_values)

    
    # def DM_ext_cmy_pdf(self, z, A, beta, sigma_igm, dm_ext_obs, dm_ext_th_values, n_samples=66):
    
    #     # Função auxiliar para o cálculo da integral para cada valor de z_val
    #     def compute_cmy_monte_carlo(z_val, dm_ext_th, sigma, dm_obs):
    #         # Gerar amostras de X (normal) e Y (lognormal)
    #         X_samples = np.random.normal(dm_ext_th - A * (1 + z_val) ** (beta - 1), sigma, n_samples)  # DM_IGM ~ Normal
    #         Y_samples = np.random.lognormal(np.log(63.55), 1.24, n_samples)  # DM_HG ~ Lognormal

    #         # Calcular Z = X + Y para cada amostra
    #         Z_samples = X_samples + A * (1 + z_val)**(beta - 1) * Y_samples

    #         # Estimar a PDF de Z usando a média das amostras
    #         pdf_estimate = np.mean(np.exp(-0.5 * ((Z_samples - dm_obs)**2) / (sigma**2)))

    #         return pdf_estimate
        
    #     # Paralelizar o cálculo para cada valor de z
    #     pdf_cmy_values = [compute_cmy_monte_carlo(z_val, dm_ext_th, sigma, dm_obs)
    #                     for z_val, dm_ext_th, sigma, dm_obs in zip(z, dm_ext_th_values, sigma_igm, dm_ext_obs)]

    #     # Retornar o resultado como um array numpy
    #     return np.array(pdf_cmy_values)

    def integral_solve(self, z, A, beta, Omega_bh2, H0, f_IGM, q0, j0, sigma_igm, dm_ext_obs):

        dm_igm_th_cmy = self.DM_IGM_cmy(z, Omega_bh2, H0, f_IGM, q0, j0)

        def integrand_cmy(x):
                # Parte Gaussiana para DM_IGM usando scipy.stats.norm
                pdf_igm = np.exp(-((dm_ext_obs - np.exp(x) / (1 + z) - dm_igm_th_cmy)**2) / (2 * sigma_igm**2))

                mu_host = np.log(A * (1 + z) ** beta)
                sigma_host = 50 / (1 + z)
                pdf_host = np.exp(-((x - mu_host)**2) / (2 * sigma_host**2))
                normalization = 1 / (2 * np.pi * sigma_igm * sigma_host)

                return normalization * pdf_igm * pdf_host 
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            warnings.simplefilter("ignore", RuntimeWarning)

            # Executar a integração
            # Limites da integração
            lower_limit = -np.inf
            upper_limit = np.log(dm_ext_obs)
            integral_result, _ = [quad(integrand_cmy, lower_limit, upper_limit, epsabs=1e-5, epsrel=1e-5)]
        return np.array(integral_result)
    
    
    def H_cmy(self, z, H0, q0, j0):

        H_cosmo = H0 * (1 + (1 + q0) * z + 0.5 * (j0 - q0 ** 2) * z ** 2)
                        #  + 0.16 * (3 * q0 ** 2 + q0 ** 3 - 3 * j0 - 4 * j0 * q0 - s0) * z ** 3)

        return H_cosmo
     
    def D_L_cmy(self, z, H0, q0, j0):

        d_l = (299792 / H0) * (z + 0.5 * (1 - q0) * z ** 2
                                + 0.16 * (3 * q0 ** 2 + q0 - 1 - j0) * z ** 3)
        #  + 0.0416 * (2 - 2 * q0 - 15 * q0 ** 2 - 15 * q0 ** 3 + 5 * j0 + 10 * q0 * j0 + s0) * z ** 4)

        return d_l
    
    def Mu_cmy(self, z, H0, q0, j0):

        lumi_std = self.D_L_cmy(z, H0, q0, j0)
        
        return 5 * np.log10(lumi_std) + 25 
        


  
import numpy as np
import pandas as pd
from scipy import linalg

# H(z) Observational data and its errors

class H_data:

    def H_z_data(self):
        
        data_path_H = 'data/Hz35data.txt'
        Hz_data = pd.read_csv(data_path_H, delim_whitespace=True)

        self.z_val = Hz_data['z']
        self.H_z = Hz_data['H(z)']
        self.errors = Hz_data['sigma']
        
        return  self.z_val, self.H_z, self.errors


# FRB data and erros associated (16 point of date)
# Eur. Phys. J. C (2023) 83:138
# DOI: https://doi.org/10.1140/epjc/s10052-023-11275-7

# FRB data and erros associated

class FRB_data:

    def __init__(self, n_frb):
        self.n_frb: int = n_frb

    def select_data(self):

        if self.n_frb == 16:
    
            z_obs = np.array([0.0337, 0.098, 0.1178, 0.16, 0.19273, 0.234, 0.2365, 0.2432, 0.291, 0.3214, 0.3305, 
                            0.3688, 0.378, 0.4755, 0.522, 0.66])

            DM_obs = np.array([348.8, 413.52, 338.7, 380.25, 557.0, 506.92, 504.13, 297.5, 363.6, 361.42, 536.0, 
                            577.8, 321.4, 589.27, 593.1, 760.8])

            DM_obs_error = np.array([0.2, 0.5, 0.5, 0.4, 2.0, 0.04, 2.0, 0.05, 0.3, 0.06, 8.0, 0.02, 0.2, 0.03, 0.4, 0.6])

            DM_ISM_obs = np.array([200.0, 123.2, 37.2, 27.0, 188.0, 44.7, 38.0, 33.0, 57.3, 40.5, 152.0, 36.0, 57.83, 
                            102.0, 56.4, 37.0])
            
            # DM of the Milky Way halo
            DM_MW_halo = 50.0

            # Observed local DM and its error
            DM_MW_obs = DM_MW_halo + DM_ISM_obs
            DM_MW_obs_error = 10.0

            # Host galaxy error
            DM_host_error = 50 / (1 + z_obs)
            #DM_host_error = 50 

            # DM_IGM error
            DM_IGM_error = 173.8 * z_obs ** 0.4

            # Observed extragalactic DM and its error
            DM_obs_ext = DM_obs - DM_MW_obs
            DM_obs_ext_error = np.sqrt(DM_obs_error ** 2 + DM_MW_obs_error ** 2 + DM_IGM_error ** 2 + DM_host_error ** 2)
            #DM_obs_ext_error = np.sqrt(DM_obs_error ** 2 + DM_MW_obs_error ** 2 + DM_host_error ** 2)

            return z_obs, DM_obs_ext, DM_obs_ext_error

        
        elif self.n_frb == 50:
            # Load your data
            data_path_frb_50 = 'data/frb_50data.txt'
            dm_frb = np.loadtxt(data_path_frb_50, skiprows=1, usecols=range(1, 5))
            #dm_frb = np.loadtxt('data/frb_50data_symetric_errors.txt', skiprows=1, usecols=range(1, 5))
            z_obs = dm_frb[:, 0]
            DM_obs = dm_frb[:, 1]
            error_plus = dm_frb[:, 2]
            error_minus = dm_frb[:, 3]
            DM_ISM_obs = np.array([200.0, 123.2, 37.2, 27.0, 188.0, 44.7, 38.0, 33.0, 57.3, 40.5, 152.0, 36.0, 57.83, 
                            102.0, 56.4, 37.0])
            DM_ISM_obs_new = np.mean(DM_ISM_obs)

            # DM of the Milky Way halo
            DM_MW_halo = 50.0

            # Observed local DM and its error
            DM_MW_obs = DM_MW_halo + DM_ISM_obs_new
            DM_MW_obs_error = 10.0

            # Host galaxy error
            DM_host_error = 50 / (1 + z_obs)
            #DM_host_error = 50

            # DM_IGM error
            DM_IGM_error = 173.8 * z_obs ** 0.4

            # Observed extragalactic DM and its error
            DM_obs_ext = DM_obs - DM_MW_obs
            DM_obs_ext_error_plus = np.sqrt(error_plus ** 2 + DM_MW_obs_error ** 2 + DM_IGM_error ** 2 + DM_host_error ** 2)
            DM_obs_ext_error_minus = np.sqrt(error_minus ** 2 + DM_MW_obs_error ** 2 + DM_IGM_error ** 2 + DM_host_error ** 2)

            return z_obs, DM_obs_ext, DM_obs_ext_error_plus, DM_obs_ext_error_minus
        
        elif self.n_frb == 66:
            # Load your data
            data_path_frb_66 = 'data/frb_66data.txt'
            dm_frb = np.loadtxt(data_path_frb_66, skiprows=1, usecols=range(2, 6))
            #dm_frb = np.loadtxt('data/frb_50data_symetric_errors.txt', skiprows=1, usecols=range(1, 5))
            z_obs = dm_frb[:, 0]
            DM_obs = dm_frb[:, 1]
            DM_obs_error = dm_frb[:, 2]

            # Observed local DM and its error
            DM_MW_obs = dm_frb[:, 3]
            DM_MW_obs_error = 10.0

            # Host galaxy error
            DM_host_error = 50 / (1 + z_obs)
            #DM_host_error = 50

            # DM_IGM error
            DM_IGM_error = 173.8 * z_obs ** 0.4

            # Observed extragalactic DM and its error
            DM_obs_ext = DM_obs - DM_MW_obs
            DM_obs_ext_error = np.sqrt(DM_obs_error ** 2 + DM_MW_obs_error ** 2 + DM_IGM_error ** 2 + DM_host_error ** 2)
            #DM_obs_ext_error = np.sqrt(DM_obs_error ** 2 + DM_MW_obs_error ** 2 + DM_host_error ** 2)

            return z_obs, DM_obs_ext, DM_obs_ext_error
        else:
            raise ValueError("Invalid number of FRBs. Choose 16, 50 or 64.")
        

class SNe_data:
    
    def __init__(self, sample_sne):
        self.sample_sne: str =  sample_sne

    def load_data(self):

        if self.sample_sne == 'Pantheon+':

            # Carregar os dados do SNe
            sne_data = pd.read_csv('data/Pantheon+SH0ES.dat', delim_whitespace=True)
            
            # Extrair as colunas desejadas
            self.z_sne = sne_data['zCMB']
            self.mu_sne = sne_data['m_b_corr'] 
            
            # Ler a matriz de covariância
            with open('data/Pantheon+SH0ES_STAT+SYS.cov', 'r') as f:
                # Lê o número de linhas/colunas da matriz (primeira linha)
                N = int(f.readline().strip())
                
                # Lê os dados da matriz achatada (NxN elementos em uma única coluna)
                data = np.loadtxt(f)
            
                # Reconstrói a matriz de covariância NxN
                cov_matrix = np.reshape(data, (N, N))

                # Calcula a inversa usando decomposição de Cholesky
                self.cov_inv = linalg.cho_solve(linalg.cho_factor(cov_matrix), 
                           np.identity(cov_matrix.shape[0]))
                
            return self.z_sne, self.mu_sne, self.cov_inv
        
        if self.sample_sne == 'Union3':
            
            # Carregar os dados do SNe
            sne_data = pd.read_csv('data/lcparam_full.txt', delim_whitespace=True)
            
            # Extrair as colunas desejadas
            self.z_sne = sne_data['zcmb']
            self.mu_sne = sne_data['mb']

            # Ler a matriz de covariância
            with open('data/mag_covmat.txt', 'r') as f:
                # Lê o número de linhas/colunas da matriz (primeira linha)
                N = int(f.readline().strip())
                
                # Lê os dados da matriz achatada (NxN elementos em uma única coluna)
                data = np.loadtxt(f)
            
                # Reconstrói a matriz de covariância NxN
                cov_matrix = np.reshape(data, (N, N))

                # Calcula a inversa usando decomposição de Cholesky
                self.cov_inv = linalg.cho_solve(linalg.cho_factor(cov_matrix), 
                           np.identity(cov_matrix.shape[0]))
                
            return self.z_sne, self.mu_sne, self.cov_inv






        
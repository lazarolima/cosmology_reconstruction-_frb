import numpy as np
import pandas as pd

# H(z) Observational data and its errors

class H_data:

    def H_z_data(self):

        """z_val = np.array([0.17, 0.179, 0.199, 0.2, 0.27, 0.28, 0.352, 0.3802, 0.4, 0.4004, 0.4247, 0.44497,
                         0.4783, 0.48, 0.593, 0.68, 0.781, 0.875, 0.88, 0.9, 1.037, 1.3, 1.363, 1.43, 1.53, 1.75, 1.965])
    
        H_z = np.array([83, 75, 75, 72.9, 77, 88.8, 83, 83, 95, 77, 87.1, 92.8, 
                         80.9, 97, 104, 92, 105, 125, 90, 117, 154, 168, 160, 177, 140, 202, 186.5])

        errors = np.array([8, 4, 5, 29.6, 14, 36.6, 14, 13.5, 17, 10.2, 11.2, 12.9, 9, 62, 13, 8, 12,
                         17, 40, 23, 20, 17, 33.6, 18, 14, 40, 50.4])"""
        
        Hz_data = pd.read_csv("data/Hz35data.txt", delim_whitespace=True)

        z_val = Hz_data['z']
        H_z = Hz_data['H(z)']
        errors = Hz_data['sigma']
        
        return  z_val, H_z, errors


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
            dm_frb = np.loadtxt('data/frb_50data.txt', skiprows=1, usecols=range(1, 5))
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
            dm_frb = np.loadtxt('data/frb_66data.txt', skiprows=1, usecols=range(2, 6))
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
            sne_data = pd.read_csv("data/Pantheon+SH0ES.dat", delim_whitespace=True)
            
            # Extrair as colunas desejadas
            self.z_sne = sne_data['zCMB']
            self.mu_sne = sne_data['MU_SH0ES']
            self.dmb_sne = sne_data['MU_SH0ES_ERR_DIAG']
            cov = self.dmb_sne ** 2
            dcov = np.diag(cov)
            
            # Ler a matriz de covariância
            with open("data/Pantheon+SH0ES_STAT+SYS.cov", 'r') as f:
                # Lê o número de linhas/colunas da matriz (primeira linha)
                N = int(f.readline().strip())
                
                # Lê os dados da matriz achatada (NxN elementos em uma única coluna)
                data = np.loadtxt(f)
            
                # Reconstrói a matriz de covariância NxN
                self.cov_matrix = np.reshape(data, (N, N)) + dcov
                
            return self.z_sne, self.mu_sne, self.cov_matrix
        
        if self.sample_sne == 'Union3':
            
            # Carregar os dados do SNe
            sne_data = pd.read_csv("data/lcparam_full.txt", delim_whitespace=True)
            
            # Extrair as colunas desejadas
            self.z_sne = sne_data['zcmb']
            self.mu_sne = sne_data['mb']  + 19.214
            self.dmb_sne = sne_data['dmb']
            cov = self.dmb_sne ** 2
            dcov = np.diag(cov)

            # Ler a matriz de covariância
            with open("data/mag_covmat.txt", 'r') as f:
                # Lê o número de linhas/colunas da matriz (primeira linha)
                N = int(f.readline().strip())
                
                # Lê os dados da matriz achatada (NxN elementos em uma única coluna)
                data = np.loadtxt(f)
            
                # Reconstrói a matriz de covariância NxN
                self.cov_matrix = np.reshape(data, (N, N)) + dcov
                
            return self.z_sne, self.mu_sne, self.cov_matrix


"""import pandas as pd
import numpy as np

class SNe_data:
    
    def __init__(self):
        pass

    def load_data(self):
        # Carregar os dados do SNe
        sne_data = pd.read_csv("data/Pantheon+SH0ES.dat", delim_whitespace=True)
        
        # Extrair as colunas desejadas
        self.zCMB = sne_data['zCMB']
        self.mu_sne = sne_data['MU_SH0ES']
        self.is_calibrator = sne_data['IS_CALIBRATOR'].astype(bool)  # Para verificar se é calibrador
        
        # Filtrar dados que atendem a um critério específico
        self.ww = (self.zCMB > 0.01) | self.is_calibrator  # Máscara para os dados relevantes

        # Filtrar dados com base na máscara
        self.zCMB_filtered = self.zCMB[self.ww]
        self.mu_sne_filtered = self.mu_sne[self.ww]

        # Ler a matriz de covariância
        cov_matrix = self.load_covariance()

        # Filtrar a matriz de covariância com base na máscara
        self.cov_matrix_filtered = cov_matrix[self.ww, :][:, self.ww]

        return self.zCMB_filtered, self.mu_sne_filtered, self.cov_matrix_filtered

    def load_covariance(self):
        #Carregar a matriz de covariância a partir do arquivo.
        with open("data/Pantheon+SH0ES_STAT+SYS.cov", 'r') as file:
            # Ler o número de linhas e colunas
            N = int(file.readline().strip())
            # Ler a matriz em formato de lista
            cov_matrix = np.zeros((N, N))
            for i in range(N):
                cov_matrix[i] = np.array(file.readline().strip().split(), dtype=float)
        
        return cov_matrix"""


        
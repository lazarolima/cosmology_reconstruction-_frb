import numpy as np
import pandas as pd
from scipy import linalg

# H(z) Observational data and its errors

class H_data:

    def H_z_data(self):
        # Carrega os dados do arquivo
        Hz_data = np.loadtxt('data/Hz35data.txt', skiprows=1)

        # Extraindo z, H(z) e erros
        self.z_val = Hz_data[:, 0]
        self.H_z = Hz_data[:, 1]
        self.errors = Hz_data[:, 2]

        # Aplicando a máscara para z < 1
        mask = self.z_val < 1.018
        self.z_val = self.z_val[mask]
        self.H_z = self.H_z[mask]
        self.errors = self.errors[mask]

        return self.z_val, self.H_z, self.errors



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
            z_obs = dm_frb[:, 0]
            DM_obs = dm_frb[:, 1]
            DM_obs_error = dm_frb[:, 2]

            # Observed local DM and its error
            DM_MW_obs = dm_frb[:, 3]
            DM_MW_obs_error = 10.0

            # Host galaxy error
            DM_host_error = 50 / (1 + z_obs)
            # DM_host_error = 50

            # DM_IGM error
            DM_IGM_error = 173.8 * z_obs ** 0.4

            # Observed extragalactic DM and its error
            DM_obs_ext = DM_obs - DM_MW_obs
            DM_obs_ext_error = np.sqrt(DM_obs_error ** 2 + DM_MW_obs_error ** 2 + DM_IGM_error ** 2 + DM_host_error ** 2)
            # DM_obs_ext_error = np.sqrt(DM_obs_error ** 2 + DM_MW_obs_error ** 2 + DM_host_error ** 2)
            # DM_obs_ext_error = np.sqrt(DM_obs_error ** 2 + DM_MW_obs_error ** 2)
            # DM_obs_ext_error = np.sqrt(DM_MW_obs_error ** 2 + DM_IGM_error ** 2 + DM_host_error ** 2)

            return z_obs, DM_obs_ext, DM_obs_ext_error #DM_IGM_error
        
        elif self.n_frb == 5000:
            # Load your data
            data_path_frb_mock = 'data/redshift_dist_mock_frblip.txt'
            dm_frb = np.loadtxt(data_path_frb_mock, skiprows=1)
            z_obs = dm_frb[:, 0]
            DM_obs_ext = dm_frb[:, 1]

            # Observed local DM and its error
            DM_MW_obs_error = 10.0

            # Host galaxy error
            DM_host_error = 50 / (1 + z_obs)
            # DM_host_error = 50

            DM_obs_error = 0.5

            # DM_IGM error
            DM_IGM_error = 173.8 * z_obs ** 0.4

            DM_obs_ext_error = np.sqrt(DM_obs_error ** 2 + DM_MW_obs_error ** 2 + DM_IGM_error ** 2 + DM_host_error ** 2)
            # DM_IGM_error = np.sqrt(DM_obs_error ** 2 + DM_MW_obs_error ** 2 + DM_host_error ** 2)
            # DM_obs_ext_error = np.sqrt(DM_obs_error ** 2 + DM_MW_obs_error ** 2)

            return z_obs, DM_obs_ext, DM_obs_ext_error
        else:
            raise ValueError("Invalid number of FRBs. Choose 16, 50, 64 or 5000.")
        

class SNe_data:
    
    def __init__(self, sample_sne: str):
        self.sample_sne = sample_sne

    def load_data(self):
        """
        Carrega os dados do conjunto de SNe escolhido, realiza filtros necessários e
        calcula a matriz inversa de covariância.
        """
        if self.sample_sne == 'Pantheon+SH0ES':

            # Carregar os dados de supernovas
            sne_data = pd.read_csv('data/Pantheon+SH0ES.dat', sep='\s+')

            # Filtrar SNe com zCMB > 0.01 ou usadas como calibradoras (Cefeidas) e também z < 1
            #zmask = ((sne_data['zCMB'] > 0.01) | (sne_data['IS_CALIBRATOR'].astype(bool))) & (sne_data['zCMB'] < 1.018)
            zmask = ((sne_data['zCMB'] > 0.01)) & (sne_data['zCMB'] < 1.018)
            filtered_data = sne_data[zmask]

            # Extrair colunas relevantes
            self.z_sne = filtered_data['zCMB'].values
            #self.z_alt = filtered_data['zHEL'].values  # Alternativa redshift heliocêntrico
            self.mu_sne = filtered_data['MU_SH0ES'].values # Magnitude corrigida + 19.245 (Mb)
            #self.mu_sigma = filtered_data['m_b_corr_err_DIAG'].values   
            
            # Ler a matriz de covariância 
            with open('data/Pantheon+SH0ES_STAT+SYS.cov', 'r') as f:
                N = int(f.readline().strip())  # Número de linhas/colunas
                data = np.loadtxt(f)  # Ler os dados em formato achatado
                cov_matrix = np.reshape(data, (N, N))  # Reconstituir matriz NxN

            # Filtrar a matriz de covariância de acordo com a máscara aplicada nos dados
            self.cov_matrix = cov_matrix[np.ix_(zmask, zmask)] 

            # Selecionar os índices filtrados
            # idx_filtered = np.where(zmask)[0]
            # self.cov_matrix = cov_matrix[np.ix_(idx_filtered, idx_filtered)] + self.mu_std

            # Calcular a matriz inversa de covariância usando decomposição de Cholesky
            self.cov_inv = linalg.cho_solve(
                linalg.cho_factor(self.cov_matrix), 
                np.identity(self.cov_matrix.shape[0])
            )
 
            # Retornar os valores processados
            #return self.z_sne, self.z_alt, self.mu_sne, self.cov_inv
            return self.z_sne, self.mu_sne, self.cov_inv
            #return self.z_sne, self.mu_sne, self.mu_sigma


        if self.sample_sne == 'DESY5':

            # Carregar os dados de supernovas
            sne_data = pd.read_csv('data/DES-SN5YR_HD.csv', sep=',')

            # Filtrar SNe com zCMB > 0.01 ou usadas como calibradoras (Cefeidas)
            zmask = (sne_data['zCMB'] > 0.01) & (sne_data['zCMB'] < 1.018)
            filtered_data = sne_data[zmask]

            # Extrair colunas relevantes
            self.z_sne = filtered_data['zCMB'].values
            self.mu_sne = filtered_data['MU'].values
            # self.z_sne = sne_data['zCMB']
            # self.mu_sne = sne_data['MU']
            
            # Ler a matriz de covariância 
            with open('data/covsys_000.txt', 'r') as f:
                N = int(f.readline().strip())  # Número de linhas/colunas
                data = np.loadtxt(f)  # Ler os dados em formato achatado
                cov_matrix = np.reshape(data, (N, N))  # Reconstituir matriz NxN

            # Filtrar a matriz de covariância de acordo com a máscara aplicada nos dados
            self.cov_matrix = cov_matrix[np.ix_(zmask, zmask)] 

            # Calcular a matriz inversa de covariância usando decomposição de Cholesky
            # self.cov_inv = linalg.cho_solve(
            #     linalg.cho_factor(self.cov_matrix), 
            #     np.identity(self.cov_matrix.shape[0])
            # )
 
            # Retornar os valores processados
            return self.z_sne, self.mu_sne, self.cov_matrix
            #return self.z_sne, self.mu_sne, self.mu_sigma


        if self.sample_sne == 'Union3':
            
            # Carregar os dados do SNe
            sne_data = pd.read_csv('data/lcparam_full.txt', sep='\s+')
            
            # Extrair as colunas desejadas
            self.z_sne = sne_data['zcmb'].values
            self.mu_sne = sne_data['mb'].values + 19.245

            # Ler a matriz de covariância
            with open('data/mag_covmat.txt', 'r') as f:
                # Lê o número de linhas/colunas da matriz (primeira linha)
                N = int(f.readline().strip())
                
                # Lê os dados da matriz achatada (NxN elementos em uma única coluna)
                data = np.loadtxt(f)
            
                # Reconstrói a matriz de covariância NxN
                self.cov_matrix = np.reshape(data, (N, N))

                # Calcula a inversa usando decomposição de Cholesky
                # self.cov_inv = linalg.cho_solve(linalg.cho_factor(cov_matrix), 
                #            np.identity(cov_matrix.shape[0]))
                
            return self.z_sne, self.mu_sne, self.cov_matrix #self.cov_inv






        
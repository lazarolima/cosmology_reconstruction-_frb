import numpy as np
import pandas as pd

# Carregar os dados do SNe
sne_data = pd.read_csv("data/lcparam_full.txt", delim_whitespace=True)

# Extrair as colunas desejadas
z_sne = sne_data['zcmb']
mu_sne = sne_data['mb']  + 19.214
dmb_sne = sne_data['dmb']
cov = dmb_sne ** 2
dcov = np.diag(cov)

# Ler a matriz de covariância
with open("data/mag_covmat.txt", 'r') as f:
    # Lê o número de linhas/colunas da matriz (primeira linha)
    N = int(f.readline().strip())
    
    # Lê os dados da matriz achatada (NxN elementos em uma única coluna)
    data = np.loadtxt(f)

    # Reconstrói a matriz de covariância NxN
    cov_matrix = np.reshape(data, (N, N)) + dcov

cov_inv = np.linalg.inv(cov_matrix)
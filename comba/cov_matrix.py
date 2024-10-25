import numpy as np
from scipy import linalg

# Lê e reconstrói a matriz de covariância
with open("data/Pantheon+SH0ES_STAT+SYS.cov", 'r') as f:
    N = int(f.readline().strip())  # Número de linhas/colunas da matriz
    data = np.loadtxt(f)           # Dados achatados
    cov_matrix = np.reshape(data, (N, N))  # Reconstrói a matriz NxN

# Calcula a inversa usando decomposição de Cholesky
cov_inv = linalg.cho_solve(linalg.cho_factor(cov_matrix), 
                           np.identity(cov_matrix.shape[0]))

# Salvar a matriz inversa no formato desejado
with open("data/cov_inverse.dat", 'w') as f:
    f.write(f"{N}\n")  # Escreve o número de linhas/colunas
    np.savetxt(f, cov_inv.flatten(), fmt='%.6e')  # Salva a matriz achatada

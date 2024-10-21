import numpy as np

# Carrega a função reconstruída da ANN
data = np.loadtxt('data/DM_IGM_detected_bingo+mirror_4m_alpha=-1.5_5yrs.txt', skiprows=1)

z = data[:, 0]
dm = data[:, 1]

def sigma(z): 
    return 0.2 * dm / np.sqrt(z)
    #return 173.8 * z ** 0.4

sigma_IGM = sigma(z)

# Storage the data
data_IGM_candidates = np.column_stack((z, dm, sigma_IGM))
np.savetxt('data/DM_IGM_detected_bingo+mirror_4m_alpha=-1.5_5yrs_with_erros.txt', data_IGM_candidates, fmt='%f', delimiter='\t', header='z_cadidates \t DM_IGM_candidates \t sigma_dm')
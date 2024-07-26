import numpy as np

# Dados observacionais de H(z) e seus erros

def z_func():
    return np.array([0.17, 0.179, 0.199, 0.2, 0.27, 0.28, 0.352, 0.3802, 0.4, 0.4004, 0.4247, 0.44497,
                     0.4783, 0.48, 0.593, 0.68, 0.781, 0.875, 0.88, 0.9, 1.037, 1.3, 1.363, 1.43, 1.53, 1.75, 1.965])

def H_func(): 
    return np.array([83, 75, 75, 72.9, 77, 88.8, 83, 83, 95, 77, 87.1, 92.8, 
                  80.9, 97, 104, 92, 105, 125, 90, 117, 154, 168, 160, 177, 140, 202, 186.5])

def errors_func():
    return np.array([8, 4, 5, 29.6, 14, 36.6, 14, 13.5, 17, 10.2, 11.2, 12.9, 9, 62, 13, 8, 12,
                   17, 40, 23, 20, 17, 33.6, 18, 14, 40, 50.4])


# FRB data and erros associated (16 point of date)
def z_frb():
    return np.array([0.0337, 0.098, 0.1178, 0.16, 0.19273, 0.234, 0.2365, 0.2432, 0.291, 0.3214, 0.3305, 
                     0.3688, 0.378, 0.4755, 0.522, 0.66])

def DM_frb():
    return np.array([348.8, 413.52, 338.7, 380.25, 557.0, 506.92, 504.13, 297.5, 363.6, 361.42, 536.0, 
                     577.8, 321.4, 589.27, 593.1, 760.8])

def error_frb():
    return np.array([0.2, 0.5, 0.5, 0.4, 2.0, 0.04, 2.0, 0.05, 0.3, 0.06, 8.0, 0.02, 0.2, 0.03, 0.4, 0.6])

# DM meio interestelar observado
def DM_ISM_frb():
    return np.array([200.0, 123.2, 37.2, 27.0, 188.0, 44.7, 38.0, 33.0, 57.3, 40.5, 152.0, 36.0, 57.83, 
                     102.0, 56.4, 37.0])

# FRB data and erros associated
z_obs = z_frb()
DM_obs = DM_frb()
DM_obs_error = error_frb()

# DM meio interestelar observado
DM_ISM_obs = DM_ISM_frb()

# DM do halo da Via Láctea
DM_MW_halo = 50.0

# DM local observado e seu erro
DM_MW_obs = DM_MW_halo + DM_ISM_obs
DM_MW_obs_error = 10.0

# DM extragaláctico observado e seu erro
DM_obs_ext = DM_obs - DM_MW_obs
DM_obs_ext_error = np.sqrt(DM_obs_error ** 2 + DM_MW_obs_error ** 2)

# DM da galáxia hospedeira e seu erro
DM_host = 100 / (1 + z_obs)
DM_host_error = 50 / (1 + z_obs)

# DM do meio intergaláctico observado e seu erro
DM_IGM_obs = DM_obs_ext - DM_host
DM_IGM_obs_error = np.sqrt(DM_obs_ext_error ** 2 + DM_host_error ** 2)
import numpy as np
from equations import FiducialModel

# Defining a seed for reproducibility
np.random.seed(42)

# Defining a fiducial model 
fiducial_model = FiducialModel()

class RedshiftSimulation:
    def __init__(self, n, z_max):
        self.n = n
        self.z_max = z_max
        self.new_z = None
        self.DM_IGM_sim = None
        self.sigma_DM_IGM = None
        self.sigma_obs: float = 0.5
        self.sigma_MW = 10

    def generate_redshifts(self):
        uniform_samples = np.random.uniform(0, 1, self.n)
        self.new_z = -np.log(1 - uniform_samples * (1 - np.exp(-self.z_max)))
        return self.new_z
    
    def sigma_DM_IGM_sim(self):
        if self.new_z is None:
            self.generate_redshifts()
        sigma_IGM = 173.8 * self.new_z ** 0.4
        self.sigma_DM_IGM = np.sqrt(self.sigma_obs**2 + self.sigma_MW**2 + sigma_IGM**2 + (50/(1+self.new_z))**2)
        return self.sigma_DM_IGM
    
    def simulate_DM_IGM(self):
        if self.sigma_DM_IGM is None:
            self.sigma_DM_IGM_sim()
        DM_IGM = fiducial_model.DM_IGM(self.new_z)
        self.DM_IGM_sim = np.random.normal(loc=DM_IGM, scale=self.sigma_DM_IGM, size=self.n)
        return self.DM_IGM_sim

    def get_new_z(self):
        return self.new_z

    def get_DM_IGM_sim(self):
        return self.DM_IGM_sim
    
    def get_sigma_DM_IGM_sim(self):
        return self.sigma_DM_IGM
    


    
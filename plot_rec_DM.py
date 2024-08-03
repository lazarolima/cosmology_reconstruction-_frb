import obs_data as od
import matplotlib.pyplot as plt
import numpy as np
from equations import FiducialModel

class DMIGMReconstructionPlot:
    def __init__(self, gp_dm_igm, new_z, DM_IGM_sim):
        self.gp_dm_igm = gp_dm_igm
        self.new_z = new_z
        self.DM_IGM_sim = DM_IGM_sim
        self.mean, self.var, self.mean_deriv, self.var_deriv = gp_dm_igm.predict()
        self.z_pred = gp_dm_igm.z_pred()
        self.fiducial_model = FiducialModel()
        self.dm_igm_theory = self.fiducial_model.DM_IGM(self.z_pred.flatten())

    def plot(self, filename='DM_IGM_reconstructed.png', dpi=600):
        plt.figure(figsize=(8, 6))

        # Dados originais com barras de erro
        plt.errorbar(self.new_z, self.DM_IGM_sim, fmt='ro', color='purple', alpha=0.5, label='Mock data')
        plt.errorbar(od.z_obs, od.DM_IGM_obs, fmt='ro', color='k', alpha=1, label='Data')

        # Função predita pelo GP
        plt.plot(self.z_pred, self.mean, 'k-', label='GP Reconstruction', lw=2)

        # Adicionar curvas sombreadas de 1σ e 2σ
        plt.fill_between(self.z_pred.flatten(), 
                         self.mean.flatten() - 1*np.sqrt(self.var.flatten()), 
                         self.mean.flatten() + 1*np.sqrt(self.var.flatten()), 
                         alpha=0.5, color='k', label='1σ')
        plt.fill_between(self.z_pred.flatten(), 
                         self.mean.flatten() - 2*np.sqrt(self.var.flatten()), 
                         self.mean.flatten() + 2*np.sqrt(self.var.flatten()), 
                         alpha=0.3, color='gray', label='2σ')

        # Modelo fiducial
        plt.plot(self.z_pred, self.dm_igm_theory, 'b--', label='Fiducial model')

        plt.xlabel('Redshift ($z$)')
        plt.ylabel('$DM_{IGM}$ (pc/cm³)')
        plt.legend()
        plt.savefig(filename, dpi=dpi)
        plt.show()



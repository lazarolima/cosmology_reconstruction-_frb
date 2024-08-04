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
        #self.z_pred = gp_dm_igm.z_pred()
        self.z_pred = np.linspace(0, 2, 100)
        self.fiducial_model = FiducialModel()
        self.dm_igm_theory = self.fiducial_model.DM_IGM(self.z_pred)
        self.dm_igm_deriv = self.fiducial_model.I(self.z_pred)

    def plot_DM(self, filename='DM_IGM_reconstructed.png', dpi=600):
        
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

        plt.xlabel('Redshift ($z$)', fontsize=14)
        plt.ylabel('$DM_{IGM}$ (pc/cm³)', fontsize=14)
        plt.legend(fontsize=14)
        plt.savefig(filename, dpi=dpi)
        plt.show()

    def plot_dDM(self, filename='dDM_IGM_reconstructed.png', dpi=600):

        # Plotar as derivadas
        plt.figure(figsize=(8, 6))
        plt.plot(self.z_pred, self.mean_deriv.flatten(), 'k-', label='Derivative reconstrution')

        # Adicionar curvas sombreadas de 1σ e 2σ para a derivada
        plt.fill_between(self.z_pred.flatten(), 
                        self.mean_deriv.flatten() - 1*np.sqrt(self.var_deriv.flatten()), 
                        self.mean_deriv.flatten() + 1*np.sqrt(self.var_deriv.flatten()), 
                        alpha=0.3, color='k', label='1σ')
        plt.fill_between(self.z_pred.flatten(), 
                        self.mean_deriv.flatten() - 2*np.sqrt(self.var_deriv.flatten()), 
                        self.mean_deriv.flatten() + 2*np.sqrt(self.var_deriv.flatten()), 
                        alpha=0.2, color='gray', label='2σ')

        # Derivada de DM_IGM
        plt.plot(self.z_pred.flatten(), self.dm_igm_deriv, 'b--', label='Model derivative')

        plt.xlabel('Redshift (z)', fontsize=14)
        plt.ylabel('$dDM_{IGM}/dz$ (pc/cm$^{3}$)', fontsize=14)
        plt.legend(fontsize=14)
        plt.savefig(filename, dpi=dpi)
        plt.show()


class HReconstructionPlot:
    def __init__(self, gp_h, fiducial_model):
        self.gp_h = gp_h
        self.z_values = gp_h.z_values
        self.H_obs = gp_h.H_obs
        self.errors = gp_h.errors
        self.fiducial_model = fiducial_model
        self.mean, self.var, self.mean_deriv, self.var_deriv = gp_h.predict()
        #self.z_test = gp_h.z_pred()
        self.z_test = np.linspace(0, 2, 100)
        self.H_theory = fiducial_model.H_std(self.z_test)
    
    def plot(self, filename='H_reconstructed.png', dpi=600):
        plt.figure(figsize=(8, 6))

        # Dados originais com barras de erro
        plt.errorbar(self.z_values, self.H_obs, yerr=self.errors, fmt='o', capsize=4, color='k', label='Data')

        # Função predita pelo GP
        plt.plot(self.z_test, self.mean, 'k-', label='Reconstruction')

        # Adicionar curvas sombreadas de 1σ e 2σ
        plt.fill_between(self.z_test.flatten(), 
                         self.mean.flatten() - 1*np.sqrt(self.var.flatten()), 
                         self.mean.flatten() + 1*np.sqrt(self.var.flatten()), 
                         alpha=0.3, color='k', label='1σ')
        plt.fill_between(self.z_test.flatten(), 
                         self.mean.flatten() - 2*np.sqrt(self.var.flatten()), 
                         self.mean.flatten() + 2*np.sqrt(self.var.flatten()), 
                         alpha=0.2, color='gray', label='2σ')

        plt.plot(self.z_test, self.H_theory, 'b--', label='$\\Lambda$CDM model')

        plt.xlabel('Redshift ($z$)', fontsize=14)
        plt.ylabel('$H(z)$', fontsize=14)
        plt.legend()
        plt.savefig(filename, dpi=dpi)
        plt.show()
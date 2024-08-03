import matplotlib.pyplot as plt
import numpy as np

class HReconstructionPlot:
    def __init__(self, gp_h, z_values, H_obs, errors, fiducial_model):
        self.gp_h = gp_h
        self.z_values = z_values
        self.H_obs = H_obs
        self.errors = errors
        self.fiducial_model = fiducial_model
        self.mean, self.var, self.mean_deriv, self.var_deriv = gp_h.predict()
        self.z_test = gp_h.z_pred()
        self.H_theory = fiducial_model.H_padrao(self.z_test.flatten())
    
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

        plt.xlabel('Redshift (z)')
        plt.ylabel('H(z)')
        plt.legend()
        plt.savefig(filename, dpi=dpi)
        plt.show()

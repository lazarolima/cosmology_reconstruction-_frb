import numpy as np
import GPy
from mock import RedshiftSimulation
import obs_data as od

class GPReconstructionH:
    def __init__(self, z_values, H_obs, errors, kernel_params=None):
        self.z_values = z_values
        self.H_obs = H_obs
        self.errors = errors
        
        # Preparar os dados para o GP
        self.X = self.z_values.reshape(-1, 1)
        self.Y = self.H_obs.reshape(-1, 1)
        
        # Configurar o kernel
        if kernel_params is None:
            kernel_params = {'input_dim': 1, 'variance': 100., 'lengthscale': 0.1}
        
        self.kernel = GPy.kern.RBF(**kernel_params)
        self.model = GPy.models.GPRegression(self.X, self.Y, self.kernel)
        
        # Definir a variância do ruído
        self.model.Gaussian_noise.variance = np.mean(self.errors**2)
        
    def optimize(self, num_restarts=10, verbose=False):
        self.model.optimize_restarts(num_restarts=num_restarts, verbose=verbose)
        
    def z_pred(self, num_points=100):
        return np.linspace(0, 2, num_points).reshape(-1, 1)
        
    def predict(self):
        mean, var = self.model.predict(self.z_pred())
        mean_deriv, var_deriv = self.model.predict_jacobian(self.z_pred())
        return mean, var, mean_deriv, var_deriv

class GPReconstructionDMIGM:
    def __init__(self, n_new=None, z_max=None, kernel_params=None):

        # Configurar valores padrão para n_new e z_max
        if n_new is None:
            n_new = 500
        if z_max is None:
            z_max = 2

        # Criar uma instância da classe RedshiftSimulation
        self.sim = RedshiftSimulation(n_new, z_max)
        
        # Gerar novos dados
        self.sim.generate_redshifts()
        self.sim.simulate_DM_IGM()
        self.sim.sigma_DM_IGM_sim()  # Corrigido para chamar o método
        
        # Obter os resultados
        self.new_z = self.sim.get_new_z()
        self.DM_IGM_sim = self.sim.get_DM_IGM_sim()
        self.sigma_DM_IGM_sim = self.sim.get_sigma_DM_IGM_sim()

        # Salvar em arquivo.txt
        with open('data/DM_IGM_sim.txt', 'w') as file:
            for z, dm, sigma in zip(self.new_z, self.DM_IGM_sim, self.sigma_DM_IGM_sim):
                file.write(f"{z}\t{dm}\t{sigma}\n")
        
        # Preparar os dados para o GP
        self.X1 = self.new_z.reshape(-1, 1)
        self.Y1 = self.DM_IGM_sim.reshape(-1, 1)
        
        # Configurar o kernel
        if kernel_params is None:
            kernel_params = {'input_dim': 1, 'variance': 100., 'lengthscale': 0.1}
        
        self.kernel1 = GPy.kern.RBF(**kernel_params)
        self.model1 = GPy.models.GPRegression(self.X1, self.Y1, self.kernel1)
        
        # Definir a variância do ruído
        self.model1.Gaussian_noise.variance = np.mean(od.DM_IGM_obs_error**2)
        
    def optimize(self, num_restarts=10, verbose=False):
        self.model1.optimize_restarts(num_restarts=num_restarts, verbose=verbose)
        
    def z_pred(self, num_points=100):
        return np.linspace(0, 2, num_points).reshape(-1, 1)
        
    def predict(self):
        mean, var = self.model1.predict(self.z_pred())
        mean_deriv, var_deriv = self.model1.predict_jacobian(self.z_pred())
        return mean, var, mean_deriv, var_deriv
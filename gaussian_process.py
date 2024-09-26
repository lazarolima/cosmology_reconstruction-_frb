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
        
        #self.kernel = GPy.kern.RBF(**kernel_params)
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
        self.X = self.new_z.reshape(-1, 1)
        self.Y = self.DM_IGM_sim.reshape(-1, 1)
        
        # Configurar o kernel
        if kernel_params is None:
            kernel_params = {'input_dim': 1, 'variance': 100., 'lengthscale': 0.1}
        
        self.kernel1 = GPy.kern.RBF(**kernel_params)
        self.model1 = GPy.models.GPRegression(self.X, self.Y, self.kernel1)
        
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
    

# Same as above, but with simulation mock of bingo
class GPReconstructionDMIGM_noSim:
    def __init__(self, z_val, dm_val, error_val, kernel_params=None):
        self.z_val = z_val
        self.dm_val = dm_val
        self.error_val = error_val
        
        # Preparar os dados para o GP
        self.X = self.z_val.reshape(-1, 1)
        self.Y = self.dm_val.reshape(-1, 1)
        
        # Configurar o kernel
        if kernel_params is None:
            kernel_params = {'input_dim': 1, 'variance': 1., 'lengthscale': 1.}
        
        self.kernel = GPy.kern.RBF(**kernel_params)

        self.noise = np.mean(self.error_val**2)

        self.model = GPy.models.GPRegression(X=self.X, Y=self.Y, kernel=self.kernel,noise_var=self.noise)
        
        # Definir a variância do ruído
        #self.model.Gaussian_noise.variance = np.mean(self.error_val**2)

    def optimize(self, num_restarts=10, verbose=False):
        # Otimizar com um otimizador robusto
        self.model.optimize(optimizer='lbfgs', max_iters=1000)
        self.model.optimize_restarts(num_restarts=num_restarts, verbose=verbose, robust=True, parallel=True)
        
    def z_pred(self, num_points=100):
        return np.linspace(0, 2, num_points).reshape(-1, 1)
        
    def predict(self):
        mean, var = self.model.predict(self.z_pred())
        mean_deriv, var_deriv = self.model.predict_jacobian(self.z_pred())
        return mean, var, mean_deriv, var_deriv



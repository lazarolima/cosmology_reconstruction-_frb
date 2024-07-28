import numpy as np
import GPy
from obs_data import H_data

# Reconstrução de H(z)
z_values = H_data.z_func()
H_obs = H_data.H_func()
errors = H_data.errors_func()

# Preparar os dados para o GP
X = z_values.reshape(-1, 1)
Y = H_obs.reshape(-1, 1)

# Criar e treinar o modelo GP
kernel = GPy.kern.RBF(input_dim=1, variance=100., lengthscale=0.1)
m = GPy.models.GPRegression(X, Y, kernel)

# Definir a variância do ruído
m.Gaussian_noise.variance = np.mean(errors**2)

# Otimizar o modelo
m.optimize_restarts(num_restarts=10, verbose=False)

# Criar pontos para predição
def z_pred():
    return np.linspace(0, 2, 100).reshape(-1, 1)

# Prever a função e sua derivada
def pred():
    mean, var = m.predict(z_pred())
    mean_deriv, var_deriv = m.predict_jacobian(z_pred())
    return mean, var, mean_deriv, var_deriv

# Reconstrução de DM_IGM

from mock import RedshiftSimulation

# Parâmetros para gerar novos dados
n_new = 500  # Número de novos pontos de dados
z_max = 2  # Máximo redshift

# Criar uma instância da classe RedshiftSimulation
sim = RedshiftSimulation(n_new, z_max)

# Gerar novos dados
sim.generate_redshifts()
sim.simulate_DM_IGM()

# Obter os resultados
new_z = sim.get_new_z()
DM_IGM_sim = sim.get_DM_IGM_sim()

# Preparar os dados para o GP
X1 = new_z.reshape(-1, 1)
Y1 = DM_IGM_sim.reshape(-1, 1)

# Criar e treinar o modelo GP
kernel1 = GPy.kern.RBF(input_dim=1, variance=100., lengthscale=0.1)
m1 = GPy.models.GPRegression(X1, Y1, kernel1)

# Definir a variância do ruído
import obs_data as od

m1.Gaussian_noise.variance = np.mean(od.DM_IGM_obs_error**2)

# Otimizar o modelo
m1.optimize_restarts(num_restarts=10, verbose=False)

# Prever a função e sua derivada
def pred_new():
    mean1, var1 = m1.predict(z_pred())
    mean_deriv1, var_deriv1 = m1.predict_jacobian(z_pred())
    return mean1, var1, mean_deriv1, var_deriv1
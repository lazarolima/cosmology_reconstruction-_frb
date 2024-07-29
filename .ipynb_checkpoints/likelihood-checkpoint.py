from obs_data import H_data
from equations import H_Model
import numpy as np

z_values = H_data.z_func()
H_obs = H_data.H_func()
errors = H_data.errors_func()

# Defining the Priors
class Priors:

    parameters1 = ['$f_{IGM}$']

    @staticmethod
    def prior_transform1(cube):
        params = cube.copy()
        params[0] = cube[0]
        return params

    parameters2 = ['$f_{IGM}$', '$\\alpha$']

    @staticmethod
    def prior_transform2(cube):
        params = cube.copy()
        params[0] = cube[0]
        params[1] = 5 * cube[1]
        return params

    parameters3 = ['$f_{IGM}$', '$s$']

    @staticmethod
    def prior_transform3(cube):
        params = cube.copy()
        params[0] = cube[0]
        params[1] = 10 * cube[1] - 5
        return params

class LikelihoodFunction:

    @staticmethod
    def log_likelihood1(params):
        f_IGM = params
        # Defining a fiducial model for H(z)
        model = H_Model()
        y_model = model.H_p1(z_values, f_IGM=f_IGM)
        loglike = -0.5 * np.sum(((y_model - H_obs) / errors)**2)
        return loglike

    @staticmethod
    def log_likelihood2(params):
        f_IGM, alpha = params
        # Defining a fiducial model for H(z)
        model = H_Model()
        y_model = model.H_p2(z_values, f_IGM=f_IGM, alpha=alpha)
        loglike = -0.5 * np.sum(((y_model - H_obs) / errors)**2)
        return loglike

    @staticmethod
    def log_likelihood3(params):
        f_IGM, alpha = params
        # Defining a fiducial model for H(z)
        model = H_Model()
        y_model = model.H_p3(z_values, f_IGM=f_IGM, alpha=alpha)
        loglike = -0.5 * np.sum(((y_model - H_obs) / errors)**2)
        return loglike

    @staticmethod
    def log_likelihood4(params):
        f_IGM, s = params
        # Defining a fiducial model for H(z)
        model = H_Model()
        y_model = model.H_p4(z_values, f_IGM=f_IGM, s=s)
        loglike = -0.5 * np.sum(((y_model - H_obs) / errors)**2)
        return loglike
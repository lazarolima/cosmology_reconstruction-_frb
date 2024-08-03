import numpy as np

class LikelihoodFunction:
    
    def __init__(self, model_func):
        self.model_func = model_func
    
    def log_likelihood(self, params, z_values, H_obs, errors):
        y_model = self.model_func(z_values, *params)
        loglike = -0.5 * np.sum(((y_model - H_obs) / errors)**2)
        return loglike

class Priors:
    
    def __init__(self, param_names, intervals):
        self.param_names = param_names
        self.intervals = intervals

    def prior_transform(self, cube):
        params = cube.copy()
        for i, (low, high) in enumerate(self.intervals):
            params[i] = low + (high - low) * cube[i]
        return params


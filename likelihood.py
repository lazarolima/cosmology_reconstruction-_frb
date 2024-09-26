import numpy as np

class LikelihoodFunction:
    
    def __init__(self, model_func):
        self.model_func = model_func
    
    def log_likelihood(self, params, z_values, y_obs, errors=None, err_neg=None, err_pos=None):
        y_model = self.model_func(z_values, *params)

        if errors is None:
            # Caso assimétrico
            errors = np.where(y_model > y_obs, err_pos, err_neg)
            loglike = -0.5 * np.sum(((y_model - y_obs) / errors) ** 2)
        
            # Caso simétrico
        if err_neg is None or err_pos is None:
            loglike = -0.5 * np.sum(((y_model - y_obs) / errors) ** 2)            
        
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


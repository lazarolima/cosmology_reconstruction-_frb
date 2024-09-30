import numpy as np

"""class LikelihoodFunction:
    
    def __init__(self, model_func):
        self.model_func = model_func
    
    def log_likelihood(self, params, z_values, y_obs, errors=None, err_neg=None, err_pos=None):
        # Compute the model values
        y_model = self.model_func(z_values, *params)
        
        # Asymmetric case
        if errors is None and err_neg is not None and err_pos is not None:
            # Select the appropriate error based on the condition
            error_new = np.where(y_model > y_obs, err_pos, err_neg)
            loglike = -0.5 * np.sum(((y_model - y_obs) / error_new) ** 2)

        # Symmetric case
        elif errors is not None and err_neg is None and err_pos is None:
            loglike = -0.5 * np.sum(((y_model - y_obs) / errors) ** 2)

        # Invalid error configuration
        else:
            raise ValueError("Error: Provide either symmetric errors (errors) or asymmetric errors (err_neg and err_pos).")
        
        return loglike"""

class LikelihoodFunction:
    
    def __init__(self, model_func):
        self.model_func = model_func
    
    def log_likelihood(self, params, z_values, y_obs, errors=None, err_neg=None, err_pos=None):
        # Extração dos parâmetros
        f_IGM, DM_host_0, model_type, Omega_b, Omega_m, H_today, param = params

        # Compute the model values
        y_model = self.model_func(z_values, f_IGM, DM_host_0, model_type, Omega_b, Omega_m, H_today, param)
        
        # Asymmetric case
        if errors is None and err_neg is not None and err_pos is not None:
            # Select the appropriate error based on the condition
            error_new = np.where(y_model > y_obs, err_pos, err_neg)
            loglike = -0.5 * np.sum(((y_model - y_obs) / error_new) ** 2)

        # Symmetric case
        elif errors is not None and err_neg is None and err_pos is None:
            loglike = -0.5 * np.sum(((y_model - y_obs) / errors) ** 2)

        # Invalid error configuration
        else:
            raise ValueError("Error: Provide either symmetric errors (errors) or asymmetric errors (err_neg and err_pos).")
        
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


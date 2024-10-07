import numpy as np
from scipy import stats

class LikelihoodFunction:
    
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
        
        return loglike


"""class Priors:
    
    def __init__(self, param_names, intervals):
        self.param_names = param_names
        self.intervals = intervals

    def prior_transform(self, cube):
        params = cube.copy()
        for i, (low, high) in enumerate(self.intervals):
            params[i] = low + (high - low) * cube[i]
        return params"""

class Priors:
    
    def __init__(self, param_names, intervals, distributions):
        """
        param_names : list of parameter names
        intervals   : list of tuples, each defining the (low, high) bounds for uniform priors
        distributions : list of distributions ('uniform', 'gaussian', 'log-normal', etc.)
        """
        self.param_names = param_names
        self.intervals = intervals
        self.distributions = distributions

    def prior_transform(self, cube):
        """
        Transform unit cube values into prior distributions.
        cube : array of values in [0, 1] from ultranest
        """
        params = cube.copy()
        for i, dist in enumerate(self.distributions):
            low, high = self.intervals[i]
            
            if dist == 'uniform':
                # Prior uniforme simples
                params[i] = low + (high - low) * cube[i]
                
            elif dist == 'gaussian':
                # Prior Gaussiano: mapear do intervalo unitário [0, 1] para distribuição normal
                mu, sigma = (low + high) / 2, (high - low) / 6  # Média entre low e high e sigma aproximado
                params[i] = stats.norm.ppf(cube[i], loc=mu, scale=sigma)  # ppf é a inversa da CDF
            
            elif dist == 'log-normal':
                # Prior Log-normal: mapear para uma escala logarítmica
                mu, sigma = np.log((low + high) / 2), (np.log(high) - np.log(low)) / 6
                params[i] = np.exp(stats.norm.ppf(cube[i], loc=mu, scale=sigma))
                
            # Outras distribuições poderiam ser adicionadas aqui.
            # Por exemplo, Jeffreys, Beta, etc.

            else:
                raise ValueError(f"Dist must be uniform, gaussian, log-normal, etc. Got {dist}")
            
        return params




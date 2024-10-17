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
    

# New classes for join analysis (like FRBs + H(z) data) 

"""class JointLikelihoodFunction:

    def __init__(self, model_funcs):
        
        #Inicializa a função de verossimilhança conjunta.
        
        #:param model_funcs: Um dicionário onde as chaves são os nomes dos conjuntos de dados
        #                    e os valores são as funções do modelo correspondentes.
        
        self.model_funcs = model_funcs

    def log_likelihood(self, params, data_sets):
        total_loglike = 0
        
        for data_name, (z_values, y_obs, errors) in data_sets.items():
            model_func = self.model_funcs[data_name]
            
            # Filtra os parâmetros necessários para cada modelo
            filtered_params = {k: v for k, v in params.items() if k in model_func.__code__.co_varnames}
            
            y_model = model_func(z_values, **filtered_params)
            
            if isinstance(errors, tuple):  # Caso assimétrico
                err_neg, err_pos = errors
                error_new = np.where(y_model > y_obs, err_pos, err_neg)
            else:  # Caso simétrico
                error_new = errors
            
            loglike = -0.5 * np.sum(((y_model - y_obs) / error_new) ** 2)
            total_loglike += loglike
        
        return total_loglike"""

class JointLikelihoodFunction:

    def __init__(self, model_funcs):
        """
        Inicializa a função de verossimilhança conjunta.
        
        :param model_funcs: Um dicionário onde as chaves são os nomes dos conjuntos de dados
                            e os valores são as funções do modelo correspondentes.
        """
        self.model_funcs = model_funcs

    def log_likelihood(self, params, data_sets):
        total_loglike = 0
        
        for data_name, data_info in data_sets.items():
            model_func = self.model_funcs[data_name]
            
            # Filtra os parâmetros necessários para cada modelo
            filtered_params = {k: v for k, v in params.items() if k in model_func.__code__.co_varnames}
            
            # Para SNe, a entrada tem (z_values, y_obs, cov_matrix)
            if data_name == "SNe":
                z_values, y_obs, cov_matrix = data_info
                y_model = model_func(z_values, **filtered_params)
                
                # Calcula a diferença entre o modelo e a observação
                delta = y_model - y_obs
                
                # Calcula o chi^2 utilizando a matriz de covariância inversa
                cov_inv = np.linalg.inv(cov_matrix)
                chi2 = np.sum(np.dot(delta.T, np.dot(cov_inv, delta)))
                
                # Adiciona à verossimilhança total
                total_loglike += -0.5 * chi2
            
            # Para outros conjuntos de dados (FRBs, H_0), a entrada tem (z_values, y_obs, errors)
            else:
                z_values, y_obs, errors = data_info
                y_model = model_func(z_values, **filtered_params)
                
                # Caso de erros assimétricos
                if isinstance(errors, tuple):
                    err_neg, err_pos = errors
                    error_new = np.where(y_model > y_obs, err_pos, err_neg)
                else:  # Erros simétricos
                    error_new = errors
                
                # Cálculo da verossimilhança gaussiana simples
                loglike = -0.5 * np.sum(((y_model - y_obs) / error_new) ** 2)
                total_loglike += loglike
        
        return total_loglike


class JointPriors:
    def __init__(self, param_configs):
        """
        Inicializa os priors conjuntos.
        
        :param param_configs: Um dicionário onde as chaves são os nomes dos parâmetros
                              e os valores são tuplas (intervalo, distribuição).
        """
        self.param_configs = param_configs
        self.param_names = list(param_configs.keys())
    
    def prior_transform(self, cube):
        """
        Transforma os valores do cubo unitário em distribuições de prior.
        
        :param cube: Array de valores em [0, 1] do ultranest.
        :return: Array de parâmetros transformados.
        """
        params = np.zeros_like(cube)
        
        for i, param_name in enumerate(self.param_names):
            (low, high), dist = self.param_configs[param_name]
            
            if dist == 'uniform':
                params[i] = low + (high - low) * cube[i]
            elif dist == 'gaussian':
                mu, sigma = (low + high) / 2, (high - low) / 6
                params[i] = stats.norm.ppf(cube[i], loc=mu, scale=sigma)
            elif dist == 'log-normal':
                mu, sigma = np.log((low + high) / 2), (np.log(high) - np.log(low)) / 6
                params[i] = np.exp(stats.norm.ppf(cube[i], loc=mu, scale=sigma))
            else:
                raise ValueError(f"Distribuição não suportada: {dist}")
        
        return params




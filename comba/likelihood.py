import numpy as np
from scipy import stats
from joblib import Parallel, delayed
import pymp
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

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

class JointLikelihoodFunction:
    
    def __init__(self, model_funcs=None, shared_params=None):
        # Inicializa atributos específicos
        self.model_funcs = model_funcs or {}
        self.shared_params = shared_params

    def estimate_pdf_frb(self, dm_obs, dm_ext_samples, sigma):
        """
        Estima a log-PDF considerando o erro total associado ao DM_obs_ext_error,
        usando `scipy.stats.gaussian_kde`.
ss
        Args:
            dm_obs (float): Valor observado.
            dm_ext_samples (array): Amostras para ajustar o KDE.
            sigma (float): Erro associado ao DM_obs_ext.

        Returns:
            float: Log-PDF ajustado no ponto dm_obs.
        """
        # Ajusta a estimativa KDE com as amostras
        kde = gaussian_kde(dm_ext_samples)

        # Estima a densidade em `dm_obs`
        pdf = kde(dm_obs)  # Retorna a densidade no ponto dm_obs
        log_pdf = np.log(pdf[0])  # Obtem o log-PDF

        # Ajusta a log-PDF para incluir o erro associado
        log_pdf_adjusted = log_pdf - 0.5 * np.log(2 * np.pi * sigma**2) - (dm_obs**2) / (2 * sigma**2)

        return log_pdf_adjusted


    def log_likelihood(self, params, data_sets=None):
        """Calcula a log-verossimilhança combinada para vários datasets."""
        
        total_loglike = 0

        # Log-verossimilhança para outros datasets fornecidos
        if data_sets:
            for data_name, data_info in data_sets.items():
                if data_name not in self.model_funcs:
                    raise ValueError(f"Modelo para {data_name} não foi fornecido.")

                # Marginaliza parâmetros não compartilhados
                model_params = self.marginalize_non_shared_params(params) if data_name in ["SNe", "CC"] else params
                
                model_func = self.model_funcs[data_name]
                model_params = {k: v for k, v in model_params.items() if k in model_func.__code__.co_varnames}

                if data_name == "SNe":
                    # Dados para supernovas (SNe)
                    z_values, y_obs, cov_matrix_inv = data_info
                    y_model = model_func(z_values, **model_params)

                    delta = y_model - y_obs
                    chi2 = np.dot(delta.T, np.dot(cov_matrix_inv, delta))
                    total_loglike += -0.5 * chi2

                # elif data_name == "PDF":
                #     from equations import Cosmography
                #     dm_ext = Cosmography()
                #     z_values, y_obs, error = data_info
                #     y_model = model_func(z_values, **model_params)

                #     dm_ext_th = dm_ext.DM_ext_cmy_pdf(z=z_values, A=model_params['A'], beta=model_params['beta'], sigma_igm=error, dm_ext_obs=y_obs, dm_ext_th_values=y_model)
                    
                #     logpdf = np.sum(dm_ext_th)

                #     total_loglike += logpdf

                elif data_name == "PDF":
                    z_values, y_obs, errors = data_info
                    y_model = model_func(z_values, **model_params)
        
                    for dm_obs, dm_ext_samples, sigma in zip(y_obs, y_model, errors):
                        loglike = self.estimate_pdf_frb(dm_obs=dm_obs, dm_ext_samples=dm_ext_samples, sigma=sigma)

                        total_loglike += np.sum(loglike)

                else:
                    # Outros datasets (genéricos)
                    z_values, y_obs, errors = data_info
                    y_model = model_func(z_values, **model_params)
                    error_new = (
                        np.where(y_model > y_obs, errors[1], errors[0]) 
                        if isinstance(errors, tuple) else errors
                    )
                    loglike = - 0.5 * np.sum(((y_model - y_obs) / error_new) ** 2) 
                    total_loglike += loglike

        return total_loglike


    def marginalize_non_shared_params(self, params):
        """Marginaliza os parâmetros que não são compartilhados."""
        if self.shared_params is None:
            return params
        return {k: v for k, v in params.items() if k in self.shared_params}


class JointPriors:
    def __init__(self, param_configs):
        """
        Inicializa os priors conjuntos.
        
        :param param_configs: Um dicionário onde as chaves são os nomes dos parâmetros
                              e os valores são tuplas para a distribuição desejada:
                              - Prior uniforme: ((low, high), 'uniform')
                              - Prior gaussiana: ((mu, sigma), 'gaussian')
                              - Prior log-normal: ((low, high), 'log-normal')
        """
        self.param_configs = param_configs
        self.param_names = list(param_configs.keys())

    def convert_cosmo_params(cosmo_params):
        """
        Converte parâmetros do Cosmoprimo para o formato aceito por JointPriors.
        """
        converted = {}
        for param in cosmo_params:
            param_name = param.name
            limits = param.limits
            converted[param_name] = (tuple(limits), 'uniform')  # Formato esperado para prior uniforme
        return converted

    def prior_transform(self, cube):
        """
        Transforma os valores do cubo unitário em distribuições de prior.
        
        :param cube: Array de valores em [0, 1] do ultranest.
        :return: Array de parâmetros transformados.
        """
        params = np.zeros_like(cube)
        
        for i, param_name in enumerate(self.param_names):
            config = self.param_configs[param_name]
            
            # Desempacotando a configuração da distribuição
            if isinstance(config[0], tuple) and len(config[0]) == 2:
                if config[1] == 'uniform':
                    low, high = config[0]
                elif config[1] == 'gaussian':
                    mu, sigma = config[0]
                elif config[1] == 'log-normal':
                    low, high = config[0]
                dist = config[1]
            else:
                raise ValueError(f"Configuração inválida para o parâmetro {param_name}")

            # Aplicando a transformação de acordo com a distribuição
            if dist == 'uniform':
                params[i] = low + (high - low) * cube[i]
            elif dist == 'gaussian':
                params[i] = stats.norm.ppf(cube[i], loc=mu, scale=sigma)
            elif dist == 'log-normal':
                mu, sigma = np.log((low + high) / 2), (np.log(high) - np.log(low)) / 6
                params[i] = np.exp(stats.norm.ppf(cube[i], loc=mu, scale=sigma))
            else:
                raise ValueError(f"Distribuição não suportada: {dist}")
        
        return params






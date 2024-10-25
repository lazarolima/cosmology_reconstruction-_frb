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
        
        Inicializa a função de verossimilhança conjunta.
        
        :param model_funcs: Um dicionário onde as chaves são os nomes dos conjuntos de dados
                            e os valores são as funções do modelo correspondentes.
        
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
                chi2 = np.dot(delta.T, np.dot(cov_matrix, delta))
                
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
        
        return total_loglike"""

import numpy as np
from desilike.likelihoods.supernovae import (
    PantheonPlusSHOESSNLikelihood, 
    PantheonPlusSNLikelihood, 
    PantheonSNLikelihood
)

"""class SNe_desilike:
    def __init__(self, cosmo):
        self.cosmo = cosmo

    def likelihood_function(self, cosmo):

        from desilike import setup_logging

        setup_logging()  # set up logging

        likelihoods = {#'Pantheon': PantheonSNLikelihood(cosmo=cosmo),
                    #'Pantheon+': PantheonPlusSNLikelihood(cosmo=cosmo),
                    'Pantheon+ & SH0ES': PantheonPlusSHOESSNLikelihood(cosmo=cosmo)}

        loglike = likelihoods.get('Pantheon+ & SH0ES').all_params['PantheonPlusSHOESSN.loglikelihood']

        return loglike"""
        

"""class JointLikelihoodFunction:

    def __init__(self, model_funcs, shared_params=None):
        
        Inicializa a função de verossimilhança conjunta.

        :param model_funcs: Dicionário onde as chaves são nomes dos conjuntos de dados
                            e os valores são as funções do modelo correspondentes.
        :param shared_params: Lista opcional dos nomes dos parâmetros comuns entre modelos.
                              Se não fornecida, todos os parâmetros serão utilizados.
        
        self.model_funcs = model_funcs
        self.shared_params = shared_params

    def log_likelihood(self, params, data_sets):
        
        Calcula a verossimilhança total para um conjunto de dados fornecido.

        :param params: Dicionário com todos os parâmetros fornecidos.
        :param data_sets: Dicionário com conjuntos de dados e informações associadas.
        :return: Verossimilhança total.
        
        total_loglike = 0

        for data_name, data_info in data_sets.items():
            model_func = self.model_funcs[data_name]

            # Filtra os parâmetros relevantes para cada modelo
            model_params = {
                k: v for k, v in params.items() if k in model_func.__code__.co_varnames
            }

            if data_name == "SNe":  # Para supernovas
                z_values, y_obs, cov_matrix = data_info
                y_model = model_func(z_values, **model_params)

                delta = y_model - y_obs
                chi2 = np.dot(delta.T, np.dot(cov_matrix, delta))
                total_loglike += -0.5 * chi2

            else:  # Para FRBs ou outros conjuntos
                z_values, y_obs, errors = data_info
                y_model = model_func(z_values, **model_params)

                if isinstance(errors, tuple):  # Erros assimétricos
                    err_neg, err_pos = errors
                    error_new = np.where(y_model > y_obs, err_pos, err_neg)
                else:  # Erros simétricos
                    error_new = errors

                loglike = -0.5 * np.sum(((y_model - y_obs) / error_new) ** 2)
                total_loglike += loglike

        return total_loglike

    def marginalize_non_shared_params(self, params):
        
        Marginaliza os parâmetros não compartilhados (se aplicável).

        :param params: Dicionário com todos os parâmetros fornecidos.
        :return: Dicionário contendo apenas os parâmetros compartilhados (se especificados).
        
        # Se shared_params não foi fornecido, retorna todos os parâmetros
        if self.shared_params is None:
            return params

        # Retorna apenas os parâmetros compartilhados
        return {k: v for k, v in params.items() if k in self.shared_params}"""

"""class JointPriors:
    def __init__(self, param_configs):
        
        Inicializa os priors conjuntos.
        
        :param param_configs: Um dicionário onde as chaves são os nomes dos parâmetros
                              e os valores são tuplas (intervalo, distribuição).
        
        self.param_configs = param_configs
        self.param_names = list(param_configs.keys())
    
    def prior_transform(self, cube):
        
        Transforma os valores do cubo unitário em distribuições de prior.
        
        :param cube: Array de valores em [0, 1] do ultranest.
        :return: Array de parâmetros transformados.
        
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
        
        return params"""


import numpy as np
from desilike.likelihoods.supernovae import (
    PantheonPlusSHOESSNLikelihood, 
    PantheonPlusSNLikelihood, 
    PantheonSNLikelihood
)
from desilike import setup_logging

class JointLikelihoodFunction(PantheonPlusSHOESSNLikelihood):
    """
    Classe que combina a verossimilhança de supernovas Pantheon+ & SH0ES com outras fontes de dados.
    """

    def __init__(self, model_funcs=None, cosmo=None, shared_params=None, **kwargs):
        """
        Inicializa a classe e seus argumentos, aproveitando a inicialização da classe pai.

        Parameters:
        -----------
        model_funcs : dict, optional
            Dicionário com funções de modelos para cada dataset adicional.
        cosmo : Cosmology, optional
            Instância de cosmologia para uso na análise.
        shared_params : list, optional
            Lista de parâmetros compartilhados entre modelos.
        kwargs : dict
            Argumentos adicionais a serem passados para a classe pai.
        """
        # Guarda os argumentos específicos desta classe
        self.model_funcs = model_funcs if model_funcs else {}
        self.shared_params = shared_params
        self.cosmo = cosmo

        # Filtra os kwargs para remover argumentos específicos desta classe
        parent_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ['model_funcs', 'shared_params']}
        
        # Inicializa a classe pai apenas com os argumentos relevantes
        super().__init__(**parent_kwargs)

        setup_logging()

    def initialize(self, *args, **kwargs):
        """
        Sobrescreve o método initialize para garantir que apenas argumentos válidos
        sejam passados para a classe pai.
        """
        # Filtra os kwargs para remover argumentos específicos desta classe
        parent_kwargs = {k: v for k, v in kwargs.items() 
                        if k not in ['model_funcs', 'shared_params']}
        
        # Chama o initialize da classe pai com os argumentos filtrados
        super().initialize(*args, **parent_kwargs, cosmo=self.cosmo)
        #super().get()

    def log_likelihood(self, params, data_sets=None):
        """
        Calcula a log-verossimilhança combinada para todos os datasets.

        Parameters:
        -----------
        params : dict
            Dicionário com os parâmetros do modelo.
        data_sets : dict, optional
            Dicionário com os dados adicionais para análise.

        Returns:
        --------
        float
            Log-verossimilhança total combinada.
        """
        total_loglike = 0

        # Calcula a verossimilhança para "Pantheon+ & SH0ES", se aplicável
        if self.cosmo is not None:
            loglike = self.calculate()
            total_loglike += loglike

        # Verifica e processa outros datasets, se fornecidos
        if data_sets:
            for data_name, data_info in data_sets.items():
                if data_name not in self.model_funcs:
                    raise ValueError(f"Modelo para {data_name} não fornecido.")

                model_func = self.model_funcs[data_name]
                model_params = {k: v for k, v in params.items() 
                              if k in model_func.__code__.co_varnames}

                if data_name == "SNe":  # Dados de supernovas
                    z_values, y_obs, cov_matrix = data_info
                    y_model = model_func(z_values, **model_params)
                    delta = y_model - y_obs
                    chi2 = np.dot(delta.T, np.dot(cov_matrix, delta))
                    total_loglike += -0.5 * chi2

                else:  # Outros datasets (como FRBs)
                    z_values, y_obs, errors = data_info
                    y_model = model_func(z_values, **model_params)
                    error_new = (
                        np.where(y_model > y_obs, errors[1], errors[0]) 
                        if isinstance(errors, tuple) else errors
                    )
                    loglike = -0.5 * np.sum(((y_model - y_obs) / error_new) ** 2)
                    total_loglike += loglike

        return total_loglike

    def marginalize_non_shared_params(self, params):
        """
        Marginaliza os parâmetros que não são compartilhados entre modelos.

        Parameters:
        -----------
        params : dict
            Dicionário com todos os parâmetros fornecidos.

        Returns:
        --------
        dict
            Dicionário contendo apenas os parâmetros compartilhados.
        """
        if self.shared_params is None:
            return params
        return {k: v for k, v in params.items() if k in self.shared_params}


class JointPriors:
    def __init__(self, param_configs):
        """
        Inicializa os priors conjuntos.
        
        :param param_configs: Um dicionário onde as chaves são os nomes dos parâmetros
                              e os valores são tuplas (intervalo, distribuição) ou dicionários.
        """
        self.param_configs = param_configs
        self.param_names = list(param_configs.keys())

    def convert_cosmo_params(cosmo_params):
        """
        Converte parâmetros do Cosmoprimo para o formato aceito por JointPriors.
        """
        converted = {}
        for param in cosmo_params:
            param_name = param.name  # Nome do parâmetro, ex.: 'Omega_m'
            limits = param.limits  # Acesse os limites diretamente
            converted[param_name] = (tuple(limits), 'uniform')  # Formato esperado
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
            
            # Verifica se a configuração é um dicionário (Cosmoprimo) ou uma tupla
            if isinstance(config, dict):
                low, high = config['prior']['limits']
                dist = 'uniform'  # Assume uniforme se não especificado
            else:
                (low, high), dist = config
            
            # Aplica a transformação correspondente
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





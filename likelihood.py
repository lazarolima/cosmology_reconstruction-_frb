# Definindo as priors
parameters1 = ['$f_{IGM}$']

def prior_transform1(cube):

    params = cube.copy()
    params[0] = cube[0]
    
    return params

parameters2 = ['$f_{IGM}$', '$\\alpha$']

def prior_transform2(cube):

    params = cube.copy()
    params[0] = cube[0]
    params[1] = 5*cube[1]
    
    return params

# Definindo a função likelihood
def log_likelihood1(params):

    f_IGM = params
    y_model = H_p1(z_values, f_IGM=f_IGM)
    loglike = -0.5 * (( (y_model - H_obs) /  errors)**2 ).sum()

    return loglike

def log_likelihood2(params):

    f_IGM, alpha = params
    y_model = H_p2(z_values, f_IGM=f_IGM, alpha=alpha)
    loglike = -0.5 * (( (y_model - H_obs) /  errors)**2 ).sum()

    return loglike

def log_likelihood3(params):

    f_IGM, alpha = params
    y_model = H_p3(z_values, f_IGM=f_IGM, alpha=alpha)
    loglike = -0.5 * (( (y_model - H_obs) /  errors)**2 ).sum()

    return loglike
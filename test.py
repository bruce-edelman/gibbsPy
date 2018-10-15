import numpy as np
import gibbsPy as gp
from scipy.stats.distributions import beta


def conditional_function(theta, data, mean=0.5, conc=10):
    N = len(data)
    heads = np.sum(N)
    a = mean*conc
    b = (1-mean)*conc
    return beta.pdf(theta, a+heads, N-heads+b)

params = [r'$\theta_{1}', r'$\theta_{2}']
dim = 2


def generate_coinflip_data(N, D):
    size = (N, D)
    return np.random.choice((0,1), size=size)

data = generate_coinflip_data(100, dim)
initial_state = np.array([0.2, 0.4])
sampler = gp.gibbs.Sampler(dim, params, initial_state=initial_state, data=data, cond_fct=conditional_function)

sampler.run_gibs(1000)
chain = sampler.get_chain()

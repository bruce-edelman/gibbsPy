import numpy as np
import gibbsPy as gp
import scipy.stats as stats
import scipy.stats.distributions


def conditional_function(pos, idx, data, a=1, b=1, random=None):
    N = len(data[:,idx])
    heads = np.sum(data[:,idx])
    return scipy.stats.distributions.beta.rvs(a+heads, N-heads+b, random_state=random if random is not None else None)


params = [r'$\theta_{1}$', r'$\theta_{2}$', r'$\theta_{3}$', r'$\theta_{4}$']
dim = 4
nobs = 150


def generate_coinflip_data(N, D):
    size = (N, D)
    data = np.zeros(size)
    theta_trues = np.random.uniform(0.,1., size=D)
    for i in range(D):
        data[:,i] = np.random.choice((1,0), size=N, p=(theta_trues[i], 1-theta_trues[i]))
    return data, theta_trues

data, thetas = generate_coinflip_data(nobs, dim)

initial_state = np.random.uniform(0,1, size=dim)
sampler = gp.sampler.Sampler(dim, params, initial_state=initial_state, data=data, cond_fct=conditional_function)

sampler.run_gibs(10000, progress=True)
chain = sampler.get_chain()

gp.utils.plot_corner(chain, params, trues=thetas)
gp.utils.plot_trace(chain, params, trues=thetas)




import numpy as np
import gibbsPy as gp
import scipy.stats as stats
import scipy.stats.distributions
import corner
import matplotlib.pyplot as plt


def conditional_function(pos, idx, data, a=1, b=1, random=None):
    N = len(data[:,idx])
    heads = np.sum(data[:,idx])
    return scipy.stats.distributions.beta.rvs(a+heads, N-heads+b, random_state=random if random is not None else None)

params = [r'$\theta_{1}$', r'$\theta_{2}$']
dim = 2

def generate_coinflip_data(N, D):
    size = (N, D)
    data = np.zeros(size)
    data[:,0] = np.random.choice((0, 1), size=N, p=(0.7, 0.3))
    data[:,1] = np.random.choice((0, 1), size=N, p=(0.3, 0.7))
    return data

data = generate_coinflip_data(48, 2)

initial_state = np.array([0.2, 0.4])
sampler = gp.sampler.Sampler(dim, params, initial_state=initial_state, data=data, cond_fct=conditional_function)

sampler.run_gibs(10000, progress=True)
chain = sampler.get_chain()
corner.corner(chain, range=[(0.,1.), (0.,1.)], labels=params, show_titles=True,
              quantities=(0.05,0.95))

plt.show()

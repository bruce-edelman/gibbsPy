## gibbsPy

This is a python package to implement Gibbs sampling simply and efficiently. 

# Usage

All that is needed to use is a function that describes the conditional probability distribution of your problem. 
The code will use the same function for each of the variables unless you input a list of functions the same size as the dimension and it will use the matching conditional function for each parameter. 

  import gibbsPy
  from scipy.stats import multivariate_norm as mvn
  
  def cond_fct(theta, data, sigma_cond=1):
    resid = data - theta
    return  mvn(loc=resid, scale=sigma_cond)
    
  params = ['x1', 'x2']
  dim = 2
  # some data gathered:
  data = get_data()
  initial_state = [1., 1.]
  
  # set up the sampler
  sampler = gibbspy.sampler(dim, params, initial_state=initial_state, data=data, cond_fct=conf_fct)
  last_state = sampler.run(10000)
   # get the chain data out via get_chain() fct
   
  chain = sampler.get_chain()
  
  

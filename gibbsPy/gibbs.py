import numpy as np
from scipy.stats import multivariate_normal as mvn


class Sampler(object):

    def __init__(self, sampling_params, static_params=None, initial_state=None, random=None, backend=None,
                 resume=False, **kwargs):
        """

        :param model:
        :param sampling_params:
        :param static_params:
        :param initial_state:
        """

        self._previous_state = None
        self.backend = backend.Backend() if backend is None else backend

        if not self.backend.initialized:
            self._previous_state = None
            self.backend.reset(self.model.dim)
            state = np.random.get_state()
        else:
            if self.backend.shape != self.model.dim:
                raise ValueError("The shape of backend does not match the model dimension")

            state = self.backend.random_state
            if state is None:
                if random is None:
                    state = np.random.get_state()
                else:
                    state = random
            iteration = self.backend.iteration
            if resume:
                if iteration > 0:
                    self._previous_state = self.backend.get_last_sample()
                else:
                    raise ValueError("Must Run the chain before resuming:")

        self._random = np.random.mtrand.RandomState()
        self._random.set_state(state)


        self.hypers = static_params
        if initial_state is not None:
            if len(initial_state) != len(sampling_params):
                raise ValueError("Initial state must have values for each of the sampling parameters")
            if self._previous_state is None:
                self._previous_state = initial_state

        if self._previous_state is None and initial_state is None:
            raise ValueError("Must input initial state if not resuming a run:")
        self.dim = len(sampling_params)

        self.model = Model(self.dim, sampling_params, static_params=static_params, **kwargs)
        self.lnprob_fct = self.model.wrapped_fct


    def get_random_state(self):
        return self._random.get_state()

    def run_gibs(self, n):
        """

        :param n:
        :return:
        """
        initial_state = self._previous_state
        results = None
        acc = None
        for acc, results in self.sample(initial_state, n, store=True):
            pass

        self._previous_state = results
        return results, acc

    def sample(self, initial, n, store=False, thin=1):
        if store:
            self.backend.grow(n)
        for _ in range(n):
            for _ in range(thin):
                newState = self.propose(initial)
                acc, newState = self.decide(newState, initial)

            if store:
                self.backend.save_sample(acc, newState)
            yield acc, newState



class Model(object):
    """
    This is a class object to hold the structure of the model we setup for our gibbs sampling. This will hold the fcts
    for priors and likliehoods and will also have metthods that the sampler will make use of when performing Gibbs
    Sampling
    """
    def __init__(self, D, params, static_params=None, likliehoods=None, priors=None, random=None, **kwargs):
        """
        This is the initialization of the Model class to be used in our Gibbs Sampler

        :param likliehoods: (OPTIONAL) This is either a single function to be used individualy for each sampling
        parameter or it is a list of functions that is the length of sampling params in the order to match with which
        parameter to use in State.params. If none is passed will use a Multivaraite Normal Likliehood Function

        :param priors: (OPTIONAL) This is the same as likliehoods but for the prior for each or one to be used for each
        parameter. If nothing is passed The sampler will use an uninformed uniform prior

        """

        self.dim = D

        if likliehoods is None:
            self.liklie = mvn
        elif len(likliehoods) != self.dim or len(likliehoods) > 1:
            raise ValueError("Likliehoods argument Must be same size as dimension, a sinlge function, or None")
        if priors is None:
            self.prior = 1
        elif len(priors) != self.dim or len(priors) > 1:
            raise ValueError("Likliehoods argument Must be same size as dimension, a sinlge function, or None")

        if params is None:
            self.params = []*self.dim
            for i in range(self.dim):
                self.params[i] = 'x%s' % i
        elif len(params) != self.dim:
            raise ValueError('List of parameter names must be the same length as the dimension of the probelem or None:')
        else:
            self.params = params

        self.liklie = likliehoods
        self.prior = priors

        self.wrapped_fct = _FnWrap(self.lnprob, static_params, **kwargs)


    def lnprob(self,x, *args, **kwargs):
        """

        :param x:
        :param args:
        :param kwargs:
        :return:
        """
        pass




class State(object):
    """
    This is a class to hold the current state and handle state updating for the Gibbs Sampler Object
    Its main purposes is to hold the params list and whatever point in parameter space the markov chain is currently at
    Will also contain methods used in the sampler object
    """
    def __init__(self, sampling_params, logprob=None, random=None):
        """
        This is the Initalization of our State class to be used in the sampler

        :param sampling_params: This is a dictionary that stores the string name of each sampling parameter and the
        current value of each of the parameters
        :param static_params: This is a dictionary that stores the string name of each static parameter and the constant
        value it takes
        """

        self.params = sampling_params
        self.pos = self.params[[val for val in self.params.keys()]]
        if random is not None:
            self.random_state = random
        else:
            self.random_state = np.random.get_state()
        self.logprob = logprob


    def get_param_val(self, name):
        if name is not in self.params.keys:
            raise ValueError("Param: %s is not in sampling_params dictionary:" % name)
        else:
            return self.params["%s" % name]

    def get_param_names(self):
        for i in self.params.keys:
            yield i

    def __repr__(self):
        return "State({0}, log_prob={1}, random_state={2})".format(self.params, self.logprob, self.random_state)

    def __iter__(self):
        return iter((self.params, self.logprob, self.random_state))


class _FnWrap(object):
    """
    This is a wrapper class for ease of calling the ln_prob_fct (i.e. the likliehood fcts that the model holds)
    """
    def __init__(self,func, *args, **kwargs):
        """

        :param func:
        :param args:
        :param kwargs:
        """
        self.args = [] if args is None else args
        self.kwargs = {} if kwargs is None else kwargs
        self.function = func

    def __call__(self,x):
        """

        :param x:
        :return:
        """
        try:
            return self.function(x, *self.args, *self.kwargs)
        except:
            import traceback
            print("gibbsPy: Exception while calling your likelihood function:")
            print("  params:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise



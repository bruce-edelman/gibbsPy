import numpy as np
import types


class Sampler(object):

    def __init__(self, D, sampling_params, static_params=None, initial_state=None, random=None, back=None,
                 resume=False, data=None, **kwargs):
        """

        :param model:
        :param sampling_params:
        :param static_params:
        :param initial_state:
        """
        self.dim = D

        if sampling_params is None:
            self.params = [] * self.dim
            for i in range(self.dim):
                self.params[i] = 'x%s' % i
        elif len(sampling_params) != self.dim:
            raise ValueError(
                'List of parameter names must be the same length as the dimension of the probelem or None:')
        else:
            self.params = sampling_params
        self._previous_state = None
        self.backend = Backend() if back is None else back
        if data is not None:
            self.data = data
        else:
            self.data = None
        if not self.backend.initialized:
            self._previous_state = None
            self.backend.reset(self.dim)
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
            if len(initial_state) != self.dim:
                raise ValueError("Initial state must have values for each of the sampling parameters")
            if self._previous_state is None:
                self._previous_state = State(initial_state, data=self.data,random=self.get_random_state())

        if self._previous_state is None and initial_state is None:
            raise ValueError("Must input initial state if not resuming a run:")

        self.model = Model(self.dim, self.params, static_params=None if static_params is None else static_params,
                           data=self.data,random=self.get_random_state(), **kwargs)
        self.conditional_fct = self.model.wrapped_fct

    def has_data(self):
        return True if self.data is not None else False

    def get_random_state(self):
        return self._random.get_state()

    def run_gibs(self, n):
        """

        :param n:
        :return:
        """
        if self._previous_state is not None:
            initial_state = self._previous_state
        else:
            raise ValueError("The previous sate of the sampler must be set when "
                             "intializing sampler or the backend must have been ran before with resume=True:")
        results = None
        for results in self.sample(initial_state, n, store=True):
            pass

        self._previous_state = results
        return results

    def sample(self, initial, n, store=False, thin=1):
        newState = initial
        if store:
            self.backend.grow(n)
        for _ in range(n):
            for _ in range(thin):
                for i in range(self.dim):
                    newState.pos[i] = self.conditional_fct(initial.pos[np.arange(int(len(initial.pos)))!=i])

            if store:
                self.backend.save_sample(newState)
            yield newState

    def get_chain(self, **kwargs):
        if self.backend.initialized:
            return self.backend.get_chain(**kwargs)
        else:
            raise ValueError("Must have backend initialized and chain ran before retrieving chain")


class Model(object):
    """
    This is a class object to hold the structure of the model we setup for our gibbs sampling. This will hold the fcts
    for priors and likliehoods and will also have metthods that the sampler will make use of when performing Gibbs
    Sampling
    """
    def __init__(self, D, params, cond_fct, static_params=None,data=None, random=None, **kwargs):
        """
        This is the initialization of the Model class to be used in our Gibbs Sampler

        :param likliehoods: (OPTIONAL) This is either a single function to be used individualy for each sampling
        parameter or it is a list of functions that is the length of sampling params in the order to match with which
        parameter to use in State.params. If none is passed will use a Multivaraite Normal Likliehood Function

        :param priors: (OPTIONAL) This is the same as likliehoods but for the prior for each or one to be used for each
        parameter. If nothing is passed The sampler will use an uninformed uniform prior

        """

        self.dim = D
        if not isinstance(cond_fct, types.FunctionType):
            if len(cond_fct) != self.dim:
                raise ValueError("Cond_fct must be a single fct for each parameter or a list of D fcts where "
                             "D is the dimension of the model:")

        if params is None:
            self.params = []*self.dim
            for i in range(self.dim):
                self.params[i] = 'x%s' % i
        elif len(params) != self.dim:
            raise ValueError('List of parameter names must be the same length as the dimension of the probelem or None:')
        else:
            self.params = params

        if data is not None:
            self.data = data
        else:
            self.data = None
        if isinstance(cond_fct, types.FunctionType):
            self.wrapped_fct = _FnWrap(cond_fct, static_params, data=self.data, random=random, **kwargs)
        else:
            self.wrapped_fct = self.multi_wrap(cond_fct, static_params, data=self.data, random=random, **kwargs)

    def multi_wrap(self, fcts, hypers, data=None,random=None, **kwargs):
        return _FnWrap(fcts, hypers, data=data, random=random, **kwargs)

    def has_data(self):
        return True if self.data is not None else False


class State(object):
    """
    This is a class to hold the current state and handle state updating for the Gibbs Sampler Object
    Its main purposes is to hold the params list and whatever point in parameter space the markov chain is currently at
    Will also contain methods used in the sampler object
    """
    def __init__(self, pos, data=None, random=None):
        """
        This is the Initalization of our State class to be used in the sampler

        :param sampling_params: This is a dictionary that stores the string name of each sampling parameter and the
        current value of each of the parameters
        :param static_params: This is a dictionary that stores the string name of each static parameter and the constant
        value it takes
        """

        self.pos = pos
        if random is not None:
            self.random_state = random
        else:
            self.random_state = np.random.get_state()
        self.data = data

    def has_data(self):
        return True if self.data is not None else False

    def __repr__(self):
        return "State(pos={0}, data={1}, random_state={2})".format(self.pos, self.data, self.random_state)

    def __iter__(self):
        return iter((self.pos, self.data, self.random_state))


class _FnWrap(object):
    """
    This is a wrapper class for ease of calling the ln_prob_fct (i.e. the likliehood fcts that the model holds)
    """
    def __init__(self,func, *args, data = None, random = None, **kwargs):
        """

        :param func:
        :param args:
        :param kwargs:
        """
        self.args = []
        for arg in args:
            if arg is not None:
                self.args.append(arg)
        print(self.args)
        self.kwargs = {} if kwargs is None else kwargs
        self.function = func
        self.data = data
        self.random = random

    def __call__(self,x):
        """

        :param x:
        :return:
        """
        try:
            print(self.args, self.kwargs)
            return self.function(x, self.data,random=self.random, *self.args, *self.kwargs)
        except:
            import traceback
            print("gibbsPy: Exception while calling your likelihood function:")
            print("  pos[i]:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise


class Backend(object):
    def __init__(self, random=None):
        self.initialized = False
        if random is not None:
            self.random_state = random

    def reset(self, ndim):
        self.dim = ndim
        self.iteration = 0
        self.chain = np.empty((0, self.dim))
        self.random_state = None
        self.initialized = True

    def grow(self, n):
        i = n - (len(self.chain) - self.iteration)
        a = np.empty((i, self.dim))
        self.chain = np.concatenate((self.chain, a), axis=0)

    def save_sample(self, state):
        self._check_state(state)
        self.chain[self.iteration, :] = state.pos
        self.random_state = state.random
        self.iteration += 1

    def _check_state(self, state):
        if state.pos.shape != (self.dim):
            raise ValueError("Invalid State Position dimension; expected {0}".format((self.dim)))

    def get_attribute(self, name, discard=0, thin=1):
        if self.iteration <= 0:
            raise AttributeError("Must run sampler and store values to retrieve attribute from the backend:")

        return getattr(self, name)[discard+thin-1:self.iteration:thin]

    def get_chain(self, **kwargs):
        return self.get_attribute("chain", **kwargs)

    def get_last_sample(self):
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError("Must run sampler and store values to retrieve last state from the backend:")

        return State(self.get_chain(discard=self.iteration-1)[0], random=self.random_state)



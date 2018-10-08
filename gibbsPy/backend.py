from . import gibbs
import numpy as np


class Backend(object):
    def __init__(self, random=None):
        self.initalized = False
        if random is not None:
            self.random_state = random

    def reset(self, ndim):
        self.dim = ndim
        self.iteration = 0
        self. acc = 0
        self.chain = np.empty((0, self.dim))
        self.logprobs = np.empty((0))
        self.random_state = None
        self.initalized = True

    def grow(self, n):
        i = n - (len(self.chain) - self.iteration)
        a = np.empty((i, self.dim))
        self.chain = np.concatenate((self.chain, a), axis=0)
        b = np.empty((i))
        self.logprobs = np.concatenate((self.logprobs, b), axis=0)

    def save_sample(self, state, acc):
        self._check_state(state)
        self.chain[self.iteration, :] = state.pos
        self.logprobs[self.iteration] = state.logprob
        self.acc += acc
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

    def get_logprobs(self, **kwargs):
        return self.get_attribute("logprobs", **kwargs)

    def get_last_sample(self):
        if (not self.initalized) or self.iteration <= 0:
            raise AttributeError("Must run sampler and store values to retrieve last state from the backend:")

        return gibbs.State(self.get_chain(discard=self.iteration-1)[0],
                           logprob=self.get_logprobs(discard=self.iteration-1), random=self.random_state
                     )


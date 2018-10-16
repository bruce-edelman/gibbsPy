# Copyright (C) 2018  Bruce Edelman
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import numpy as np
from . import backend
from . import model

"""
This File sets up the Sampler class that performs the gibbs sampling in gibbsPy
"""


class Sampler(object):
    """
    This is the Sampler object that does the gibbs sampling
    """
    def __init__(self, D, sampling_params=None, static_params=None, initial_state=None, random=None, back=None,
                 resume=False, data=None, **kwargs):
        """
        The intialization function called when we set up an instance of our sampler object

        :param D: The dimension of the problem we want to sample

        :param sampling_params: (Optional) This is a list of strings with the name of each parameter we are sampling in.
        This defaults to being ['x1', ... 'xD'] with D the dimension of the problem

        :param static_params: (Optional) This is an optional dictionary that will serve as the static paramerters in our
        problem. These are parameters that need to be passed to our conditional function later:
        the keys are the string names for each static param and the value is the value that we hold that parameter at:

        :param initial_state: (optional) This is optional but if it is not set than an intialized instance of the
        backend.Backend() object must be passed into the backend parameter:
        If this is passed it is a numpy array of length = D that holds the starting paramter value for each sampling
        param in order of sampling_params list:
        If initial_state is given and also resume=True along with an initialized backend we take the backend previous
        state over the passed initial state:

        :param random: (optional) If we pass this it must be an instance of a numpy random state. If it is set and also
        backend is set from a previous run the backends random state takes priority:

        :param back: (optional) If we want to resume a previous run we must have resume=True, and back set to a
        backend.Backend() object instance that was from the previous run. This backend must be initialized (meaning it
        was used in a previous run and has _previous_state attribute stored that is a state.State() object instance

        :param resume: (optional) this is a Bool, defaults to False. If set to true it will try and add on to the chain
        in the backend object, if false we will either create a new backend object or reset the passed in backend and
        start a new chain

        :param data: (optional) This optional parameter that stores and holds the data that was gathered that is involved
        in our conditional probability distribution. This must be a 2-d numpy array with one axis the data points and
        the second axis must have length = D so that we have at least one data point for each paramter. (#TODO currently
        requires N be the same for each paramter. fix this later?)

        :param kwargs: (optional) These can be any keyword arguments that we may need to pass to our conditional function
        This depends on the user-generated conditional function that we want to sample from in our gibbs sampling
        """

        # store the dimension
        self.dim = D

        # Store parameter names and make sure same length as dimension
        if sampling_params is None:
            self.params = [] * self.dim
            for i in range(self.dim):
                self.params[i] = 'x%s' % i
        elif len(sampling_params) != self.dim:
            raise ValueError(
                'List of parameter names must be the same length as the dimension of the probelem or None:')
        else:
            self.params = sampling_params

        # Handle resuming from backend sent in if there: Make sure to get the current random state from backend if
        # possible to keep us starting from that pos:
        state = None
        self._previous_state = None
        self.backend = backend.Backend() if back is None else back
        if data is not None:
            self.data = data
        else:
            self.data = None
        if not self.backend.initialized and not resume:
            self._previous_state = None
            self.backend.reset(self.dim)
            state = np.random.get_state()
        elif self.backend.initialized and resume:
            if self.backend.shape != self.model.dim:
                raise ValueError("The shape of backend does not match the model dimension")

            state = self.backend.random_state
            iteration = self.backend.iteration
            if iteration > 0:
                self._previous_state = self.backend.get_last_sample()
            else:
                raise ValueError("Must Run the chain before resuming:")
        elif self.backend.initialized and not resume:
            self._previous_state = self.backend.get_last_sample()
            self.backend.reset(self.dim)
            state = self.backend.random_state

        if state is None:
            if random is None:
                state = np.random.get_state()
            else:
                state = random
        self._random = np.random.RandomState()
        self._random.set_state(state)

        # initialize our hyper parameters(static_params)
        self.hypers = static_params

        # Handle exceptions with initial state:
        if initial_state is not None:
            if len(initial_state) != self.dim:
                raise ValueError("Initial state must have values for each of the sampling parameters")
            if self._previous_state is None:
                self._previous_state = state.State(initial_state, data=self.data,random=self._random)

        if self._previous_state is None and initial_state is None:
            raise ValueError("Must input initial state if not resuming a run:")

        self.model = model.Model(self.dim, self.params, static_params=None if static_params is None else static_params,
                           data=self.data,random=self._random, **kwargs)
        self.conditional_fct = self.model.wrapped_fct

    def has_data(self):
        return True if self.data is not None else False

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
                    newState.pos[i] = self.conditional_fct(initial.pos, i)

            if store:
                self.backend.save_sample(newState)
            yield newState

    def get_chain(self, **kwargs):
        if self.backend.initialized:
            return self.backend.get_chain(**kwargs)
        else:
            raise ValueError("Must have backend initialized and chain ran before retrieving chain")
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
from . import state
from .pbar import *

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
        state over the passed initial ranstate:

        :param random: (optional) If we pass this it must be an instance of a numpy random ranstate. If it is set and also
        backend is set from a previous run the backends random ranstate takes priority:

        :param back: (optional) If we want to resume a previous run we must have resume=True, and back set to a
        backend.Backend() object instance that was from the previous run. This backend must be initialized (meaning it
        was used in a previous run and has _previous_state attribute stored that is a ranstate.State() object instance

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
        ranstate = None
        self._previous_state = None
        self.backend = backend.Backend() if back is None else back
        if data is not None:
            self.data = data
        else:
            self.data = None
        if not self.backend.initialized and not resume:
            self._previous_state = None
            self.backend.reset(self.dim)
            ranstate = np.random.get_state()
        elif self.backend.initialized and resume:
            if self.backend.shape != self.model.dim:
                raise ValueError("The shape of backend does not match the model dimension")

            ranstate = self.backend.random_state
            iteration = self.backend.iteration
            if iteration > 0:
                self._previous_state = self.backend.get_last_sample()
            else:
                raise ValueError("Must Run the chain before resuming:")
        elif self.backend.initialized and not resume:
            self._previous_state = self.backend.get_last_sample()
            self.backend.reset(self.dim)
            ranstate = self.backend.random_state

        # Setup the Random number generator with new state random or the passed in state from the previous backend
        if ranstate is None:
            if random is None:
                ranstate = np.random.get_state()
            else:
                ranstate = random
        self._random = np.random.RandomState()
        self._random.set_state(ranstate)

        # initialize our hyper parameters(static_params)
        self.hypers = static_params

        # Handle exceptions with initial ranstate:
        if initial_state is not None:
            if len(initial_state) != self.dim:
                raise ValueError("Initial state must have values for each of the sampling parameters")
            if self._previous_state is None:
                self._previous_state = state.State(initial_state, random=self._random)

        # make sure we have either a previous state or initialstate
        if self._previous_state is None and initial_state is None:
            raise ValueError("Must input initial ranstate if not resuming a run:")

        # Setup the model to be used:
        self.model = model.Model(self.dim, params=self.params, static_params=None if static_params is None else static_params,
                                 data=self.data,random=self._random, **kwargs)
        # retreive the wrapped conditional function from the model (uses our handy function wrapper so that we can
        # use kwargs or args when calling the fct without having to call them each time:
        self.conditional_fct = self.model.wrapped_fct

    def has_data(self):
        """
        Function returns true/false whether or not we setup the sampler with some observed data or not.

        :return: True if self.data is not None, otherwise return False
        """
        return True if self.data is not None else False

    def run_gibs(self, n, store=True, **kwargs):
        """
        This is the main function that will run the gibbs sampler

        :param n: This is the number of steps to run the sampler for

        :param store: (optional) This a bool value that determines if we save our samples in our backend object or not
        defaults to True

        :param kwargs: (optional) these keyword args are passed into sample (currently include thin=1, progress=False)

        :return: this returns the final state of the chain (must use the backend object to retreieve all of the samples
        """

        # Setup the initial state
        if self._previous_state is not None:
            initial_state = self._previous_state
        else:
            raise ValueError("The previous sate of the sampler must be set when "
                             "intializing sampler or the backend must have been ran before with resume=True:")

        results = None
        # run the generator to generate each successive samples
        for results in self.sample(initial_state, n, store=store, **kwargs):
            pass
        # store the last state as the previous state for the sample/backend
        self._previous_state = results
        return results

    def sample(self, initial, n, store=False, thin=1, progress=False):
        """
        This is the Generator for sampling the next values from the conditional distribution we are trying to sample
        from

        :param initial: THis is the initial state we are in must be an instance of State object

        :param n: Number of steps to evovle our chain

        :param store: (optional) bool value that sets whether we store the values in the backend or not defaults to False

        :param thin: (optional) This value is how many samples we want to thin the chain by. defaults to no thinning
        (thin = 1)

        :param progress: (optional) Boolean value set from the kwargs passed into run_gibbs or burnin_gibbs that decides
        whether or not to show a progress bar during sampling. defaults to False

        :return: This is a generator so it yields the next sample at each iteration: (samples are object instances of
        the State object)
        """
        # Initialize the newState as the old
        newState = initial

        # if we store the values grow the backend for faster saving
        if store:
            self.backend.grow(n)
            # Check if we set thinning up or not
        if thin is not None:
            thin = int(thin)
            # error check to make sure thin is not negative or 0
            if thin <= 0:
                raise ValueError("Thin must be strictly positive:")
            intermediate_step = thin
        else:
            # if no thin set int_step=1
            intermediate_step = 1
        # set the total iterations for pbar
        total = n * intermediate_step
        # set up our progress bar
        with progress_bar(progress, total) as prog_bar:
            # Loop through our desired range
            for _ in range(n):
                # loop through the thinning procedure
                for _ in range(thin):
                    # Loop through each parameter dimension since Gibbs Sampling algorithm has us directly sample each
                    # parameter from the conditional probability functions:
                    for i in range(self.dim):
                        # find the new value for paramter i feom the conditinoal distribution
                        try:
                            idx = self.conditional_fct[i].idx
                            newState.pos[idx] = self.conditional_fct[i](initial.pos, i)
                        except TypeError:
                            newState.pos[i] = self.conditional_fct(initial.pos, i)

                    prog_bar.update(1)
                # If we store we want to save each sample in the backend (after thinning since n is final amount of
                # samples we want)
                if store:
                    self.backend.save_sample(newState)
                # generate the state
                yield newState

    def get_chain(self, **kwargs):
        """
        This is a function that connects the sampler with the backend so we can get the chain out:

        :param kwargs: These kwargs are optional values passed to the backend get_chain_fct. These are listed in the
        backend.get_chain() method documentation

        :return: This returns the numpy array of the chain position in parameter space at each iteration
        """

        # make sure the backend was intialized before retrieving the chain data
        if self.backend.initialized:
            return self.backend.get_chain(**kwargs)
        else:
            raise ValueError("Must have backend initialized and chain ran before retrieving chain")
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
from . import state

'''
This file sets up the Backend object that handles to data structure and storage for our gibbs sampler
'''


class Backend(object):
    """
    This is Backend object that will handle storing the data for our markov chains
    """

    def __init__(self, random=None):
        """
        Function to intilize the Backend

        :param random: (optional) if we want to initialize it with an already specified random_state that is istance of
        a numpy.random.Random() object class
        """
        # set the variable to knows if the backend has been intiitalized yet:
        self.initialized = False

        # if we pass in a state then we use it as our random_state if it is a correct numpy.random.Random()
        # class instance
        if random is not None and isinstance(np.random.RandomState, random):
            self.random_state = random

    def reset(self, ndim):
        """
        This function resets and intiializes the backend whether it had been run before or not

        :param ndim: This is the dimension of the problem for the backend storage and is required

        :return: This function does not return anything
        """

        self.dim = ndim
        self.iteration = 0
        self.chain = np.empty((0, self.dim))
        self.random_state = None
        self.initialized = True

    def grow(self, n):
        """
        This function grows the size of the backend data structures to be prepaared to store more data in them

        :param n: this is the size of how much space we want to add to the data arrays:

        :return: This function does not return anything:
        """
        # take current length plus how much we want to grow by
        i = n - (len(self.chain) - self.iteration)

        # make an empyty array of correct size
        a = np.empty((i, self.dim))

        # add in stuff we already have:
        self.chain = np.concatenate((self.chain, a), axis=0)

    def save_sample(self, state):
        """
        This function saves a single new state and adds it into self,chiain at the next iteration value:

        :param state: This is the state we want to save. It must be an instance of the State object

        :return: This function does not return anything:
        """
        # Check to make sure the passed in state is an instance of the State object and is of the correct shape:
        self._check_state(state)

        # add in the position of that state to our chain
        self.chain[self.iteration, :] = state.pos

        # update our random state with the last one used in the saved state object
        self.random_state = state.random_state

        # update our iteration variable that will store the current itration value of our backend chain
        self.iteration += 1

    def _check_state(self, state):
        """
        This function checks the state and makes sure it has correct shape and object atrtributes:

        :param state: The state that we want to check. Should be an instance of State Object with state.pos.shape == D

        :return: This function does not return anything but will raise a ValueError if the state does not check out:
        """

        # get the position shape and make sure it == D
        if state.pos.shape != (self.dim,):
            raise ValueError("Invalid State Position dimension; expected {}".format(self.dim))

    def get_attribute(self, name, discard=0, thin=1):
        """
        This function gets any attribute we want out of the backend that is stored at any iteration.
        Currently this is just the chain positions but will add more stored attributes in the future that this function
        will easily retrieve from our backend.

        :param name: a string that is the name of the desired attribute we want to retrieve

        :param discard: (optional) this is an optional kw arg that is the number of samples to ignore at the beginning
        of the stored chain values, defaults to 0

        :param thin: (optional) this is an optional kw arg that is the factor by which we want to thin the chain by.
        (warning, if we thin the chain upon output and during sampling just know it might be thinned twice)

        :return:  This returns the output array of the desired attribute
        """
        # make sure the backend has some values stored already
        if self.iteration <= 0:
            raise AttributeError("Must run sampler and store values to retrieve attribute from the backend:")

        # retur the correct array with correct slicing
        return getattr(self, name)[discard+thin-1:self.iteration:thin]

    def get_chain(self, **kwargs):
        """
        This function uses the get_attribute fct to retreive the chain.

        :param kwargs: These are the optional kwargs to be passed to get_attribute() method. THese are thin, and discard
        which default to 1 and 0 respectively

        :return:  returns the output array of the stored chain
        """
        return self.get_attribute("chain", **kwargs)

    def get_last_sample(self):
        """
        This function retrieves the last sample of whatever is stored and returns it as a State object instance to be
        used for resuming in the sampler

        :return: returns a State object instance with position = the last sample stored in self.chain and the random
        state whatever is stored in the backend if any
        """

        # make sure the backend has been ran before we try and get a sample:
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError("Must run sampler and store values to retrieve last state from the backend:")

        # return the State object
        return state.State(self.get_chain(discard=self.iteration-1)[0], random=self.random_state)





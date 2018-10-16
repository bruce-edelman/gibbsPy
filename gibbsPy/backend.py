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
        self.random_state = state.random_state
        self.iteration += 1

    def _check_state(self, state):
        if state.pos.shape != (self.dim,):
            raise ValueError("Invalid State Position dimension; expected {}".format(self.dim))

    def get_attribute(self, name, discard=0, thin=1):
        if self.iteration <= 0:
            raise AttributeError("Must run sampler and store values to retrieve attribute from the backend:")

        return getattr(self, name)[discard+thin-1:self.iteration:thin]

    def get_chain(self, **kwargs):
        return self.get_attribute("chain", **kwargs)

    def get_last_sample(self):
        if (not self.initialized) or self.iteration <= 0:
            raise AttributeError("Must run sampler and store values to retrieve last state from the backend:")

        return state.State(self.get_chain(discard=self.iteration-1)[0], random=self.random_state)





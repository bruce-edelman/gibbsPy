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


class State(object):
    """
    This is a class to hold the current state and handle state updating for the Gibbs Sampler Object
    Its main purposes is to hold the params list and whatever point in parameter space the markov chain is currently at
    Will also contain methods used in the sampler object
    """
    def __init__(self, pos, random=None):
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

    def __repr__(self):
        return "State(pos={0}, random_state={1})".format(self.pos, self.random_state)

    def __iter__(self):
        return iter((self.pos, self.random_state))
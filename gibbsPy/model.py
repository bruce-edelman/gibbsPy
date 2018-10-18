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


from . import utils
import types

"""
This file sets up the Model to be used in our GibbsSampling This is the object that holds most of the details specfic
to a given gibbs sampling problem
"""

class Model(object):
    """
    This is a class object to hold the structure of the model we setup for our gibbs sampling.
    """
    def __init__(self, D, cond_fct=None, params=None, static_params=None,data=None, random=None, **kwargs):
        """
        This is the initialization of the Model class to be used in our Gibbs Sampler

        :param D: The dimension of our model

        :param cond_fct: This is a set up function by the user that returns the sampling of the conditional pdf Gibbs
        sampling samples from

        :param params: List containing strings that are the names of each parameter we are using. Defaults to None then
        the model creates its own of x1 ... xD

        :param static_params: (optional) These are the optioanl static_parameters that our model may useL

        :param data: This is a data set that with an array of data for each parameter that we wish to infer from

        :param random: (optional) THis is a np.random.RandomState istance for resuming

        :param kwargs: (optional) These are optioanal kwargs that may need to be passed to the cond_fct
        """

        self.dim = D

        # check to make sure cond_fct is a type Function and has the correct shape if not:
        if not isinstance(cond_fct, types.FunctionType):
            if len(cond_fct) != self.dim:
                raise ValueError("Cond_fct must be a single fct for each parameter or a list of D fcts where "
                             "D is the dimension of the model:")

        # Handle the defaulting for the params
        if params is None:
            self.params = []*self.dim
            for i in range(self.dim):
                self.params[i] = 'x%s' % i
        elif len(params) != self.dim:
            raise ValueError('List of parameter names must be the same length as the dimension of the probelem or None:')
        else:
            self.params = params

        # store the data
        if data is not None:
            self.data = data
        else:
            self.data = None

        # If we pass a single function to cond_fct, wrap it up with _FnWrapper
        if isinstance(cond_fct, types.FunctionType):
            self.wrapped_fct = utils._FnWrap(cond_fct, static_params, data=self.data, random=random, **kwargs)
        # Else we use (WIP) mulit wrap fct to wrap each cond_fct with _FnWrapper
        else:
            self.wrapped_fct = self.multi_wrap(cond_fct, static_params, data=self.data, random=random, **kwargs)

    def multi_wrap(self, fcts, hypers, data=None,random=None, **kwargs):
        """
        #TODO finish this function (right now it does nothing for us)
        :param fcts:
        :param hypers:
        :param data:
        :param random:
        :param kwargs:
        :return:
        """
        return utils._FnWrap(fcts, hypers, data=data, random=random, **kwargs)

    def has_data(self):
        """
        Simple boolean fct that returns True if the model is storing data and false if not
        :return: True/False if model has data or not
        """
        return True if self.data is not None else False

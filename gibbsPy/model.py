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
            self.wrapped_fct = utils._FnWrap(cond_fct, static_params, data=self.data, random=random, **kwargs)
        else:
            self.wrapped_fct = self.multi_wrap(cond_fct, static_params, data=self.data, random=random, **kwargs)

    def multi_wrap(self, fcts, hypers, data=None,random=None, **kwargs):
        return utils._FnWrap(fcts, hypers, data=data, random=random, **kwargs)

    def has_data(self):
        return True if self.data is not None else False

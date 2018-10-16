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


class _FnWrap(object):
    """
    This is a wrapper class for ease of calling the ln_prob_fct (i.e. the likliehood fcts that the model holds)
    """
    def __init__(self, func, *args, data = None, random = None, **kwargs):
        """

        :param func:
        :param args:
        :param kwargs:
        """
        self.args = []
        for arg in args:
            if arg is not None:
                self.args.append(arg)
        self.kwargs = {} if kwargs is None else kwargs
        self.function = func
        self.data = data
        self.random = random if random is not None else None

    def __call__(self, x, idx):
        """

        :param x:
        :return:
        """
        try:
            return self.function(x, idx, self.data, random=self.random, *self.args, *self.kwargs)
        except:
            import traceback
            print("gibbsPy: Exception while calling your likelihood function:")
            print("  pos[i]:", x)
            print("  args:", self.args)
            print("  kwargs:", self.kwargs)
            print("  exception:")
            traceback.print_exc()
            raise

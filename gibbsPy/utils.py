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

"""
This file sets up useful utility functions and/or classes to be used in other files of gibbsPy
"""
import numpy as np
import corner
import matplotlib.pyplot as plt

class _FnWrap(object):
    """
    This is a wrapper class for ease of calling the conditional function (i.e. the cond_fct that the model holds)
    """
    def __init__(self, func, *args, data = None, random = None, idx=None, **kwargs):
        """
        The intialization of our function wrapper class

        :param func: the function we want to wrap. Must be of type function

        :param args: (optional) args that are passed into the function
        :param data: data that is passed to the function
        :param random: (optional) random number state from numpy.random.RandomState()
        :param idx: (optional) If we wrap multiple functions for the conditional then this object will store the index
        of which parameter it corresponds to:
        :param kwargs: (optional) keyword arguments that the function may need to use
        """

        if idx is not None:
            self.idx = idx
        else:
            self.idx = None
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
        The call for our function after it is wrapped

        :param x: The position in parameter space we want to evaluate the function at

        :param idx: The index of the parameter we wish to evaluate at

        :return:retuns the output of the wrapped function
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

def compute_acl(chain):
    pass

def compute_act(chain):
    pass

def plot_corner(chain, labels, trues=None, file=None):
    """

    :param chain:
    :param labels:
    :param trues:
    :param file:
    :return:
    """
    fig = corner.corner(chain, range=[(0., 1.), (0., 1.), (0., 1.), (0., 1.)], labels=labels, show_titles=True,
                        quantities=(0.05, 0.95))
    dim = len(labels)
    # Extract the axes
    axes = np.array(fig.axes).reshape((dim, dim))
    if trues is not None:
        # Loop over the diagonal
        for i in range(dim):
            ax = axes[i, i]
            ax.axvline(trues[i], color="g")
        # Loop over the histograms
        for yi in range(dim):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(trues[xi], color="g")
                ax.axhline(trues[yi], color="g")
                ax.plot(trues[xi], trues[yi], "sg")
    plt.show()
    if file is not None:
        plt.savefig(file)


def plot_trace(chain, labels, trues=None, file=None):
    """

    :param chain:
    :param labels:
    :param trues:
    :param file:
    :return:
    """
    dim = len(labels)
    fig, axs = plt.subplots(nrows=dim, ncols=2, figsize=(10, 15))
    fig.subplots_adjust(hspace=0.75)
    for i in range(dim):
        ax = axs[i][0]
        ax.set_title('%s histogram' % labels[i])
        ax.hist(chain[:, i], bins=50, density=True, alpha=0.5)
        if trues is not None:
            ax.axvline(trues[i], color='r', label=r'$\theta_{true}$')
        ax.set_xlim(0, 1)
        ax.set_xlabel(labels[i])
        ax.set_ylabel('density')
        ax.legend()
        ax = axs[i][1]
        ax.set_ylim(0, 1)
        ax.set_title('%s Traceplot' % labels[i])
        ax.set_xlabel('Iteration')
        ax.set_ylabel(labels[i])
        if trues is not None:
            ax.axhline(trues[i], color='r', label=r'$\theta_{true}$')
        ax.plot(chain[:, i], alpha=0.4)
        ax.legend()

    plt.suptitle('GibbsPy TracePlot')
    plt.show()
    if file is not None:
        plt.savefig(file)

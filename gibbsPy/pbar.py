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
This File uses the same method as github.com/dfm/emcee/emcee/pbar.py to set up the tqdm progress bar for our Gibbs Sampling
"""
try:
    import tqdm
except ImportError:
    tqdm = None


# Set up a dummy class if we don't want a progress bar
class _NoProgress(object):
    """
    dummy wrapper class if we don't use a progress bar
    """

    def __init__(self):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, ct):
        pass


"""
This function will call the tqdm progress bar for the mcmc chain same as used in the reference pytthon file listed above
"""


def progress_bar(progress, total):
    """
    This function calls the tqdm progress bar or our dummy no progress bar class if we dont want to use one
    :param progress: this is a bool or string, if bool False will return the dummy bar, and true the default tqdm bar,
    if string must be a string to call a specific tqdm progress bar
    :param total: int that is the size of the progerss bar
    :return: this returns a tqdm progress bar object or our dummy wrapper class
    """

    if progress:
        if tqdm is None:
            return _NoProgress()
        else:
            if progress is True:
                return tqdm.tqdm(total=total)
            else:
                return getattr(tqdm, 'tqdm_' + progress)(total=total)

    else:
        return _NoProgress
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from warnings import warn


class ReactiveComplexes:
    """
    The base class for all reactive complex generators.
    """

    def __init__(self):
        self.options = self.Options()

    class Options:
        """
        Options attribute to be implemented by child classes
        """

        def __init__(self):
            raise NotImplementedError

    def set_options(self, option_dict):
        """
        Sets the options for the ReactiveComplexes from a dictionary.
        Generates a warning if an option is unknown.

        Parameters
        ----------
        option_dict :: Dict[str, Union[bool, int, float]]
            Dictionary with options to be used for generating reactive
            complexes.
        """

        for option_name, value in option_dict.items():
            if not hasattr(self.options, option_name):
                warn(
                    "Option '{}' is unknown and will be ignored.".format(option_name),
                    stacklevel=2,
                )
                continue
            setattr(self.options, option_name, value)

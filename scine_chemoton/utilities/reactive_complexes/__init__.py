#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from abc import ABC
from typing import Any, Dict
from warnings import warn


class ReactiveComplexes:
    """
    The base class for all reactive complex generators.
    """

    def __init__(self) -> None:
        self.options = self.Options()

    class Options(ABC):
        """
        Options attribute to be implemented by child classes
        """

    def set_options(self, option_dict: Dict[str, Any]) -> None:
        """
        Sets the options for the ReactiveComplexes from a dictionary.
        Generates a warning if an option is unknown.

        Parameters
        ----------
        option_dict : Dict[str, Any]
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

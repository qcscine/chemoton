#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""


import pathlib


def resources_root_path():
    """
    A small getter to abstract the position of this directory.

    Returns
    -------
    result :: str
        The path to the root directory of the test resources.
    """
    return pathlib.Path(__file__).parent.absolute()

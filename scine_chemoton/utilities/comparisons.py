#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""


def attribute_comparison(a: object, b: object) -> bool:
    """
    Compare two objects by their attributes. This is useful for comparing objects that are not the identical objects
    but have the same attributes and are of the same type.
    :param a: The first object.
    :param b: The second object.
    :return: True if the objects have the same attributes, False otherwise.
    """
    if not isinstance(a, type(b)):
        return False
    for attr in dir(a):
        if attr.startswith("__"):
            continue
        if callable(getattr(a, attr)):
            continue
        if getattr(a, attr) != getattr(b, attr):
            return False
    return True

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from ...utilities.masm import distinct_atoms, distinguish_components

import scine_molassembler as masm


def test_distinct_components():
    propane = masm.io.experimental.from_smiles("CCC")
    assert len(distinct_atoms(propane, h_only=True)) == 5
    assert len(distinct_atoms(propane, h_only=False)) == 4

    cyclopropane = masm.io.experimental.from_smiles("C1CC1")
    assert len(distinct_atoms(cyclopropane, h_only=True)) == 4
    assert len(distinct_atoms(cyclopropane, h_only=False)) == 2


def test_distinguish_components():
    components = [0, 1, 0, 0, 1, 1, 2]
    split = distinguish_components(components, lambda i: i % 2)
    assert len(split) == len(components)
    assert len(set(split)) == max(split) + 1
    assert len(set(split)) == 5
    assert split[0] == split[2]
    assert split[0] != split[3]
    assert split[1] == split[5]
    assert split[4] != split[5]

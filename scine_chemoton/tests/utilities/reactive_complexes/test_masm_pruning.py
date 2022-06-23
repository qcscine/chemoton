#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from ....utilities.masm import mol_to_cbor, get_atom_pairs
import scine_molassembler as masm

from functools import partial


def test_masm_pruning():
    propane = masm.io.experimental.from_smiles("CCC")
    propane.canonicalize()

    class MockStructure(object):
        graph: dict = {}

        def __init__(self, graph: dict):
            self.graph = graph

        def get_graph(self, key: str) -> str:
            return self.graph[key]

    graph = {"masm_cbor_graph": mol_to_cbor(propane), "masm_idx_map": str([(0, i) for i in range(propane.graph.V)])}
    pair_fn = partial(get_atom_pairs, MockStructure(graph))

    assert len(pair_fn((1, 1), "All")) == 3
    assert len(pair_fn((1, 2), "All")) == 8
    assert len(pair_fn((1, 3), "All")) == 10
    assert len(pair_fn((1, 4), "All")) == 11

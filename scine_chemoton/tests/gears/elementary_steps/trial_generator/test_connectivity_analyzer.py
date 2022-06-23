#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from itertools import combinations

# Third party imports
import scine_molassembler as masm
import scine_utilities as utils

# Local application imports
from .....gears.elementary_steps.trial_generator.connectivity_analyzer import ConnectivityAnalyzer, ReactionType
from .....utilities.masm import mol_to_cbor


class MockStructure(object):
    graph: dict = {}
    n_atoms: int = 0

    def __init__(self, graph: dict, n_atoms: int):
        self.graph = graph
        self.n_atoms = n_atoms

    def get_graph(self, key: str) -> str:
        return self.graph[key]

    def get_atoms(self):
        """
        Returns an empty atom collection with
        the size of the molecule
        """
        return utils.AtomCollection(self.n_atoms)


def test_reaction_types():
    cyclopropane = masm.io.experimental.from_smiles("C1CC1")
    cyclopropane.canonicalize()

    graph = {
        "masm_cbor_graph": mol_to_cbor(cyclopropane),
        "masm_idx_map": str([(0, i) for i in range(cyclopropane.graph.V)]),
    }
    structure = MockStructure(graph, len(list(cyclopropane.graph.atoms())))
    analyzer = ConnectivityAnalyzer(structure)
    reaction_type_fn = analyzer.get_reaction_type

    dis_reactions = [ReactionType.Dissociative, ReactionType.Disconnective]

    # Atom + atom combinations
    for i, j in combinations(cyclopropane.graph.atoms(), 2):
        reaction_type = reaction_type_fn([(i, j)])
        if cyclopropane.graph.adjacent(i, j):
            if cyclopropane.graph.can_remove(masm.BondIndex(i, j)):
                assert reaction_type == ReactionType.Dissociative
            else:
                assert reaction_type == ReactionType.Disconnective
        else:
            assert reaction_type == ReactionType.Associative

    # Atom + bond combinations
    for bond in cyclopropane.graph.bonds():
        for i in cyclopropane.graph.atoms():
            if i in bond:
                continue
            coord = ((i, bond[0]), (i, bond[1]))
            reaction_type = reaction_type_fn(coord)

            cross_adjacent = [cyclopropane.graph.adjacent(i, j) for j in bond]
            cross_count = cross_adjacent.count(True)

            if cross_count == 2:
                # A triangle! Occurs once e.g. in cyclopropane
                assert reaction_type in dis_reactions
            elif cross_count == 1:
                # Bond + already adjacent vertex -> Mixed type
                assert reaction_type == ReactionType.Mixed
            else:
                # Bond + non-adjacent vertex -> Associative
                assert reaction_type == ReactionType.Associative

    # Two bonds with 4 distinct atoms
    for a, b in combinations(cyclopropane.graph.bonds(), 2):
        # Must be different atom indices
        if any(i == j for i in a for j in b):
            continue

        # Interpret bonds as coords
        # Reaction between bonded atoms has to be dissociative/disconnective
        reaction_type = reaction_type_fn([a, b])
        assert reaction_type in dis_reactions

        # Interpret bonds as fragments
        coords = [((a[0], b[0]), (a[1], b[1])), ((a[0], b[1]), (a[1], b[0]))]

        for coord in coords:
            bond_count = [cyclopropane.graph.adjacent(pair[0], pair[1]) for pair in coord].count(True)
            if bond_count == 2:
                # both pairs bound
                assert reaction_type_fn(coord) in dis_reactions
            elif bond_count == 1:
                # one pair bound
                assert reaction_type_fn(coord) == ReactionType.Mixed
            elif bond_count == 0:
                # not bound
                assert reaction_type_fn(coord) == ReactionType.Associative


def test_adjacency_matrix():
    methylcyclopropane = masm.io.experimental.from_smiles("CC1CC1")
    methylcyclopropane.canonicalize()

    graph = {
        "masm_cbor_graph": mol_to_cbor(methylcyclopropane),
        "masm_idx_map": str([(0, i) for i in range(methylcyclopropane.graph.V)]),
    }
    structure = MockStructure(graph, len(list(methylcyclopropane.graph.atoms())))
    connectivity_analyzer = ConnectivityAnalyzer(structure)
    adjacency_matrix = connectivity_analyzer.get_adjacency_matrix()

    for i in methylcyclopropane.graph.atoms():
        for j in methylcyclopropane.graph.atoms():
            if methylcyclopropane.graph.adjacent(i, j):
                assert adjacency_matrix[i, j]
            else:
                assert not adjacency_matrix[i, j]


def test_get_graph_distance():
    cyclopropane = masm.io.experimental.from_smiles("C1CC1")
    cyclopropane.canonicalize()

    graph = {
        "masm_cbor_graph": mol_to_cbor(cyclopropane),
        "masm_idx_map": str([(0, i) for i in range(cyclopropane.graph.V)]),
    }
    structure = MockStructure(graph, len(list(cyclopropane.graph.atoms())))
    connectivity_analyzer = ConnectivityAnalyzer(structure)

    for i in cyclopropane.graph.atoms():
        for j in cyclopropane.graph.atoms():
            if i == j:
                assert connectivity_analyzer.get_graph_distance(i, j) == 0
            elif cyclopropane.graph.adjacent(i, j):
                assert connectivity_analyzer.get_graph_distance(i, j) == 1
            elif (
                cyclopropane.graph.elements()[i] == utils.ElementType.H
                and cyclopropane.graph.elements()[j] == utils.ElementType.H
            ):
                if next(cyclopropane.graph.adjacents(i)) == next(cyclopropane.graph.adjacents(j)):
                    # Two H bound to the same C
                    assert connectivity_analyzer.get_graph_distance(i, j) == 2
                else:
                    # Two H bound to different C
                    assert connectivity_analyzer.get_graph_distance(i, j) == 3
            else:
                # C and an H bound to a different C
                assert connectivity_analyzer.get_graph_distance(i, j) == 2


def test_pair_is_bound():
    water = masm.io.experimental.from_smiles("O")
    water.canonicalize()

    graph = {
        "masm_cbor_graph": mol_to_cbor(water),
        "masm_idx_map": str([(0, i) for i in range(water.graph.V)]),
    }
    structure = MockStructure(graph, len(list(water.graph.atoms())))
    connectivity_analyzer = ConnectivityAnalyzer(structure)

    for i in water.graph.atoms():
        for j in water.graph.atoms():
            if i == j:
                continue
            elif water.graph.adjacent(i, j):
                assert connectivity_analyzer.pair_is_bound(i, j)

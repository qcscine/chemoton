#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import ast
from copy import deepcopy
from enum import Enum
from math import inf
from typing import Tuple, List

from scipy.sparse import lil_matrix

# Third party imports
import scine_molassembler as masm
import scine_database as db

# Local application imports
from ....utilities.masm import deserialize_molecules


class ReactionType(Enum):
    """
    Different types of reactions to be distinguished here.

    - Mixed: Neither purely associative nor dissociative
    - Associative: Only new bonds are generated
    - Dissociative: Only bond breaking but the molecule is not split
    - Disconnective: Only bond breaking and the molecule is split
    """

    Mixed = None
    Associative = 0
    Dissociative = 1
    Disconnective = 2


# TODO Consider caching for fewer deserializations
class ConnectivityAnalyzer:
    """
    A class holding the molecule representation of a structure allowing to
    retrieve information about its bonding situation and the connectivity
    effect of trial reaction coordinates.

    Attributes
    ----------
    structure : db.Structure
        The structure of interest. It has to be connected to a database and has
        to have a graph attached.
    """

    def __init__(self, structure: db.Structure) -> None:
        self.structure = structure
        self.molecules = deserialize_molecules(self.structure)
        self.graphs = [m.graph for m in self.molecules]
        self.idx_map = ast.literal_eval(structure.get_graph("masm_idx_map"))

    def pair_is_bound(self, atom1: int, atom2: int) -> bool:
        """
        Check whether two atoms are bound in a structure according to its
        graph.

        Parameters
        ----------
        atom1 : int
            Index of the first atom.
        atom2 : int
            Index of the second atom.

        Returns
        -------
        bool
            True if the two atoms are bound and False if not.
        """

        # c,d: idx of corresponding molecules
        # m_i/m_j: idx of i/j within the corresponding molecule
        component_idx_i, mol_idx_i = self.idx_map[atom1]
        component_idx_j, mol_idx_j = self.idx_map[atom2]
        # If different molecules not bound
        if component_idx_i != component_idx_j:
            return False

        # Same molecule
        assert mol_idx_i != mol_idx_j
        return self.graphs[component_idx_i].adjacent(mol_idx_i, mol_idx_j)

    def get_reaction_type(self, reactive_pair_list: List[Tuple[int, int]]) -> ReactionType:
        """
        Determines what kind of intrastructural reaction
        (associative, dissociative, disconnective) a reactive complex guess
        describes.

        Parameters
        ----------
        reactive_pair_list : List[Tuple[int]]
            List of reacting atom index pairs

        Returns
        -------
        reaction_type : ElementaryStepGear.ElementaryStepType
            What kind of reaction is initiated by this reactive complex.

        """
        associative_count = 0
        dissociated_bonds = []
        for i, j in reactive_pair_list:
            component_idx_i, mol_idx_i = self.idx_map[i]
            component_idx_j, mol_idx_j = self.idx_map[j]
            if component_idx_i != component_idx_j:
                # Different molecules, always associative
                associative_count += 1
                continue

            # Atom cannot react with itself
            assert mol_idx_i != mol_idx_j
            if self.graphs[component_idx_i].adjacent(mol_idx_i, mol_idx_j):
                dissociated_bonds.append((component_idx_i, masm.BondIndex(mol_idx_i, mol_idx_j)))
            else:
                associative_count += 1

        Type = ReactionType

        # Purely associative
        if associative_count > 0 and len(dissociated_bonds) == 0:
            return Type.Associative

        # Mixed reaction type
        if associative_count > 0 and len(dissociated_bonds) > 0:
            return Type.Mixed

        # Now to distinguish dissociative and disconnective
        dissociated_bonds = list(set(dissociated_bonds))
        modified_graphs = deepcopy(self.graphs)
        for c, bond in dissociated_bonds:
            if modified_graphs[c].can_remove(bond):
                modified_graphs[c].remove_bond(bond)
            else:
                return Type.Disconnective

        return Type.Dissociative

    def get_adjacency_matrix(self) -> lil_matrix:
        """
        Gets the adjacency matrix for the structure with the indices ordered as
        in the structure.

        Returns
        -------
        lil_matrix
            The adjacency matrix which is `True` if there is a direct bond and
            `False` otherwise.
        """
        n_atoms = self.structure.get_atoms().size()
        adjacency_matrix = lil_matrix((n_atoms, n_atoms), dtype=bool)

        # Loop over bonds
        for component_idx, graph in enumerate(self.graphs):
            for bond in graph.bonds():
                s_idx1 = self._get_structure_idx(component_idx, bond[0])
                s_idx2 = self._get_structure_idx(component_idx, bond[1])
                adjacency_matrix[s_idx1, s_idx2] = True

        # Make symmetric
        rows, cols = adjacency_matrix.nonzero()
        adjacency_matrix[cols, rows] = adjacency_matrix[rows, cols]
        return adjacency_matrix

    def get_graph_distance(self, atom1: int, atom2: int):
        """
        Calculates the graph distance between the atoms with indices `atom1` and
        `atom2`.

        Parameters
        ----------
        atom1 : int
            The index of the first atom of interest.
        atom2 : int
            The index of the second atom of interest.

        Returns
        -------
        int
            The graph distance between atoms 1 and 2.

        Notes
        -----
        Return infinity if the two atoms do not belong to the same graph
        """
        component_idx1, mol_idx_1 = self.idx_map[atom1]
        component_idx2, mol_idx_2 = self.idx_map[atom2]

        if component_idx1 != component_idx2:
            # Graph distance between different Molecules is not defined
            return inf
        return masm.distance(mol_idx_1, self.graphs[component_idx1])[mol_idx_2]

    def _get_structure_idx(self, component_idx: int, mol_idx: int) -> int:
        """
        Get the index of an atom in the structure from its component index
        and index in the molecule.

        Parameters
        ----------
        component_idx : int
            Which component/molecule the atom belongs to.
        mol_idx : int
            The index within the molecule.

        Returns
        -------
        int
            The index of the atom within the structure.
        """
        return self.idx_map.index((component_idx, mol_idx))

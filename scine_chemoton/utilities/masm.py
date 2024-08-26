#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import ast
import numpy as np

import scine_molassembler as masm
import scine_database as db
import scine_utilities as utils

from typing import List, Set, Any, Callable, Tuple, Optional, DefaultDict
from collections import defaultdict, namedtuple


def mol_to_cbor(mol: masm.Molecule) -> str:
    """
    Convert a molecule into a base-64 encoded CBOR serialization

    Parameters
    ----------
    mol : masm.Molecule
        Molecule to serialize

    Returns
    -------
    serialization : str
        The string-serialized molecule representation
    """
    serializer = masm.JsonSerialization
    cbor_format = serializer.BinaryFormat.CBOR
    serialization = serializer(mol)
    cbor_binary = serialization.to_binary(cbor_format)
    return serializer.base_64_encode(cbor_binary)


def mol_from_cbor(cbor_str: str) -> masm.Molecule:
    """
    Convert base-64 encoded CBOR to a molassembler Molecule

    Converts a single base-64 encoded CBOR string (no ';' separator as stored
    in the database) into a molecule

    Parameters
    ----------
    cbor_str : str
        String to deserialize into a Molecule

    Returns
    -------
    molecule : masm.Molecule
        The deserialized molecule
    """
    serializer = masm.JsonSerialization
    cbor_binary = serializer.base_64_decode(cbor_str)
    cbor_format = serializer.BinaryFormat.CBOR
    serialization = serializer(cbor_binary, cbor_format)
    return serialization.to_molecule()


def mols_from_properties(structure: db.Structure, properties: db.Collection) -> Optional[List[masm.Molecule]]:
    """
    Generate all molecules based on atomic positions in a structure and
    the bond orders stored in attache properties.

    Parameters
    ----------
    structure : db.Structure
        The structure whose contained molecule(s) to analyze.
    properties : db.Collection
        The collection holding all properties.

    Returns
    -------
    molecules : List[masm.Molecule]
        A list of all the molecules contained in the database structure.
    """
    atoms = structure.get_atoms()
    distance_bos = utils.BondDetector.detect_bonds(atoms)
    # Check/get bond orders
    if not structure.has_property('bond_orders'):
        return None
    bo_property_id = structure.get_property('bond_orders')
    bo_property = db.SparseMatrixProperty(bo_property_id)
    bo_property.link(properties)
    # Update bond orders
    bond_orders = utils.BondOrderCollection(len(atoms))
    bond_orders.matrix = bo_property.data()
    final_bo_matrix = (bond_orders.matrix).maximum(distance_bos.matrix)
    final_bo_matrix = (final_bo_matrix).multiply(distance_bos.matrix)
    bos = utils.BondOrderCollection(len(atoms))
    bos.matrix = final_bo_matrix
    # Build and return molecules
    return masm.interpret.molecules(atoms, bos, set(), {}, masm.interpret.BondDiscretization.Binary).molecules


def deserialize_molecules(structure: db.Structure) -> List[masm.Molecule]:
    """
    Retrieves all molecules stored for a structure

    Parameters
    ----------
    structure : db.Structure
        The structure whose contained molecules to deserialize

    Returns
    -------
    molecules : List[masm.Molecule]
        A list of all the molecules contained in the database structure
    """
    multiple_cbors = structure.get_graph("masm_cbor_graph")
    return [mol_from_cbor(m) for m in multiple_cbors.split(";")]


def distinguish_components(components: List[int], map_unary: Callable[[int], Any]) -> List[int]:
    """
    Splits components by the result of a unary mapping function

    Parameters
    ----------
    components : List[int]
        A per-index mapping to a component index. Must contain only sequential
        numbers starting from zero.
    map_unary : Callable[[int], Any]
        A unary callable that is called with an index, not a component index,
        yielding some comparable type. Components of indices are then split
        by matching results of invocations of this callable.

    Returns
    -------
    components : List[int]
        A per-index mapping to a component index. Contains only sequential
        numbers starting from zero.
    """
    assert len(set(components)) == max(components) + 1
    component_sets: List[Set[int]]
    component_sets = [set() for _ in range(max(components) + 1)]
    for i, c in enumerate(components):
        component_sets[c].add(i)

    def split_by_unary(indices: Set[int]) -> List[Set[int]]:
        results = defaultdict(set)
        for i in indices:
            results[map_unary(i)].add(i)

        return list(results.values())

    split_component_sets = []
    for subset in component_sets:
        split_component_sets.extend(split_by_unary(subset))

    new_components = [0 for _ in range(len(components))]
    for c, subset in enumerate(split_component_sets):
        for i in subset:
            new_components[i] = c

    return new_components


def distinct_components(mol: masm.Molecule, h_only: bool) -> List[int]:
    """
    Generates a flat map of atom index to component identifier

    Parameters
    ----------
    mol : masm.Molecule
        A molecule whose atoms to generate distinct components for
    h_only : bool
        Whether to only apply ranking deduplication to hydrogen atoms

    Returns
    -------
    components : List[int]
        A flat per-atom index mapping to a component index. Contains only
        sequential numbers starting from zero.
    """
    components = masm.ranking_equivalent_groups(mol)

    if h_only:
        return distinguish_components(components, lambda i: mol.graph.element_type(i) == utils.ElementType.H)

    return components


def distinct_atoms(mol: masm.Molecule, h_only: bool) -> List[int]:
    """
    Generates a list of distinct atom indices


    Parameters
    ----------
    mol : masm.Molecule
        A molecule whose atoms to list distinct atoms for
    h_only : bool
        Whether to only apply ranking deduplication to hydrogen atoms

    Returns
    -------
    components : List[int]
        A list of ranking-distinct atoms
    """

    def is_h(i: int) -> bool:
        return mol.graph.element_type(i) == utils.ElementType.H

    distinct = masm.ranking_distinct_atoms(mol)

    if h_only:
        distinct_hs = [i for i in distinct if is_h(i)]
        heavy_atoms = [i for i in range(mol.graph.V) if not is_h(i)]
        return distinct_hs + heavy_atoms

    return distinct


def make_sorted_pair(a: int, b: int) -> Tuple[int, int]:
    if b < a:
        return b, a

    return a, b


ComponentDistanceTuple = namedtuple("ComponentDistanceTuple", ["mol_idx", "components", "distance"])
StructureIndexPair = Tuple[int, int]
EquivalentPairingsMap = DefaultDict[ComponentDistanceTuple, Set[StructureIndexPair]]


def pruned_atom_pairs(
    molecules: List[masm.Molecule], idx_map: List[Tuple[int, int]], distance_bounds: Tuple[int, int], prune: str
) -> Set[Tuple[int, int]]:
    assert prune in ["Hydrogen", "All"]

    def structure_idx(c: int, i: int) -> int:
        return idx_map.index((c, i))

    pairings: EquivalentPairingsMap = defaultdict(set)

    # Idea: For each distinct atom in the molecule, distinguish the
    # distinct components of the molecule by the distance to the selected
    # atom. Then, store one atom pairing for each distinct set of component
    # and distance combination.
    for mol_idx, molecule in enumerate(molecules):
        distinct = distinct_atoms(molecule, prune == "Hydrogen")
        components = distinct_components(molecule, prune == "Hydrogen")
        for i in distinct:
            distances = masm.distance(i, molecule.graph)
            local_components = distinguish_components(components,
                                                      lambda x: distances[x])  # pylint: disable=cell-var-from-loop
            considered_components = set()
            for j, c in enumerate(local_components):
                if c in considered_components or i == j:
                    continue

                considered_components.add(c)
                if min(distance_bounds) <= distances[j] <= max(distance_bounds):
                    key = ComponentDistanceTuple(
                        mol_idx=mol_idx,
                        components=make_sorted_pair(components[i], components[j]),
                        distance=distances[j],
                    )
                    if key not in pairings:
                        s_ij = make_sorted_pair(*[structure_idx(mol_idx, x) for x in [i, j]])
                        pairings[key].add(s_ij)

    # Pick one element from each set of same-key pairings
    return set([next(iter(subset)) for subset in pairings.values()])


def unpruned_atom_pairs(
    molecules: List[masm.Molecule], idx_map: List[Tuple[int, int]], distance_bounds: Tuple[int, int]
) -> Set[Tuple[int, int]]:
    """Helper function to generate the set of unpruned atom pairs"""

    def structure_idx(c: int, i: int) -> int:
        return idx_map.index((c, i))

    pairs: Set[Tuple[int, int]] = set()

    for component, molecule in enumerate(molecules):
        for i in molecule.graph.atoms():
            distances = np.array(masm.distance(i, molecule.graph))
            partners = np.nonzero((distances <= max(distance_bounds)) & (distances >= min(distance_bounds)))[0]

            # Back-transform to structure indices and add to set
            s_i = structure_idx(component, i)
            s_partners = [structure_idx(component, j) for j in partners]
            pairs |= set(make_sorted_pair(s_i, s_j) for s_j in s_partners)

    return pairs


def get_atom_pairs(
    structure: db.Structure,
    distance_bounds: Tuple[int, int],
    prune: str = "None",
    superset: Optional[Set[Tuple[int, int]]] = None,
) -> Set[Tuple[int, int]]:
    """
    Gets a list of all atom pairs whose graph distance is smaller or equal
    to `max_graph_distance` and larger or equal to `min_graph_distance` on
    the basis of the interpreted graph representation.

    Parameters
    ----------
    structure : db.Structure
        The structure that is investigated
    distance_bounds : Tuple[int, int]
        The minimum and maximum distance between two points that is allowed so
        that they are considered a valid atom pair.
    prune : str
        Whether to prune atom pairings by Molassembler's ranking distinct
        atoms descriptor. Allowed values: `'None'`, `'Hydrogen'`, `'All'`
    superset : Optional[Set[Tuple[int, int]]]
        Optional superset of pairs to filter. If set, will filter the passed
        set. Otherwise, generates atom pairings from all possible pairs in the
        molecule.

    Returns
    -------
    pairs : Set[Tuple[int, int]]
        The indices of valid atom pairs.
    """
    valid_option_values = ["None", "Hydrogen", "All"]
    if prune not in valid_option_values:
        msg = "Option for masm atom pruning invalid: {}"
        raise RuntimeError(msg.format(prune))

    molecules = deserialize_molecules(structure)
    idx_map = ast.literal_eval(structure.get_graph("masm_idx_map"))

    if prune in ["Hydrogen", "All"]:
        if superset is not None:
            # Expand superset into the pruned keyspace
            superset_dict: EquivalentPairingsMap = defaultdict(set)
            molecule_components = [distinct_components(mol, prune == "Hydrogen") for mol in molecules]
            for s_i, s_j in superset:
                mol_idx, i = idx_map[s_i]
                cmp_idx, j = idx_map[s_j]
                assert mol_idx == cmp_idx
                components = molecule_components[mol_idx]
                key = ComponentDistanceTuple(
                    mol_idx=mol_idx,
                    components=make_sorted_pair(components[i], components[j]),
                    distance=masm.distance(i, molecules[mol_idx].graph)[j],
                )
                if min(distance_bounds) <= key.distance <= max(distance_bounds):
                    superset_dict[key].add(make_sorted_pair(s_i, s_j))

            # Pick one element from each set of same-key pairings
            return set([next(iter(subset)) for subset in superset_dict.values()])

        return pruned_atom_pairs(molecules, idx_map, distance_bounds, prune)

    if superset is not None:
        pairs: Set[Tuple[int, int]] = set()
        for s_i, s_j in superset:
            mol_idx, i = idx_map[s_i]
            cmp_idx, j = idx_map[s_j]
            assert mol_idx == cmp_idx
            distance = masm.distance(i, molecules[mol_idx])[j]
            if min(distance_bounds) <= distance <= max(distance_bounds):
                pairs.add(make_sorted_pair(s_i, s_j))

        return pairs

    return unpruned_atom_pairs(molecules, idx_map, distance_bounds)

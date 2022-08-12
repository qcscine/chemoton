#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
import json
import unittest
from itertools import combinations, product

# Third party imports
import scine_database as db

# Local application tests imports
from ... import test_database_setup as db_setup
from ...resources import resources_root_path

# Local application imports
from ....gears.elementary_steps.reactive_site_filters import (
    MasmChemicalRankingFilter,
    SimpleRankingFilter,
    ReactiveSiteFilter,
    ReactiveSiteFilterOrArray,
    ReactiveSiteFilterAndArray,
    AtomRuleBasedFilter,
    ElementWiseReactionCoordinateFilter,
    HeuristicPolarizationReactionCoordinateFilter
)
from ....gears.elementary_steps.compound_filters import (
    CompoundFilter
)


class ReactiveSiteFiltersTests(unittest.TestCase):

    def test_default_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_default_filter")

        # Get collections
        structures = manager.get_collection("structures")

        # Add structure data
        rr = resources_root_path()
        cyclohexene = json.load(open(os.path.join(rr, "cyclohexene.json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, "cyclohexene.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", cyclohexene["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", cyclohexene["masm_idx_map"])
        structure.set_aggregate(db.ID())

        # Setup filter
        f = ReactiveSiteFilter()

        # Filter and check
        allowed_atoms = f.filter_atoms([structure, structure], list(range(16)))
        assert len(allowed_atoms) == 16

        allowed_pairs = f.filter_atom_pairs([structure], list(combinations(range(16), 2)))
        assert len(allowed_pairs) == 120

        # Just two random coordinates
        allowed_coordinates = f.filter_reaction_coordinates([structure], [((0, 1), (1, 2)), ((0, 1), (3, 4))])
        assert len(allowed_coordinates) == 2

        assert [] == f.filter_atoms([structure], [])
        assert [] == f.filter_atom_pairs([structure], [])
        assert [] == f.filter_reaction_coordinates([structure], [])

        # Cleaning
        manager.wipe()

    def test_filter_chain_failure_logic_and(self):
        with self.assertRaises(TypeError) as context:
            _ = (ReactiveSiteFilter() & CompoundFilter())
        self.assertTrue('ReactiveSiteFilter' in str(context.exception))

    def test_filter_chain_failure_logic_or(self):
        with self.assertRaises(TypeError) as context:
            _ = (ReactiveSiteFilter() | CompoundFilter())
        self.assertTrue('ReactiveSiteFilter' in str(context.exception))

    def test_filter_chain_failure_class_and(self):
        with self.assertRaises(TypeError) as context:
            ReactiveSiteFilterAndArray([ReactiveSiteFilter(), CompoundFilter()])
        self.assertTrue('ReactiveSiteFilterAndArray' in str(context.exception))

    def test_filter_chain_failure_class_or(self):
        with self.assertRaises(TypeError) as context:
            ReactiveSiteFilterOrArray([ReactiveSiteFilter(), CompoundFilter()])
        self.assertTrue('ReactiveSiteFilterOrArray' in str(context.exception))

    def test_filter_chain_derived_classes_or(self):
        r1 = ReactiveSiteFilterOrArray([SimpleRankingFilter(), MasmChemicalRankingFilter()])
        assert isinstance(r1, ReactiveSiteFilterOrArray)
        assert isinstance(r1, ReactiveSiteFilter)
        r2 = (SimpleRankingFilter() | MasmChemicalRankingFilter())
        assert isinstance(r2, ReactiveSiteFilterOrArray)
        assert isinstance(r2, ReactiveSiteFilter)

    def test_filter_chain_derived_classes_and(self):
        r1 = ReactiveSiteFilterAndArray([SimpleRankingFilter(), MasmChemicalRankingFilter()])
        assert isinstance(r1, ReactiveSiteFilterAndArray)
        assert isinstance(r1, ReactiveSiteFilter)
        r2 = (SimpleRankingFilter() & MasmChemicalRankingFilter())
        assert isinstance(r2, ReactiveSiteFilterAndArray)
        assert isinstance(r2, ReactiveSiteFilter)

    def test_empty_and(self):
        manager = db_setup.get_clean_db("chemoton_test_empty_and")

        # Get collections
        structures = manager.get_collection("structures")

        # Add structure data
        rr = resources_root_path()
        cyclohexene = json.load(open(os.path.join(rr, "cyclohexene.json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, "cyclohexene.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", cyclohexene["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", cyclohexene["masm_idx_map"])
        structure.set_aggregate(db.ID())

        water = json.load(open(os.path.join(rr, "water.json"), "r"))
        structure1 = db.Structure()
        structure1.link(structures)
        structure1.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure1.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure1.set_graph("masm_cbor_graph", water["masm_cbor_graph"])
        structure1.set_graph("masm_idx_map", water["masm_idx_map"])
        structure1.set_aggregate(db.ID())

        # Setup filter
        noneFilter = ReactiveSiteFilterAndArray()

        # Filter and check pruning no atoms
        allowed_atoms = noneFilter.filter_atoms([structure], list(range(16)))
        assert len(allowed_atoms) == 16
        pairs = [(0, 1), (2, 3), (13, 14)]
        allowed_pairs = noneFilter.filter_atom_pairs([structure, structure1], pairs)
        assert len(allowed_pairs) == 3
        coords = [((0, 1),), ((2, 3), (4, 5))]
        allowed_coordinates = noneFilter.filter_reaction_coordinates([structure], coords)
        assert len(allowed_coordinates) == 2
        assert [] == noneFilter.filter_atoms([structure], [])
        assert [] == noneFilter.filter_atom_pairs([structure], [])
        assert [] == noneFilter.filter_reaction_coordinates([structure], [])

    def test_empty_or(self):
        manager = db_setup.get_clean_db("chemoton_test_empty_or")

        # Get collections
        structures = manager.get_collection("structures")

        # Add structure data
        rr = resources_root_path()
        cyclohexene = json.load(open(os.path.join(rr, "cyclohexene.json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, "cyclohexene.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", cyclohexene["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", cyclohexene["masm_idx_map"])
        structure.set_aggregate(db.ID())

        water = json.load(open(os.path.join(rr, "water.json"), "r"))
        structure1 = db.Structure()
        structure1.link(structures)
        structure1.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure1.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure1.set_graph("masm_cbor_graph", water["masm_cbor_graph"])
        structure1.set_graph("masm_idx_map", water["masm_idx_map"])
        structure1.set_aggregate(db.ID())

        # Setup filter
        noneFilter = ReactiveSiteFilterOrArray()

        # Filter and check pruning of everything
        allowed_atoms = noneFilter.filter_atoms([structure], list(range(16)))
        assert len(allowed_atoms) == 0
        pairs = [(0, 1), (2, 3), (13, 14)]
        allowed_pairs = noneFilter.filter_atom_pairs([structure, structure1], pairs)
        assert len(allowed_pairs) == 0
        coords = [((0, 1),), ((2, 3), (4, 5))]
        allowed_coordinates = noneFilter.filter_reaction_coordinates([structure], coords)
        assert len(allowed_coordinates) == 0
        assert [] == noneFilter.filter_atoms([structure], [])
        assert [] == noneFilter.filter_atom_pairs([structure], [])
        assert [] == noneFilter.filter_reaction_coordinates([structure], [])

    def test_masm_ranking_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_symmetry_filter")

        # Get collections
        structures = manager.get_collection("structures")

        # Add structure data
        rr = resources_root_path()
        cyclohexene = json.load(open(os.path.join(rr, "cyclohexene.json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, "cyclohexene.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", cyclohexene["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", cyclohexene["masm_idx_map"])
        structure.set_aggregate(db.ID())

        water = json.load(open(os.path.join(rr, "water.json"), "r"))
        structure1 = db.Structure()
        structure1.link(structures)
        structure1.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure1.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure1.set_graph("masm_cbor_graph", water["masm_cbor_graph"])
        structure1.set_graph("masm_idx_map", water["masm_idx_map"])
        structure1.set_aggregate(db.ID())

        # Setup filter
        noneFilter = MasmChemicalRankingFilter(prune="None")
        hFilter = MasmChemicalRankingFilter(prune="Hydrogen")
        allFilter = MasmChemicalRankingFilter(prune="All")

        # Filter and check pruning no atoms
        allowed_atoms = noneFilter.filter_atoms([structure], list(range(16)))
        assert len(allowed_atoms) == 16
        pairs = [(0, 1), (2, 3), (13, 14)]
        allowed_pairs = noneFilter.filter_atom_pairs([structure, structure1], pairs)
        assert len(allowed_pairs) == 3
        coords = [((0, 1),), ((2, 3), (4, 5))]
        allowed_coordinates = noneFilter.filter_reaction_coordinates([structure], coords)
        assert len(allowed_coordinates) == 2
        assert [] == noneFilter.filter_atoms([structure], [])
        assert [] == noneFilter.filter_atom_pairs([structure], [])
        assert [] == noneFilter.filter_reaction_coordinates([structure], [])

        # Check filter with pruning hydrogens
        allowed_atoms = hFilter.filter_atoms([structure, structure1], list(range(19)))
        assert len(allowed_atoms) == 14
        # Pair and coordinate filter are taken over from base class i.e. they do not filter out anything
        allowed_pairs = hFilter.filter_atom_pairs([structure], list(combinations(range(16), 2)))
        assert len(allowed_pairs) == 120
        coords = [((2, 3), (4, 5)), ((0, 17),)]
        allowed_coordinates = hFilter.filter_reaction_coordinates([structure, structure1], coords)
        assert len(allowed_coordinates) == 2
        assert [] == hFilter.filter_atoms([structure], [])
        assert [] == hFilter.filter_atom_pairs([structure], [])
        assert [] == hFilter.filter_reaction_coordinates([structure], [])

        # Check filter with pruning all
        allowed_atoms = allFilter.filter_atoms([structure, structure1], [i for i in range(19)])
        assert len(allowed_atoms) == 14
        bonds = [(2, 3), (4, 7)]
        # Pair and coordinate filter are taken over from base class i.e. they do not filter out anything
        allowed_pairs = allFilter.filter_atom_pairs(structure, bonds)
        assert allowed_pairs == bonds
        coords = [((3, 4),), ((7, 8),), ((0, 2))]
        allowed_coordinates = allFilter.filter_reaction_coordinates(structure, coords)
        assert len(allowed_coordinates) == 3
        assert [] == allFilter.filter_atoms([structure], [])
        assert [] == allFilter.filter_atom_pairs([structure], [])
        assert [] == allFilter.filter_reaction_coordinates([structure], [])

        # Cleaning
        manager.wipe()

    def test_simple_ranking_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_simple_ranking_filter")

        # Get collections
        structures = manager.get_collection("structures")

        # Add structure data
        rr = resources_root_path()
        cyclohexene = json.load(open(os.path.join(rr, "cyclohexene.json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, "cyclohexene.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", cyclohexene["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", cyclohexene["masm_idx_map"])
        structure.set_aggregate(db.ID())

        water = json.load(open(os.path.join(rr, "water.json"), "r"))
        structure1 = db.Structure()
        structure1.link(structures)
        structure1.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure1.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure1.set_graph("masm_cbor_graph", water["masm_cbor_graph"])
        structure1.set_graph("masm_idx_map", water["masm_idx_map"])
        structure1.set_aggregate(db.ID())

        # Setup filter
        f = SimpleRankingFilter(atom_threshold=1, pair_threshold=5, coordinate_threshold=4)

        # Filter and check
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(19)))
        # C and O and H in H2O
        assert len(allowed_atoms) == 9

        allowed_pairs = f.filter_atom_pairs([structure], list(combinations(range(16), 2)))
        # All C-C combinations
        assert len(allowed_pairs) == 15

        # All intermolecular coords with one atom per molecule
        coords = list((inter_pair,) for inter_pair in product(range(16), range(16, 19)))
        allowed_coordinates = f.filter_reaction_coordinates([structure, structure1], coords)
        # All O-C and H-C with H attached to O coordinates should pass the threshold
        assert len(allowed_coordinates) == 18

        # Every combination twice per coord effectively inducing the ranking value to be doubled
        double_coords = [coord + coord for coord in coords]
        allowed_coordinates = f.filter_reaction_coordinates([structure, structure1], double_coords)
        # With double counting all intermolecular coords should survive
        assert len(allowed_coordinates) == 48

        assert [] == f.filter_atoms([structure], [])
        assert [] == f.filter_atom_pairs([structure], [])
        assert [] == f.filter_reaction_coordinates([structure], [])

        # Cleaning
        manager.wipe()

    def test_filter_or_array(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_filter_or_array")

        # Get collections
        structures = manager.get_collection("structures")

        # Add structure data
        rr = resources_root_path()
        cyclohexene = json.load(open(os.path.join(rr, "cyclohexene.json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, "cyclohexene.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", cyclohexene["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", cyclohexene["masm_idx_map"])
        structure.set_aggregate(db.ID())

        water = json.load(open(os.path.join(rr, "water.json"), "r"))
        structure1 = db.Structure()
        structure1.link(structures)
        structure1.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure1.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure1.set_graph("masm_cbor_graph", water["masm_cbor_graph"])
        structure1.set_graph("masm_idx_map", water["masm_idx_map"])
        structure1.set_aggregate(db.ID())

        # Setup filter
        f = SimpleRankingFilter(
            atom_threshold=1,
            pair_threshold=0,
            coordinate_threshold=999) | MasmChemicalRankingFilter(
            prune="Hydrogen")

        # Filter and check
        allowed_atoms = f.filter_atoms([structure], list(range(16)))
        # Hydrogens do not survive SimpleRankingFilter but all but 4 survive MasmChemicalRankingFilter
        assert len(allowed_atoms) == 12

        allowed_pairs = f.filter_atom_pairs([structure], list(combinations(range(16), 2)))
        # All 16 choose 2 combinations survive SimpleRankingFilter
        assert len(allowed_pairs) == 120

        coords = (((0, 1),), ((0, 17), (1, 3)), ((2, 3),))
        allowed_coordinates = f.filter_reaction_coordinates([structure, structure1], coords)
        # MasmChemicalRanking filter_reaction_coordinates is base method so all coordinates should survive
        assert len(allowed_coordinates) == 3

        assert [] == f.filter_atoms([structure], [])
        assert [] == f.filter_atom_pairs([structure], [])
        assert [] == f.filter_reaction_coordinates([structure], [])

        # Cleaning
        manager.wipe()

    def test_filter_and_array(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_filter_and_array")

        # Get collections
        structures = manager.get_collection("structures")

        # Add structure data
        rr = resources_root_path()
        cyclohexene = json.load(open(os.path.join(rr, "cyclohexene.json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, "cyclohexene.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", cyclohexene["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", cyclohexene["masm_idx_map"])
        structure.set_aggregate(db.ID())

        water = json.load(open(os.path.join(rr, "water.json"), "r"))
        structure1 = db.Structure()
        structure1.link(structures)
        structure1.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure1.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure1.set_graph("masm_cbor_graph", water["masm_cbor_graph"])
        structure1.set_graph("masm_idx_map", water["masm_idx_map"])
        structure1.set_aggregate(db.ID())

        # Setup filter
        f = SimpleRankingFilter(atom_threshold=1, pair_threshold=5, coordinate_threshold=4) & MasmChemicalRankingFilter(
            prune="Hydrogen"
        )

        # Filter and check
        allowed_atoms = f.filter_atoms([structure], list(range(16)))
        assert len(allowed_atoms) == 6

        allowed_pairs = f.filter_atom_pairs([structure], list(combinations(range(16), 2)))
        # All pairs pass MasmChemicalRanking, only C-C SimpleRanking
        assert len(allowed_pairs) == 15

        coords = list((inter_pair,) for inter_pair in product(range(16), range(16, 19)))
        allowed_coordinates = f.filter_reaction_coordinates([structure, structure1], coords)
        assert len(allowed_coordinates) == 18

        assert [] == f.filter_atoms([structure], [])
        assert [] == f.filter_atom_pairs([structure], [])
        assert [] == f.filter_reaction_coordinates([structure], [])

        # Cleaning
        manager.wipe()

    def test_atom_rule_based_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_rule_based_filter")

        # Get collections
        structures = manager.get_collection("structures")

        # Add structure data
        rr = resources_root_path()
        proline_propanal_adduct = json.load(open(os.path.join(rr, "proline_propanal_adduct.json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, "proline_propanal_adduct.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", proline_propanal_adduct["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", proline_propanal_adduct["masm_idx_map"])
        structure.set_compound(db.ID())

        water = json.load(open(os.path.join(rr, "water.json"), "r"))
        structure1 = db.Structure()
        structure1.link(structures)
        structure1.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure1.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure1.set_graph("masm_cbor_graph", water["masm_cbor_graph"])
        structure1.set_graph("masm_idx_map", water["masm_idx_map"])
        structure1.set_compound(db.ID())

        # trivial checks
        rules = {
            'H': True,
            'C': True,
            'N': True,
            'O': True
        }
        f = AtomRuleBasedFilter(rules=rules)
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(27)))
        reference = range(27)
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in reference

        rules = {
            'H': False,
            'C': False,
            'N': True,
            'O': False
        }
        f = AtomRuleBasedFilter(rules=rules)
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(27)))
        reference = [4]
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in reference

        rules = {
            'C': [AtomRuleBasedFilter.ReactiveRuleFilterAndArray([('C', 1), ('O', 1), ('N', 1)])]
        }
        f = AtomRuleBasedFilter(rules=rules)
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(27)))
        reference = [2]
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in reference

        rules = {
            'H': False,
            'C': [AtomRuleBasedFilter.ReactiveRuleFilterOrArray([('O', 1), ('O', 2)])],
            'N': [AtomRuleBasedFilter.ReactiveRuleFilterAndArray([('C', 1)])],
            'O': True
        }
        # Setup filter and check
        f = AtomRuleBasedFilter(rules=rules)
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(27)))
        reference = [
            1,   # C
            2,   # C
            3,   # O
            4,   # N
            26,  # O (water)
        ]
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in reference

        acetal_like_group_d2 = AtomRuleBasedFilter.FunctionalGroupRule(distance=2, hetero_atoms=['O', 'N'], n_bonds=4,
                                                                       n_hetero_atoms=2)
        acetal_like_group_d1 = AtomRuleBasedFilter.FunctionalGroupRule(distance=1, hetero_atoms=['O', 'N'], n_bonds=4,
                                                                       n_hetero_atoms=2)
        rules = {
            'H': [acetal_like_group_d2],
            'C': [AtomRuleBasedFilter.ReactiveRuleFilterOrArray([('O', 1), ('O', 2)]), acetal_like_group_d1],
            'N': [AtomRuleBasedFilter.ReactiveRuleFilterAndArray([('C', 1)])],
            'O': [AtomRuleBasedFilter.ReactiveRuleFilterAndArray([('C', 1)])]
        }
        # Setup filter and check
        f = AtomRuleBasedFilter(rules=rules)
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(27)))
        reference = [
            1,   # C
            2,   # C
            3,   # O
            4,   # N
            12,  # H
            13,  # H
            15   # H
        ]
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in reference
        # Cleaning
        manager.wipe()

    def test_element_wise_reaction_coordinate_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_element_wise_reaction_coordinate_filter")

        # Get collections
        structures = manager.get_collection("structures")

        # Add structure data
        rr = resources_root_path()
        proline_propanal_adduct = json.load(open(os.path.join(rr, "proline_propanal_adduct.json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, "proline_propanal_adduct.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", proline_propanal_adduct["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", proline_propanal_adduct["masm_idx_map"])
        structure.set_compound(db.ID())

        water = json.load(open(os.path.join(rr, "water.json"), "r"))
        structure1 = db.Structure()
        structure1.link(structures)
        structure1.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure1.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure1.set_graph("masm_cbor_graph", water["masm_cbor_graph"])
        structure1.set_graph("masm_idx_map", water["masm_idx_map"])
        structure1.set_compound(db.ID())

        # trivial checks
        rules = dict()
        f = ElementWiseReactionCoordinateFilter(rules=rules)
        coords = list()
        for i in range(27):
            for j in range(i):
                coords.append(((i, j),))
        allowed_coordinates = f.filter_reaction_coordinates([structure, structure1], coords)
        assert len(allowed_coordinates) == 27 * (27 - 1) / 2

        rules = {
            'H': ['H', 'C', 'N', 'O'],
            'C': ['H', 'C', 'N', 'O'],
            'N': ['H'],
            'O': ['H']
        }
        f = ElementWiseReactionCoordinateFilter(rules=rules)
        allowed_coordinates = f.filter_reaction_coordinates([structure, structure1], coords)
        reference = [((3, 4),), ((3, 26),), ((4, 26),)]
        assert len(allowed_coordinates) == len(reference)
        for ref in reference:
            assert ref in reference

        # Cleaning
        manager.wipe()

    def test_heuristic_polarization_coordinate_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_polarization_coordinate_filter")

        # Get collections
        structures = manager.get_collection("structures")

        # Add structure data
        rr = resources_root_path()
        proline_propanal_adduct = json.load(open(os.path.join(rr, "proline_propanal_adduct.json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, "proline_propanal_adduct.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", proline_propanal_adduct["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", proline_propanal_adduct["masm_idx_map"])
        structure.set_compound(db.ID())

        water = json.load(open(os.path.join(rr, "water.json"), "r"))
        structure1 = db.Structure()
        structure1.link(structures)
        structure1.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure1.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure1.set_graph("masm_cbor_graph", water["masm_cbor_graph"])
        structure1.set_graph("masm_idx_map", water["masm_idx_map"])
        structure1.set_compound(db.ID())

        # trivial checks
        coords = list()
        for i in range(27):
            for j in range(i):
                coords.append(((i, j),))
        rules = {
            'H': [],
            'C': [],
            'N': [],
            'O': []
        }
        f = HeuristicPolarizationReactionCoordinateFilter(rules=rules)
        allowed_coordinates = f.filter_reaction_coordinates([structure, structure1], coords)
        assert len(allowed_coordinates) == 0

        rules = {
            'H': [HeuristicPolarizationReactionCoordinateFilter.PaulingElectronegativityRule()],
            'C': [HeuristicPolarizationReactionCoordinateFilter.PaulingElectronegativityRule()],
            'N': [HeuristicPolarizationReactionCoordinateFilter.PaulingElectronegativityRule()],
            'O': [HeuristicPolarizationReactionCoordinateFilter.PaulingElectronegativityRule()]
        }
        f = HeuristicPolarizationReactionCoordinateFilter(rules=rules)
        allowed_coordinates = f.filter_reaction_coordinates([structure, structure1], coords)
        reference = [((3, 2),), ((4, 2),), ((5, 3),), ((5, 4),), ((8, 3),), ((8, 4),), ((15, 3),), ((15, 4),),
                     ((24, 3),), ((24, 4),), ((25, 3),), ((25, 4),), ((26, 2),), ((26, 5),), ((26, 8),), ((26, 15),),
                     ((26, 24),), ((26, 25),)]
        assert len(allowed_coordinates) == len(reference)
        for ref in reference:
            assert ref in allowed_coordinates

        rules = {
            'H': [HeuristicPolarizationReactionCoordinateFilter.PaulingElectronegativityRule(),
                  HeuristicPolarizationReactionCoordinateFilter.FunctionalGroupRule("+", 1, ['N', 'O'], 'C', 4, 1),
                  HeuristicPolarizationReactionCoordinateFilter.FunctionalGroupRule("+", 1, ['N', 'O'], 'C', 4, 2)],
            'C': [HeuristicPolarizationReactionCoordinateFilter.PaulingElectronegativityRule()],
            'N': [HeuristicPolarizationReactionCoordinateFilter.PaulingElectronegativityRule()],
            'O': [HeuristicPolarizationReactionCoordinateFilter.PaulingElectronegativityRule()]
        }
        f = HeuristicPolarizationReactionCoordinateFilter(rules=rules)
        allowed_coordinates = f.filter_reaction_coordinates([structure, structure1], coords)
        reference = [((3, 2),), ((4, 2),), ((5, 3),), ((5, 4),), ((8, 3),), ((8, 4),), ((14, 3),), ((14, 4),),
                     ((15, 3),), ((15, 4),), ((16, 3),), ((16, 4),), ((17, 3),), ((17, 4),), ((22, 3),), ((22, 4),),
                     ((23, 3),), ((23, 4),), ((24, 3),), ((24, 4),), ((25, 3),), ((25, 4),), ((26, 2),), ((26, 5),),
                     ((26, 8),), ((26, 14),), ((26, 15),), ((26, 16),), ((26, 17),), ((26, 22),), ((26, 23),),
                     ((26, 24),), ((26, 25),)]
        assert len(allowed_coordinates) == len(reference)
        for ref in reference:
            assert ref in allowed_coordinates

        rules = {
            'H': [HeuristicPolarizationReactionCoordinateFilter.PaulingElectronegativityRule(),
                  HeuristicPolarizationReactionCoordinateFilter.FunctionalGroupRule("+", 2, ['N', 'O'], 'C', 3, 1)],
            'C': [HeuristicPolarizationReactionCoordinateFilter.PaulingElectronegativityRule()],
            'N': [HeuristicPolarizationReactionCoordinateFilter.PaulingElectronegativityRule()],
            'O': [HeuristicPolarizationReactionCoordinateFilter.PaulingElectronegativityRule()]
        }
        f = HeuristicPolarizationReactionCoordinateFilter(rules=rules)
        allowed_coordinates = f.filter_reaction_coordinates([structure, structure1], coords)
        reference = [((3, 2),), ((4, 2),), ((5, 3),), ((5, 4),), ((8, 3),), ((8, 4),), ((15, 3),), ((15, 4),),
                     ((24, 3),), ((24, 4),), ((25, 3),), ((25, 4),), ((26, 2),), ((26, 5),), ((26, 8),), ((26, 15),),
                     ((26, 24),), ((26, 25),)]
        assert len(allowed_coordinates) == len(reference)
        for ref in reference:
            assert ref in allowed_coordinates
        # Cleaning
        manager.wipe()

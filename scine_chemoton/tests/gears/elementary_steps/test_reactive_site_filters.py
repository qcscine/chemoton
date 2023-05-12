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
    HeuristicPolarizationReactionCoordinateFilter,
)
from ....gears.elementary_steps.reaction_rules.distance_rules import (
    DistanceRuleAndArray,
    DistanceRuleOrArray,
    SimpleDistanceRule,
    FunctionalGroupRule,
)
from ....gears.elementary_steps.reaction_rules.polarization_rules import (
    PolarizationRuleAndArray,
    PaulingElectronegativityRule,
    PolarizationFunctionalGroupRule
)
from ....gears.elementary_steps.aggregate_filters import AggregateFilter

# we are using star imports here, because we test `eval(repr(cls))`, which requires to know all classes
# in these submodules
from scine_chemoton.gears.elementary_steps. \
    reaction_rules.element_rules import *  # pylint: disable=(wildcard-import,unused-wildcard-import)  # noqa
from scine_chemoton.gears.elementary_steps. \
    reaction_rules.distance_rules import *  # pylint: disable=(wildcard-import,unused-wildcard-import)  # noqa
from scine_chemoton.gears.elementary_steps. \
    reaction_rules.polarization_rules import *  # pylint: disable=(wildcard-import,unused-wildcard-import)  # noqa


class ReactiveSiteFiltersTests(unittest.TestCase):

    def setUp(self) -> None:
        manager = db_setup.get_clean_db("chemoton_test_reactive_site_filters")
        structures = manager.get_collection("structures")
        model = db_setup.get_fake_model()
        rr = resources_root_path()
        cyclohexene = json.load(open(os.path.join(rr, "cyclohexene.json"), "r"))
        structure = db.Structure(db.ID(), structures)
        structure.create(os.path.join(rr, "cyclohexene.xyz"), 0, 1, model, db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", cyclohexene["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", cyclohexene["masm_idx_map"])
        structure.set_aggregate(db.ID())

        water = json.load(open(os.path.join(rr, "water.json"), "r"))
        structure1 = db.Structure(db.ID(), structures)
        structure1.create(os.path.join(rr, "water.xyz"), 0, 1, model, db.Label.MINIMUM_OPTIMIZED)
        structure1.set_graph("masm_cbor_graph", water["masm_cbor_graph"])
        structure1.set_graph("masm_idx_map", water["masm_idx_map"])
        structure1.set_aggregate(db.ID())

        adduct = json.load(open(os.path.join(rr, "proline_propanal_adduct.json"), "r"))
        structure2 = db.Structure(db.ID(), structures)
        structure2.create(os.path.join(rr, "proline_propanal_adduct.xyz"), 0, 1, model, db.Label.MINIMUM_OPTIMIZED)
        structure2.set_graph("masm_cbor_graph", adduct["masm_cbor_graph"])
        structure2.set_graph("masm_idx_map", adduct["masm_idx_map"])
        structure2.set_aggregate(db.ID())

        self.manager = manager
        self.structure = structure
        self.structure1 = structure1
        self.structure2 = structure2

    def tearDown(self) -> None:
        self.manager.wipe()

    def test_default_filter(self):
        f = ReactiveSiteFilter()

        # Filter and check
        allowed_atoms = f.filter_atoms([self.structure, self.structure], list(range(16)))
        assert len(allowed_atoms) == 16

        allowed_pairs = f.filter_atom_pairs([self.structure], list(combinations(range(16), 2)))
        assert len(allowed_pairs) == 120

        # Just two random coordinates
        allowed_coordinates = f.filter_reaction_coordinates([self.structure], [((0, 1), (1, 2)), ((0, 1), (3, 4))])
        assert len(allowed_coordinates) == 2

        assert [] == f.filter_atoms([self.structure], [])
        assert [] == f.filter_atom_pairs([self.structure], [])
        assert [] == f.filter_reaction_coordinates([self.structure], [])

    def test_filter_chain_failure_logic_and(self):
        with self.assertRaises(TypeError) as context:
            _ = (ReactiveSiteFilter() & AggregateFilter())
        self.assertTrue('ReactiveSiteFilter' in str(context.exception))

    def test_filter_chain_failure_logic_or(self):
        with self.assertRaises(TypeError) as context:
            _ = (ReactiveSiteFilter() | AggregateFilter())
        self.assertTrue('ReactiveSiteFilter' in str(context.exception))

    def test_filter_chain_failure_class_and(self):
        with self.assertRaises(TypeError) as context:
            ReactiveSiteFilterAndArray([ReactiveSiteFilter(), AggregateFilter()])
        self.assertTrue('ReactiveSiteFilterAndArray' in str(context.exception))

    def test_filter_chain_failure_class_or(self):
        with self.assertRaises(TypeError) as context:
            ReactiveSiteFilterOrArray([ReactiveSiteFilter(), AggregateFilter()])
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
        noneFilter = ReactiveSiteFilterAndArray()

        # Filter and check pruning no atoms
        allowed_atoms = noneFilter.filter_atoms([self.structure], list(range(16)))
        assert len(allowed_atoms) == 16
        pairs = [(0, 1), (2, 3), (13, 14)]
        allowed_pairs = noneFilter.filter_atom_pairs([self.structure, self.structure1], pairs)
        assert len(allowed_pairs) == 3
        coords = [((0, 1),), ((2, 3), (4, 5))]
        allowed_coordinates = noneFilter.filter_reaction_coordinates([self.structure], coords)
        assert len(allowed_coordinates) == 2
        assert [] == noneFilter.filter_atoms([self.structure], [])
        assert [] == noneFilter.filter_atom_pairs([self.structure], [])
        assert [] == noneFilter.filter_reaction_coordinates([self.structure], [])

    def test_empty_or(self):
        noneFilter = ReactiveSiteFilterOrArray()

        # Filter and check pruning of everything
        allowed_atoms = noneFilter.filter_atoms([self.structure], list(range(16)))
        assert len(allowed_atoms) == 0
        pairs = [(0, 1), (2, 3), (13, 14)]
        allowed_pairs = noneFilter.filter_atom_pairs([self.structure, self.structure1], pairs)
        assert len(allowed_pairs) == 0
        coords = [((0, 1),), ((2, 3), (4, 5))]
        allowed_coordinates = noneFilter.filter_reaction_coordinates([self.structure], coords)
        assert len(allowed_coordinates) == 0
        assert [] == noneFilter.filter_atoms([self.structure], [])
        assert [] == noneFilter.filter_atom_pairs([self.structure], [])
        assert [] == noneFilter.filter_reaction_coordinates([self.structure], [])

    def test_masm_ranking_filter(self):
        noneFilter = MasmChemicalRankingFilter(prune="None")
        hFilter = MasmChemicalRankingFilter(prune="Hydrogen")
        allFilter = MasmChemicalRankingFilter(prune="All")

        # Filter and check pruning no atoms
        allowed_atoms = noneFilter.filter_atoms([self.structure], list(range(16)))
        assert len(allowed_atoms) == 16
        pairs = [(0, 1), (2, 3), (13, 14)]
        allowed_pairs = noneFilter.filter_atom_pairs([self.structure, self.structure1], pairs)
        assert len(allowed_pairs) == 3
        coords = [((0, 1),), ((2, 3), (4, 5))]
        allowed_coordinates = noneFilter.filter_reaction_coordinates([self.structure], coords)
        assert len(allowed_coordinates) == 2
        assert [] == noneFilter.filter_atoms([self.structure], [])
        assert [] == noneFilter.filter_atom_pairs([self.structure], [])
        assert [] == noneFilter.filter_reaction_coordinates([self.structure], [])

        # Check filter with pruning hydrogens
        allowed_atoms = hFilter.filter_atoms([self.structure, self.structure1], list(range(19)))
        assert len(allowed_atoms) == 14
        # Pair and coordinate filter are taken over from base class i.e. they do not filter out anything
        allowed_pairs = hFilter.filter_atom_pairs([self.structure], list(combinations(range(16), 2)))
        assert len(allowed_pairs) == 120
        coords = [((2, 3), (4, 5)), ((0, 17),)]
        allowed_coordinates = hFilter.filter_reaction_coordinates([self.structure, self.structure1], coords)
        assert len(allowed_coordinates) == 2
        assert [] == hFilter.filter_atoms([self.structure], [])
        assert [] == hFilter.filter_atom_pairs([self.structure], [])
        assert [] == hFilter.filter_reaction_coordinates([self.structure], [])

        # Check filter with pruning all
        allowed_atoms = allFilter.filter_atoms([self.structure, self.structure1], [i for i in range(19)])
        assert len(allowed_atoms) == 14
        bonds = [(2, 3), (4, 7)]
        # Pair and coordinate filter are taken over from base class i.e. they do not filter out anything
        allowed_pairs = allFilter.filter_atom_pairs(self.structure, bonds)
        assert allowed_pairs == bonds
        coords = [((3, 4),), ((7, 8),), ((0, 2))]
        allowed_coordinates = allFilter.filter_reaction_coordinates(self.structure, coords)
        assert len(allowed_coordinates) == 3
        assert [] == allFilter.filter_atoms([self.structure], [])
        assert [] == allFilter.filter_atom_pairs([self.structure], [])
        assert [] == allFilter.filter_reaction_coordinates([self.structure], [])

    def test_simple_ranking_filter(self):
        f = SimpleRankingFilter(atom_threshold=1, pair_threshold=5, coordinate_threshold=4)

        # Filter and check
        allowed_atoms = f.filter_atoms([self.structure, self.structure1], list(range(19)))
        # C and O and H in H2O
        assert len(allowed_atoms) == 9

        allowed_pairs = f.filter_atom_pairs([self.structure], list(combinations(range(16), 2)))
        # All C-C combinations
        assert len(allowed_pairs) == 15

        # All intermolecular coords with one atom per molecule
        coords = list((inter_pair,) for inter_pair in product(range(16), range(16, 19)))
        allowed_coordinates = f.filter_reaction_coordinates([self.structure, self.structure1], coords)
        # All O-C and H-C with H attached to O coordinates should pass the threshold
        assert len(allowed_coordinates) == 18

        # Every combination twice per coord effectively inducing the ranking value to be doubled
        double_coords = [coord + coord for coord in coords]
        allowed_coordinates = f.filter_reaction_coordinates([self.structure, self.structure1], double_coords)
        # With double counting all intermolecular coords should survive
        assert len(allowed_coordinates) == 48

        assert [] == f.filter_atoms([self.structure], [])
        assert [] == f.filter_atom_pairs([self.structure], [])
        assert [] == f.filter_reaction_coordinates([self.structure], [])

    def test_filter_or_array(self):
        f = SimpleRankingFilter(
            atom_threshold=1,
            pair_threshold=0,
            coordinate_threshold=999) | MasmChemicalRankingFilter(
            prune="Hydrogen")

        # Filter and check
        allowed_atoms = f.filter_atoms([self.structure], list(range(16)))
        # Hydrogens do not survive SimpleRankingFilter but all but 4 survive MasmChemicalRankingFilter
        assert len(allowed_atoms) == 12

        allowed_pairs = f.filter_atom_pairs([self.structure], list(combinations(range(16), 2)))
        # All 16 choose 2 combinations survive SimpleRankingFilter
        assert len(allowed_pairs) == 120

        coords = (((0, 1),), ((0, 17), (1, 3)), ((2, 3),))
        allowed_coordinates = f.filter_reaction_coordinates([self.structure, self.structure1], coords)
        # MasmChemicalRanking filter_reaction_coordinates is base method so all coordinates should survive
        assert len(allowed_coordinates) == 3

        assert [] == f.filter_atoms([self.structure], [])
        assert [] == f.filter_atom_pairs([self.structure], [])
        assert [] == f.filter_reaction_coordinates([self.structure], [])

    def test_filter_and_array(self):
        f = SimpleRankingFilter(atom_threshold=1, pair_threshold=5, coordinate_threshold=4) & MasmChemicalRankingFilter(
            prune="Hydrogen"
        )

        # Filter and check
        allowed_atoms = f.filter_atoms([self.structure], list(range(16)))
        assert len(allowed_atoms) == 6

        allowed_pairs = f.filter_atom_pairs([self.structure], list(combinations(range(16), 2)))
        # All pairs pass MasmChemicalRanking, only C-C SimpleRanking
        assert len(allowed_pairs) == 15

        coords = list((inter_pair,) for inter_pair in product(range(16), range(16, 19)))
        allowed_coordinates = f.filter_reaction_coordinates([self.structure, self.structure1], coords)
        assert len(allowed_coordinates) == 18

        assert [] == f.filter_atoms([self.structure], [])
        assert [] == f.filter_atom_pairs([self.structure], [])
        assert [] == f.filter_reaction_coordinates([self.structure], [])

    def test_atom_rule_based_filter(self):
        # trivial checks
        rules = {
            'H': True,
            'C': True,
            'N': True,
            'O': True
        }
        f = AtomRuleBasedFilter(rules=rules)
        allowed_atoms = f.filter_atoms([self.structure2, self.structure1], list(range(27)))
        reference = range(27)
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        rules = {
            'H': False,
            'C': False,
            'N': True,
            'O': False
        }
        f = AtomRuleBasedFilter(rules=rules)
        assert f._rules
        assert eval(repr(f._rules))  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([self.structure2, self.structure1], list(range(27)))
        reference = [4]
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        rules = {
            'C': DistanceRuleAndArray([SimpleDistanceRule('C', 1),
                                       SimpleDistanceRule('O', 1),
                                       SimpleDistanceRule('N', 1)])
        }
        f = AtomRuleBasedFilter(rules=rules)
        assert f._rules
        assert eval(repr(f._rules))  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([self.structure2, self.structure1], list(range(27)))
        reference = [2]
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        rules = {
            'H': False,
            'C': [DistanceRuleOrArray([SimpleDistanceRule('O', 1),
                                       SimpleDistanceRule('O', 2)])],
            'N': [DistanceRuleAndArray([SimpleDistanceRule('C', 1)])],
            'O': True
        }
        # Setup filter and check
        f = AtomRuleBasedFilter(rules=rules)
        assert f._rules
        assert eval(repr(f._rules))  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([self.structure2, self.structure1], list(range(27)))
        reference = [
            1,   # C
            2,   # C
            3,   # O
            4,   # N
            26,  # O (water)
        ]
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        acetal_like_group_d2 = FunctionalGroupRule(distance=2, n_bonds=(4, 4), central_atom="C",
                                                   specified_bond_partners={'O': 1, 'N': 1}, strict_counts=True)
        acetal_like_group_d1 = FunctionalGroupRule(distance=1, n_bonds=(4, 4), central_atom="C",
                                                   specified_bond_partners={'O': 1, 'N': 1}, strict_counts=True)

        rules = {
            'H': acetal_like_group_d2,
            'C': DistanceRuleOrArray([DistanceRuleOrArray([SimpleDistanceRule('O', 1),
                                                           SimpleDistanceRule('O', 2)]),
                                      acetal_like_group_d1]),
            'N': SimpleDistanceRule('C', 1),
            'O': SimpleDistanceRule('C', 1)
        }
        # Setup filter and check
        f = AtomRuleBasedFilter(rules=rules)
        assert f._rules
        assert eval(repr(f._rules))  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([self.structure2, self.structure1], list(range(27)))
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
            assert ref in allowed_atoms

    def test_atom_rule_based_filter_exclude_mode(self):
        # trivial checks
        rules = {
            'H': True,
            'C': True,
            'N': True,
            'O': True
        }
        f = AtomRuleBasedFilter(rules=rules, exclude_mode=True)
        allowed_atoms = f.filter_atoms([self.structure2, self.structure1], list(range(27)))
        reference = range(0)
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        rules = {
            'H': True,
            'C': True,
            'N': False,
            'O': True
        }
        f = AtomRuleBasedFilter(rules=rules, exclude_mode=True)
        assert f._rules
        assert eval(repr(f._rules))  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([self.structure2, self.structure1], list(range(27)))
        reference = [4]
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        rules = {
            'C': DistanceRuleOrArray([SimpleDistanceRule('O', 1),
                                      SimpleDistanceRule('N', 1)])
        }
        f = AtomRuleBasedFilter(rules=rules, exclude_mode=True)
        assert f._rules
        assert eval(repr(f._rules))  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([self.structure2, self.structure1], list(range(27)))
        reference = [0, 1, 6, 7]
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        rules = {
            'H': True,
            'C': DistanceRuleOrArray([SimpleDistanceRule('N', 2),
                                      SimpleDistanceRule('N', 1)]),
            'N': DistanceRuleAndArray([SimpleDistanceRule('C', 1)]),
            'O': False
        }
        # Setup filter and check
        f = AtomRuleBasedFilter(rules=rules, exclude_mode=True)
        assert f._rules
        assert eval(repr(f._rules))  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([self.structure2, self.structure1], list(range(27)))
        reference = [
            0,  # C
            3,  # O
            26,  # O (water)
        ]
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

    def test_atom_based_rule_filter_2(self):
        # Get collections
        structures = self.manager.get_collection("structures")

        # Add structure data
        rr = resources_root_path()
        ni_cat = json.load(open(os.path.join(rr, "ni_catalyst.json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, "ni_catalyst.xyz"), 0, 2)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", ni_cat["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", ni_cat["masm_idx_map"])
        structure.set_compound(db.ID())

        substrate = json.load(open(os.path.join(rr, "substrate.json"), "r"))
        structure1 = db.Structure()
        structure1.link(structures)
        structure1.create(os.path.join(rr, "substrate.xyz"), 0, 1)
        structure1.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure1.set_graph("masm_cbor_graph", substrate["masm_cbor_graph"])
        structure1.set_graph("masm_idx_map", substrate["masm_idx_map"])
        structure1.set_compound(db.ID())

        rules = {
            'Ni': True,
            'C': DistanceRuleOrArray([SimpleDistanceRule('O', 2),
                                      SimpleDistanceRule('O', 3)]),
            'P': SimpleDistanceRule('Ni', 1),
        }
        f = AtomRuleBasedFilter(rules=rules)
        assert f._rules
        assert eval(repr(f._rules))  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(62)))
        reference = [
            14,  # P
            15,  # Ni
            16,  # P
            59,  # C 2 bonds away
            55,  # C 2 bonds away
            58,  # C 3 bonds away
            61,  # C 3 bonds away
        ]
        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

    def test_element_wise_reaction_coordinate_filter(self):
        # trivial checks
        rules = dict()
        f = ElementWiseReactionCoordinateFilter(rules=rules)
        pairs = list()
        for i in range(27):
            for j in range(i):
                pairs.append((i, j))
        allowed_pairs = f.filter_atom_pairs([self.structure2, self.structure1], pairs)
        assert len(allowed_pairs) == 27 * (27 - 1) / 2

        rules = {
            'H': ['H', 'C', 'N', 'O'],
            'C': ['H', 'C', 'N', 'O'],
            'N': ['H'],
            'O': ['H']
        }
        f = ElementWiseReactionCoordinateFilter(rules=rules)
        assert f._rules
        assert eval(repr(f._rules))  # pylint: disable=eval-used
        allowed_pairs = f.filter_atom_pairs([self.structure2, self.structure1], pairs)
        reference = [(4, 3), (26, 3), (26, 4)]
        assert len(allowed_pairs) == len(reference)
        print(allowed_pairs)
        for ref in reference:
            assert ref in allowed_pairs

    def test_heuristic_polarization_coordinate_filter(self):
        # trivial checks
        pairs = list()
        for i in range(27):
            for j in range(i):
                pairs.append((i, j))
        rules = {
            'H': [],
            'C': [],
            'N': [],
            'O': []
        }
        f = HeuristicPolarizationReactionCoordinateFilter(rules=rules)
        assert f._rules
        assert eval(repr(f._rules))  # pylint: disable=eval-used
        allowed_pairs = f.filter_atom_pairs([self.structure2, self.structure1], pairs)
        assert len(allowed_pairs) == 0

        rules = {
            'H': PaulingElectronegativityRule(),
            'C': PaulingElectronegativityRule(),
            'N': PaulingElectronegativityRule(),
            'O': PaulingElectronegativityRule()
        }
        f = HeuristicPolarizationReactionCoordinateFilter(rules=rules)
        assert f._rules
        assert eval(repr(f._rules))  # pylint: disable=eval-used
        allowed_pairs = f.filter_atom_pairs([self.structure2, self.structure1], pairs)
        reference = [(3, 2), (4, 2), (5, 3), (5, 4), (8, 3), (8, 4), (15, 3), (15, 4), (24, 3), (24, 4), (25, 3),
                     (25, 4), (26, 2), (26, 5), (26, 8), (26, 15), (26, 24), (26, 25)]
        assert len(allowed_pairs) == len(reference)
        for ref in reference:
            assert ref in allowed_pairs

        rules = {
            'H': PolarizationRuleAndArray([PaulingElectronegativityRule(),
                                           PolarizationFunctionalGroupRule('+', DistanceRuleOrArray([
                                               FunctionalGroupRule(1, 'C', (4, 4), {'N': 1}),
                                               FunctionalGroupRule(1, 'C', (4, 4), {'O': 1})]))]),
            'C': PaulingElectronegativityRule(),
            'N': PaulingElectronegativityRule(),
            'O': PaulingElectronegativityRule()
        }
        f = HeuristicPolarizationReactionCoordinateFilter(rules=rules)
        assert f._rules
        assert eval(repr(f._rules))  # pylint: disable=eval-used
        allowed_pairs = f.filter_atom_pairs([self.structure2, self.structure1], pairs)
        reference = [(3, 2), (4, 2), (5, 3), (5, 4), (8, 3), (8, 4), (14, 3), (14, 4),
                     (15, 3), (15, 4), (16, 3), (16, 4), (17, 3), (17, 4), (22, 3), (22, 4),
                     (23, 3), (23, 4), (24, 3), (24, 4), (25, 3), (25, 4), (26, 2), (26, 5),
                     (26, 8), (26, 14), (26, 15), (26, 16), (26, 17), (26, 22), (26, 23),
                     (26, 24), (26, 25)]
        assert len(allowed_pairs) == len(reference)
        for ref in reference:
            assert ref in allowed_pairs

        rules = {
            'H': PolarizationRuleAndArray([
                PaulingElectronegativityRule(),
                PolarizationFunctionalGroupRule("+", FunctionalGroupRule(2, 'C', (3, 3), {'N': 1, 'O': 1}))
            ]),
            'C': PaulingElectronegativityRule(),
            'N': PaulingElectronegativityRule(),
            'O': PaulingElectronegativityRule()
        }
        f = HeuristicPolarizationReactionCoordinateFilter(rules=rules)
        assert f._rules
        assert eval(repr(f._rules))  # pylint: disable=eval-used
        allowed_pairs = f.filter_atom_pairs([self.structure2, self.structure1], pairs)
        reference = [(3, 2), (4, 2), (5, 3), (5, 4), (8, 3), (8, 4), (15, 3), (15, 4), (24, 3), (24, 4), (25, 3),
                     (25, 4), (26, 2), (26, 5), (26, 8), (26, 15), (26, 24), (26, 25)]
        assert len(allowed_pairs) == len(reference)
        for ref in reference:
            assert ref in allowed_pairs

    def test_combination_filter(self):
        structures = self.manager.get_collection("structures")
        # Add structure data
        rr = resources_root_path()
        zwitter_ion = json.load(open(os.path.join(rr, "zwitter_ion.json"), "r"))
        zwitter_ion_structure = db.Structure()
        zwitter_ion_structure.link(structures)
        zwitter_ion_structure.create(os.path.join(rr, "zwitter_ion.xyz"), 0, 1)
        zwitter_ion_structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        zwitter_ion_structure.set_graph("masm_cbor_graph", zwitter_ion["masm_cbor_graph"])
        zwitter_ion_structure.set_graph("masm_idx_map", zwitter_ion["masm_idx_map"])
        zwitter_ion_structure.set_compound(db.ID())

        # 4.4) Add filters
        carb_H_2_pos = FunctionalGroupRule(2, 'C', (3, 3), {'N': 1}, strict_counts=True)
        carb_C = FunctionalGroupRule(0, 'C', (3, 3), {'O': 1, 'N': 1}, strict_counts=True)
        olefine_C = FunctionalGroupRule(0, 'C', (3, 3), {'H': 1, 'C': 1}, strict_counts=False)
        carb_C_1_pos = FunctionalGroupRule(1, 'C', (3, 3), {'O': 1, 'N': 1}, strict_counts=False)
        carb_acetal = FunctionalGroupRule(0, 'C', (4, 4), {'O': 1, 'N': 1}, strict_counts=True)
        ammonium = FunctionalGroupRule(0, 'N', (4, 4), {'H': 0, 'C': 0, 'N': 0, 'O': 0}, strict_counts=False)
        any_N = SimpleDistanceRule('N', 0)

        reactive_site_rules = {
            'H': DistanceRuleOrArray([DistanceRuleOrArray([SimpleDistanceRule('N', 1),
                                                           SimpleDistanceRule('O', 1)]),
                                      carb_H_2_pos]),
            'C': DistanceRuleOrArray([carb_C, carb_C_1_pos, olefine_C, carb_acetal]),
            'O': DistanceRuleOrArray([SimpleDistanceRule('C', 1),
                                      SimpleDistanceRule('H', 1)]),
            'N': DistanceRuleOrArray([SimpleDistanceRule('C', 1),
                                      SimpleDistanceRule('H', 1)])
        }

        pol_carb_H_2_pos = PolarizationFunctionalGroupRule('+', carb_H_2_pos)
        pol_carb_acid_H_2_pos = PolarizationFunctionalGroupRule('+',
                                                                FunctionalGroupRule(
                                                                    2, 'C', (3, 3), {'O': 1, 'N': 1}, False)
                                                                )
        pol_carb_C_1_pos = PolarizationFunctionalGroupRule('+', carb_C)
        pol_carb_acid_C_1_pos = PolarizationFunctionalGroupRule('+',
                                                                FunctionalGroupRule(
                                                                    1, 'C', (3, 3), {'O': 1, 'N': 1}, False)
                                                                )
        pol_olefine_C = PolarizationFunctionalGroupRule('+-', olefine_C)
        pol_ammonium = PolarizationFunctionalGroupRule(
            '+', DistanceRuleAndArray([ammonium, any_N]))
        amin = PolarizationFunctionalGroupRule('-',
                                               FunctionalGroupRule(
                                                   0, 'N', (3, 3), {'H': 0, 'C': 0}, False)
                                               )

        reactive_coordinate_pol_rules = {
            'H': PolarizationRuleAndArray([PaulingElectronegativityRule(),
                                           pol_carb_H_2_pos, pol_carb_acid_H_2_pos]),
            'C': PolarizationRuleAndArray([PaulingElectronegativityRule(),
                                           pol_carb_C_1_pos, pol_carb_acid_C_1_pos, pol_olefine_C]),
            'O': PaulingElectronegativityRule(),
            'N': PolarizationRuleAndArray([amin, pol_ammonium])
        }
        reaction_coordinate_rules = {
            'H': ['H', 'C'],
            'O': ['O', 'N'],
            'N': ['N']
        }
        f = ReactiveSiteFilterAndArray([AtomRuleBasedFilter(reactive_site_rules),
                                        HeuristicPolarizationReactionCoordinateFilter(reactive_coordinate_pol_rules),
                                        ElementWiseReactionCoordinateFilter(reaction_coordinate_rules, False)])
        assert all(filt._rules for filt in f)
        assert all(eval(repr(filt._rules)) for filt in f)  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([self.structure2], list(range(24)))
        reference = [2, 3, 4, 15]
        assert len(allowed_atoms) == len(reference)
        for atom in allowed_atoms:
            assert atom in reference

        pairs = list()
        for i in range(len(allowed_atoms)):
            for j in range(i):
                pairs.append((allowed_atoms[i], allowed_atoms[j]))

        reference_pairs = [(3, 2), (4, 2), (15, 3), (15, 4)]
        allowed_pairs = f.filter_atom_pairs([self.structure2], pairs)
        assert len(reference_pairs) == len(allowed_pairs)
        for pair in allowed_pairs:
            assert pair in reference_pairs

        allowed_atoms2 = f.filter_atoms([zwitter_ion_structure], list(range(27)))
        reference = [3, 9, 14, 15, 16, 17, 20]
        assert len(allowed_atoms2) == len(reference)
        for atom in allowed_atoms2:
            assert atom in reference
        pairs = list()
        for i in range(len(allowed_atoms2)):
            for j in range(i):
                pairs.append((allowed_atoms2[i], allowed_atoms2[j]))

        reference_pairs = [(14, 9), (15, 9), (16, 14), (16, 15), (17, 14), (17, 15), (20, 9), (20, 16), (20, 17)]
        allowed_pairs = f.filter_atom_pairs([zwitter_ion_structure], pairs)
        assert len(reference_pairs) == len(allowed_pairs)
        for pair in allowed_pairs:
            assert pair in reference_pairs

        pairs = list()
        for i in range(len(allowed_atoms)):
            for j in range(len(allowed_atoms2)):
                pairs.append((allowed_atoms[i], 24 + allowed_atoms2[j]))
        f = ReactiveSiteFilterAndArray([
            AtomRuleBasedFilter(reactive_site_rules),
            HeuristicPolarizationReactionCoordinateFilter(reactive_coordinate_pol_rules),
            ElementWiseReactionCoordinateFilter(reaction_coordinate_rules, False)
        ])
        assert all(filt._rules for filt in f)
        assert all(eval(repr(filt._rules)) for filt in f)  # pylint: disable=eval-used
        allowed_pairs = f.filter_atom_pairs([self.structure2, zwitter_ion_structure], pairs)
        reference_pairs = [(2, 38), (2, 39), (2, 44), (3, 33), (3, 40), (3, 41), (4, 33), (4, 40), (4, 41), (15, 38),
                           (15, 39), (15, 44)]
        assert len(reference_pairs) == len(allowed_pairs)
        for pair in allowed_pairs:
            assert pair in reference_pairs

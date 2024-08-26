#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
import json
import unittest
from itertools import combinations, product
from typing import List, Tuple

# Third party imports
import scine_database as db
from scine_database import test_database_setup as db_setup

# Local application tests imports
from ...resources import resources_root_path

# Local application imports
from scine_chemoton.filters.reactive_site_filters import (
    MasmChemicalRankingFilter,
    SimpleRankingFilter,
    ReactiveSiteFilter,
    ReactiveSiteFilterOrArray,
    ReactiveSiteFilterAndArray,
    AtomRuleBasedFilter,
    ElementWiseReactionCoordinateFilter,
    HeuristicPolarizationReactionCoordinateFilter,
    CentralSiteFilter,
    AtomPairFunctionalGroupFilter,
    SubStructureFilter,
)
from scine_chemoton.reaction_rules.distance_rules import (
    DistanceRuleAndArray,
    DistanceRuleOrArray,
    SimpleDistanceRule,
    FunctionalGroupRule,
    AlwaysTrue,
)
from scine_chemoton.reaction_rules.polarization_rules import (
    PolarizationRuleAndArray,
    PaulingElectronegativityRule,
    PolarizationFunctionalGroupRule
)
from scine_chemoton.filters.aggregate_filters import AggregateFilter

# we are using star imports here, because we test `eval(repr(cls))`, which requires to know all classes
# in these submodules
from scine_chemoton.reaction_rules.element_rules import *  # pylint: disable=(wildcard-import,unused-wildcard-import)  # noqa
from scine_chemoton.reaction_rules.distance_rules import *  # pylint: disable=(wildcard-import,unused-wildcard-import)  # noqa
from scine_chemoton.reaction_rules.polarization_rules import *  # pylint: disable=(wildcard-import,unused-wildcard-import)  # noqa
from scine_chemoton.reaction_rules.reaction_rule_library import (
    CarbonylX,
    CHOlefinC,
    SpNCX,
    CarboxylX,
    AmidX,
    AcetalX,
    CarboxylH,
    AllylicSp3X,
    AmmoniumX,
    AminX,
    DefaultOrganicChemistry
)


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

        furfuryl_alcohol_graph = json.load(open(os.path.join(rr, "furfuryl_alcohol.json"), "r"))
        furfuryl_alcohol = db.Structure(db.ID(), structures)
        furfuryl_alcohol.create(os.path.join(rr, "furfuryl_alcohol.xyz"), 0, 1, model, db.Label.MINIMUM_OPTIMIZED)
        furfuryl_alcohol.set_graph("masm_cbor_graph", furfuryl_alcohol_graph["masm_cbor_graph"])
        furfuryl_alcohol.set_graph("masm_idx_map", furfuryl_alcohol_graph["masm_idx_map"])
        furfuryl_alcohol.set_aggregate(db.ID())

        amide_acetal_graph = json.load(open(os.path.join(rr, "amide_acetal.json"), "r"))
        amide_acetal = db.Structure(db.ID(), structures)
        amide_acetal.create(os.path.join(rr, "amide_acetal.xyz"), 0, 1, model, db.Label.MINIMUM_OPTIMIZED)
        amide_acetal.set_graph("masm_cbor_graph", amide_acetal_graph["masm_cbor_graph"])
        amide_acetal.set_graph("masm_idx_map", amide_acetal_graph["masm_idx_map"])
        amide_acetal.set_aggregate(db.ID())

        proline_propanal_product_graph = json.load(open(os.path.join(rr, "proline_acid_propanal_product.json"), "r"))
        proline_propanal_product = db.Structure(db.ID(), structures)
        proline_propanal_product.create(os.path.join(rr, "proline_acid_propanal_product.xyz"), 0, 1, model,
                                        db.Label.MINIMUM_OPTIMIZED)
        proline_propanal_product.set_graph("masm_cbor_graph", proline_propanal_product_graph["masm_cbor_graph"])
        proline_propanal_product.set_graph("masm_idx_map", proline_propanal_product_graph["masm_idx_map"])
        proline_propanal_product.set_aggregate(db.ID())

        self.manager = manager
        self.structure = structure
        self.structure1 = structure1
        self.structure2 = structure2
        self.amide_acetal = amide_acetal
        self.furfuryl_alcohol = furfuryl_alcohol
        self.proline_propanal_product = proline_propanal_product

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

    def test_central_site_filter(self):
        rr = resources_root_path()
        grubbs = json.load(open(os.path.join(rr, "grubbs.json"), "r"))
        cat = db.Structure(db.ID(), self.manager.get_collection("structures"))
        cat.create(os.path.join(rr, "grubbs.xyz"), 0, 1, db_setup.get_fake_model(), db.Label.MINIMUM_OPTIMIZED)
        cat.set_graph("masm_cbor_graph", grubbs["masm_cbor_graph"])
        cat.set_graph("masm_idx_map", grubbs["masm_idx_map"])
        cat.set_aggregate(db.ID())

        coordinates = [[(0, 1), (1, 2)], [(0, 2)]]
        f = CentralSiteFilter("Ru", ligand_without_central_atom_reactive=True)
        re_f = CentralSiteFilter("Ru", ligand_without_central_atom_reactive=True,
                                 reevaluate_on_all_levels=True)
        nolig_f = CentralSiteFilter("Ru", ligand_without_central_atom_reactive=False)
        re_nolig_f = CentralSiteFilter("Ru", ligand_without_central_atom_reactive=False,
                                       reevaluate_on_all_levels=True)

        # test empty inputs
        for filt in [f, re_f, nolig_f, re_nolig_f]:
            for structure in [self.structure, self.structure1]:
                assert [] == filt.filter_atoms([structure], [])
                assert [] == filt.filter_atom_pairs([structure], [])
                assert [] == filt.filter_reaction_coordinates([structure], [])

        for structure in [self.structure, self.structure1]:
            n = len(structure.get_atoms())

            for filt in [f, re_f, nolig_f, re_nolig_f]:
                allowed_atoms = filt.filter_atoms([structure], list(range(n)))
                assert len(allowed_atoms) == 0

                pairs: List[Tuple[int, int]] = list(combinations(range(n), 2))
                expected = 0
                allowed_pairs = filt.filter_atom_pairs([self.structure], pairs)
                assert len(allowed_pairs) == expected

                # Just two random coordinates
                expected = 0 if filt.reevaluate_on_all_levels else len(coordinates)
                allowed_coordinates = filt.filter_reaction_coordinates([self.structure], coordinates)
                assert len(allowed_coordinates) == expected

        n_atoms = [len(structure.get_atoms()) for structure in [self.structure, self.structure1, cat]]

        range_0 = list(range(n_atoms[0]))
        range_01 = list(range(n_atoms[0], n_atoms[0] + n_atoms[1]))
        indices = range_0 + range_01
        for filt in [f, re_f, nolig_f, re_nolig_f]:
            allowed_atoms = filt.filter_atoms([self.structure, self.structure1], indices)
            assert len(allowed_atoms) == 0

            pairs: List[Tuple[int, int]] = list(combinations(indices, 2))
            expected = 0
            allowed_pairs = filt.filter_atom_pairs([self.structure, self.structure1], pairs)
            assert len(allowed_pairs) == expected

            coords = list([inter_pair] for inter_pair in product(range_0, range_01))
            expected = 0 if filt.reevaluate_on_all_levels else len(coords)
            allowed_coordinates = filt.filter_reaction_coordinates([self.structure, self.structure1], coords)
            assert len(allowed_coordinates) == expected

        """ catalyst """
        metal_index = 1
        ligand_indices = [0, 2, 3, 11, 34, 91]
        reactive = ligand_indices + [metal_index]

        # unimolecular
        for filt in [f, re_f]:
            allowed_atoms = filt.filter_atoms([cat], list(range(n_atoms[2])))
            assert len(allowed_atoms) == len(reactive)
            assert sorted(reactive) == sorted(allowed_atoms)

            expected = list(combinations(reactive, 2))
            allowed_pairs = filt.filter_atom_pairs([cat], list(combinations(allowed_atoms, 2)))
            assert len(allowed_pairs) == len(expected)
            assert sorted([sorted(pair) for pair in expected]) == sorted([sorted(pair) for pair in allowed_pairs])

            # coordinates are all valid pairs for ligand_without central reactive
            allowed_coordinates = filt.filter_reaction_coordinates([cat], coordinates)
            assert len(allowed_coordinates) == len(coordinates)
            assert sorted(allowed_coordinates) == sorted(coordinates)

        # no lig filter
        for filt in [nolig_f, re_nolig_f]:
            allowed_atoms = filt.filter_atoms([cat], list(range(n_atoms[2])))
            assert len(allowed_atoms) == len(reactive)
            assert sorted(reactive) == sorted(allowed_atoms)

            allowed_pairs = filt.filter_atom_pairs([cat], list(combinations(allowed_atoms, 2)))
            expected = [pair for pair in combinations(reactive, 2) if metal_index in pair]
            assert len(allowed_pairs) == len(expected)
            assert sorted([sorted(pair) for pair in expected]) == sorted([sorted(pair) for pair in allowed_pairs])

            coords = [[pair] for pair in allowed_pairs]
            allowed_coordinates = filt.filter_reaction_coordinates([cat], coords)
            assert len(allowed_coordinates) == len(coords)
            assert sorted(allowed_coordinates) == sorted(coords)

        allowed_coordinates = nolig_f.filter_reaction_coordinates([cat], coordinates)
        assert len(allowed_coordinates) == len(coordinates)  # wrong result because no reevaluation

        allowed_coordinates = re_nolig_f.filter_reaction_coordinates([cat], coordinates)
        assert len(allowed_coordinates) == 1

        # bimolecular
        range_2 = list(range(n_atoms[2]))
        range_20 = list(range(n_atoms[2], n_atoms[2] + n_atoms[0]))
        range_21 = list(range(n_atoms[2], n_atoms[2] + n_atoms[1]))
        for filt in [f, re_f]:
            for struc_range, struc_pair in zip([range_20, range_21],
                                               [[cat, self.structure], [cat, self.structure1]]):
                indices = range_2 + struc_range
                allowed_atoms = filt.filter_atoms(struc_pair, indices)
                assert len(allowed_atoms) == len(reactive) + len(struc_range)
                assert sorted(reactive + struc_range) == sorted(allowed_atoms)

                expected = list(combinations(reactive, 2)) + list(product(reactive, struc_range))
                allowed_pairs = filt.filter_atom_pairs(struc_pair, list(combinations(allowed_atoms, 2)))
                assert len(allowed_pairs) == len(expected)
                assert sorted([sorted(pair) for pair in expected]) == sorted([sorted(pair) for pair in allowed_pairs])

                coords = [[pair] for pair in allowed_pairs]
                allowed_coordinates = filt.filter_reaction_coordinates(struc_pair, coords)
                assert len(allowed_coordinates) == len(coords)
                assert sorted(allowed_coordinates) == sorted(coords)

                # coordinates are all valid pairs for ligand without central reactive
                allowed_coordinates = filt.filter_reaction_coordinates(struc_pair, coordinates)
                assert len(allowed_coordinates) == len(coordinates)
                assert sorted(allowed_coordinates) == sorted(coordinates)

        for filt in [nolig_f, re_nolig_f]:
            for struc_range, struc_pair in zip([range_20, range_21],
                                               [[cat, self.structure], [cat, self.structure1]]):
                indices = range_2 + struc_range
                allowed_atoms = filt.filter_atoms(struc_pair, indices)
                assert len(allowed_atoms) == len(reactive) + len(struc_range)
                assert sorted(reactive + struc_range) == sorted(allowed_atoms)

                expected = [pair for pair in combinations(reactive + struc_range, 2) if metal_index in pair]
                allowed_pairs = filt.filter_atom_pairs(struc_pair, list(combinations(allowed_atoms, 2)))
                assert len(allowed_pairs) == len(expected)
                assert sorted([sorted(pair) for pair in expected]) == sorted([sorted(pair) for pair in allowed_pairs])

                coords = [[pair] for pair in allowed_pairs]
                allowed_coordinates = filt.filter_reaction_coordinates(struc_pair, coords)
                assert len(allowed_coordinates) == len(coords)
                assert sorted(allowed_coordinates) == sorted(coords)

        allowed_coordinates = nolig_f.filter_reaction_coordinates([cat, self.structure], coordinates)
        assert len(allowed_coordinates) == len(coordinates)  # wrong result because no reevaluation

        allowed_coordinates = re_nolig_f.filter_reaction_coordinates([cat, self.structure], coordinates)
        assert len(allowed_coordinates) == 1

        allowed_coordinates = nolig_f.filter_reaction_coordinates([cat, self.structure1], coordinates)
        assert len(allowed_coordinates) == len(coordinates)  # wrong result because no reevaluation

        allowed_coordinates = re_nolig_f.filter_reaction_coordinates([cat, self.structure1], coordinates)
        assert len(allowed_coordinates) == 1

    def test_trivial_atom_pair_rule_based_filter(self):
        # trivial checks with association_rules == dissociation_rules and identical pairs
        rules = {
            'H': True,
            'C': True,
            'N': True,
            'O': True
        }
        f = AtomPairFunctionalGroupFilter(association_rules=[(rules, rules)], dissociation_rules=[(rules, rules)])
        assert f._association_rules
        assert f._dissociation_rules
        assert eval(repr(f._association_rules))  # pylint: disable=eval-used
        assert eval(repr(f._dissociation_rules))  # pylint: disable=eval-used
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
        f = AtomPairFunctionalGroupFilter(association_rules=[(rules, rules)], dissociation_rules=[(rules, rules)])
        assert f._association_rules
        assert f._dissociation_rules
        assert eval(repr(f._association_rules))  # pylint: disable=eval-used
        assert eval(repr(f._dissociation_rules))  # pylint: disable=eval-used
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
        f = AtomPairFunctionalGroupFilter(association_rules=[(rules, rules)], dissociation_rules=[(rules, rules)])
        assert f._association_rules
        assert f._dissociation_rules
        assert eval(repr(f._association_rules))  # pylint: disable=eval-used
        assert eval(repr(f._dissociation_rules))  # pylint: disable=eval-used
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
        f = AtomPairFunctionalGroupFilter(association_rules=[(rules, rules)], dissociation_rules=[(rules, rules)])
        assert f._association_rules
        assert f._dissociation_rules
        assert eval(repr(f._association_rules))  # pylint: disable=eval-used
        assert eval(repr(f._dissociation_rules))  # pylint: disable=eval-used
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
        f = AtomPairFunctionalGroupFilter(association_rules=[(rules, rules)], dissociation_rules=[(rules, rules)])
        assert f._association_rules
        assert f._dissociation_rules
        assert eval(repr(f._association_rules))  # pylint: disable=eval-used
        assert eval(repr(f._dissociation_rules))  # pylint: disable=eval-used
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

    def test_atom_pair_rule_based_filter_2(self):
        # some more specific checks involving different functionality

        # Get collections
        structures = self.manager.get_collection("structures")

        # Add structure data
        rr = resources_root_path()
        prol_adduct = json.load(open(os.path.join(rr, "proline_acid_propanal_product.json"), "r"))
        structure = db.Structure()
        structure.link(structures)
        structure.create(os.path.join(rr, "proline_acid_propanal_product.xyz"), 0, 1)
        structure.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure.set_graph("masm_cbor_graph", prol_adduct["masm_cbor_graph"])
        structure.set_graph("masm_idx_map", prol_adduct["masm_idx_map"])
        structure.set_compound(db.ID())

        water = json.load(open(os.path.join(rr, "water.json"), "r"))
        structure1 = db.Structure()
        structure1.link(structures)
        structure1.create(os.path.join(rr, "water.xyz"), 0, 1)
        structure1.set_label(db.Label.MINIMUM_OPTIMIZED)
        structure1.set_graph("masm_cbor_graph", water["masm_cbor_graph"])
        structure1.set_graph("masm_idx_map", water["masm_idx_map"])
        structure1.set_compound(db.ID())

        n_atoms_tot = 27 + 3

        any_O = {
            'O': AlwaysTrue()
        }
        O_bound_H = {
            'H': FunctionalGroupRule(1, 'O', (1, 3))
        }
        O_bound_C = {
            'C': FunctionalGroupRule(0, 'C', (1, 4), {'O': 1})
        }
        O_bound_C_strict = {
            'C': FunctionalGroupRule(0, 'C', (1, 4), {'O': 1}, strict_counts=True)
        }
        O_1 = {
            'O': FunctionalGroupRule(0, 'O', (1, 1))
        }
        O_12 = {
            'O': FunctionalGroupRule(0, 'O', (1, 2))
        }
        N3_bound_C = {
            'C': FunctionalGroupRule(1, 'N', (3, 3))
        }
        H2O_O = {
            'O': FunctionalGroupRule(0, 'O', (2, 2), {'H': 2}, strict_counts=True)
        }

        # Test 1: association between any O and any O-bound H:
        f = AtomPairFunctionalGroupFilter(association_rules=[(any_O, O_bound_H)], dissociation_rules=[])
        assert f._association_rules
        assert f._dissociation_rules == []
        assert eval(repr(f._association_rules))  # pylint: disable=eval-used
        assert eval(repr(f._dissociation_rules)) == []  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(n_atoms_tot)))
        reference = [
            9,    # H atom adduct
            14,   # O atom adduct
            15,   # O atom adduct
            16,   # H atom adduct
            20,   # O atom adduct
            27,   # H atom water
            28,   # H atom water
            29    # O atom water
        ]

        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        allowed_pairs = f.filter_atom_pairs([structure, structure1], list(combinations(allowed_atoms, 2)))
        reference = [
            (9, 14),
            (9, 20),
            (9, 29),
            (14, 16),
            (14, 27),
            (14, 28),
            (15, 16),
            (15, 27),
            (15, 28),
            (16, 29),
            (20, 27),
            (20, 28)
        ]

        assert len(allowed_pairs) == len(reference)
        for ref in reference:
            assert ref in allowed_pairs

        # Test 2: dissociation between any O and any O-bound H:
        f = AtomPairFunctionalGroupFilter(association_rules=[], dissociation_rules=[(any_O, O_bound_H)])
        assert f._association_rules == []
        assert f._dissociation_rules
        assert eval(repr(f._association_rules)) == []  # pylint: disable=eval-used
        assert eval(repr(f._dissociation_rules))  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(n_atoms_tot)))
        reference = [
            9,    # H atom adduct
            14,   # O atom adduct
            15,   # O atom adduct
            16,   # H atom adduct
            20,   # O atom adduct
            27,   # H atom water
            28,   # H atom water
            29    # O atom water
        ]

        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        allowed_pairs = f.filter_atom_pairs([structure, structure1], list(combinations(allowed_atoms, 2)))
        reference = [
            (9, 15),
            (16, 20),
            (27, 29),
            (28, 29)
        ]

        assert len(allowed_pairs) == len(reference)
        for ref in reference:
            assert ref in allowed_pairs

        # Test 3: association between C bound to one O (non-strict) and any O
        f = AtomPairFunctionalGroupFilter(association_rules=[(any_O, O_bound_C)], dissociation_rules=[])
        assert f._association_rules
        assert f._dissociation_rules == []
        assert eval(repr(f._association_rules))  # pylint: disable=eval-used
        assert eval(repr(f._dissociation_rules)) == []  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(n_atoms_tot)))
        reference = [
            13,
            14,
            15,
            17,
            20,
            29
        ]

        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        allowed_pairs = f.filter_atom_pairs([structure, structure1], list(combinations(allowed_atoms, 2)))
        reference = [
            (13, 20),
            (13, 29),
            (14, 17),
            (15, 17),
            (17, 29)
        ]

        assert len(allowed_pairs) == len(reference)
        for ref in reference:
            assert ref in allowed_pairs

        # Test 4: association between C bound to one O (strict) and any O
        f = AtomPairFunctionalGroupFilter(association_rules=[(O_bound_C_strict, any_O)], dissociation_rules=[])
        assert f._association_rules
        assert f._dissociation_rules == []
        assert eval(repr(f._association_rules))  # pylint: disable=eval-used
        assert eval(repr(f._dissociation_rules)) == []  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(n_atoms_tot)))
        reference = [
            14,
            15,
            17,
            20,
            29
        ]

        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        allowed_pairs = f.filter_atom_pairs([structure, structure1], list(combinations(allowed_atoms, 2)))
        reference = [
            (14, 17),
            (15, 17),
            (17, 29)
        ]

        assert len(allowed_pairs) == len(reference)
        for ref in reference:
            assert ref in allowed_pairs

        # Test 5: association between O having one bonding partner and any O-bound H
        f = AtomPairFunctionalGroupFilter(association_rules=[(O_bound_H, O_1)], dissociation_rules=[])
        assert f._association_rules
        assert f._dissociation_rules == []
        assert eval(repr(f._association_rules))  # pylint: disable=eval-used
        assert eval(repr(f._dissociation_rules)) == []  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(n_atoms_tot)))
        reference = [
            9,    # H of adduct
            14,   # carboxyl-O with only one C as bonding partner
            16,   # H of adduct
            27,   # H of water
            28    # H of water
        ]

        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        allowed_pairs = f.filter_atom_pairs([structure, structure1], list(combinations(allowed_atoms, 2)))
        reference = [
            (9, 14),
            (14, 16),
            (14, 27),
            (14, 28)
        ]

        assert len(allowed_pairs) == len(reference)
        for ref in reference:
            assert ref in allowed_pairs

        # Test 6: association between O having one or two bonding partners and any O-bound H.
        # This gives exactly the same selection as test 1.
        f = AtomPairFunctionalGroupFilter(association_rules=[(O_12, O_bound_H)], dissociation_rules=[])
        assert f._association_rules
        assert f._dissociation_rules == []
        assert eval(repr(f._association_rules))  # pylint: disable=eval-used
        assert eval(repr(f._dissociation_rules)) == []  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(n_atoms_tot)))
        reference = [
            9,    # H atom adduct
            14,   # O atom adduct
            15,   # O atom adduct
            16,   # H atom adduct
            20,   # O atom adduct
            27,   # H atom water
            28,   # H atom water
            29    # O atom water
        ]

        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        allowed_pairs = f.filter_atom_pairs([structure, structure1], list(combinations(allowed_atoms, 2)))
        reference = [
            (9, 14),
            (9, 20),
            (9, 29),
            (14, 16),
            (14, 27),
            (14, 28),
            (15, 16),
            (15, 27),
            (15, 28),
            (16, 29),
            (20, 27),
            (20, 28)
        ]

        assert len(allowed_pairs) == len(reference)
        for ref in reference:
            assert ref in allowed_pairs

        # Test 7: association between C atoms directly bound to an N atom with 3 bonding partners,
        # and an O atom from water
        f = AtomPairFunctionalGroupFilter(association_rules=[(N3_bound_C, H2O_O)], dissociation_rules=[])
        assert f._association_rules
        assert f._dissociation_rules == []
        assert eval(repr(f._association_rules))  # pylint: disable=eval-used
        assert eval(repr(f._dissociation_rules)) == []  # pylint: disable=eval-used
        allowed_atoms = f.filter_atoms([structure, structure1], list(range(n_atoms_tot)))
        reference = [
            2,   # C atom
            4,   # C atom
            17,  # C atom
            29,  # O atom
        ]

        assert len(allowed_atoms) == len(reference)
        for ref in reference:
            assert ref in allowed_atoms

        allowed_pairs = f.filter_atom_pairs([structure, structure1], list(combinations(allowed_atoms, 2)))
        reference = [
            (2, 29),
            (4, 29),
            (17, 29)
        ]

        assert len(allowed_pairs) == len(reference)
        for ref in reference:
            assert ref in allowed_pairs

    def test_library_rules_carbonyl(self):
        f = AtomRuleBasedFilter({'C': CarbonylX(0)})
        for structure in [self.structure, self.structure1, self.structure2, self.amide_acetal]:
            n_atoms_tot = len(structure.get_atoms())
            allowed_atoms = f.filter_atoms([structure], list(range(n_atoms_tot)))
            assert not allowed_atoms

        # Note that we define a carbonyl group as a sp2 carbon connected to an oxigen.
        n_atoms_tot = len(self.furfuryl_alcohol.get_atoms())
        allowed_atoms = f.filter_atoms([self.furfuryl_alcohol], list(range(n_atoms_tot)))
        assert allowed_atoms == [0, 4]

        n_atoms_tot = len(self.proline_propanal_product.get_atoms())
        allowed_atoms = f.filter_atoms([self.proline_propanal_product], list(range(n_atoms_tot)))
        assert allowed_atoms == [13]

    def test_library_rules_ch_olefin(self):
        f = AtomRuleBasedFilter({'C': CHOlefinC()})
        n_atoms_tot = len(self.furfuryl_alcohol.get_atoms())
        allowed_atoms = f.filter_atoms([self.furfuryl_alcohol], list(range(n_atoms_tot)))
        assert allowed_atoms == [1, 3]

    def test_library_rules_spNCX(self):
        fsp1 = AtomRuleBasedFilter({'C': SpNCX(0, 1)})
        n_atoms_tot = len(self.furfuryl_alcohol.get_atoms())
        allowed_atoms = fsp1.filter_atoms([self.furfuryl_alcohol], list(range(n_atoms_tot)))
        assert not allowed_atoms

        fsp2 = AtomRuleBasedFilter({'C': SpNCX(0, 2)})
        allowed_atoms = fsp2.filter_atoms([self.furfuryl_alcohol], list(range(n_atoms_tot)))
        assert allowed_atoms == [0, 1, 3, 4]

        fsp3 = AtomRuleBasedFilter({'C': SpNCX(0, 3)})
        allowed_atoms = fsp3.filter_atoms([self.furfuryl_alcohol], list(range(n_atoms_tot)))
        assert allowed_atoms == [5]

    def test_library_rules_carboxyl(self):
        f = AtomRuleBasedFilter({'C': CarboxylX(0)})
        n_atoms_tot = len(self.furfuryl_alcohol.get_atoms())
        allowed_atoms = f.filter_atoms([self.furfuryl_alcohol], list(range(n_atoms_tot)))
        assert not allowed_atoms

        n_atoms_tot = len(self.proline_propanal_product.get_atoms())
        allowed_atoms = f.filter_atoms([self.proline_propanal_product], list(range(n_atoms_tot)))
        assert allowed_atoms == [13]

    def test_library_rules_amid(self):
        f = AtomRuleBasedFilter({'C': AmidX(0)})
        n_atoms_tot = len(self.amide_acetal.get_atoms())
        allowed_atoms = f.filter_atoms([self.amide_acetal], list(range(n_atoms_tot)))
        assert not allowed_atoms

    def test_library_rules_allylic_c(self):
        f = AtomRuleBasedFilter({'C': AllylicSp3X(0)})
        n_atoms_tot = len(self.furfuryl_alcohol.get_atoms())
        allowed_atoms = f.filter_atoms([self.furfuryl_alcohol], list(range(n_atoms_tot)))
        assert allowed_atoms == [5]

        n_atoms_tot = len(self.structure2.get_atoms())
        allowed_atoms = f.filter_atoms([self.structure2], list(range(n_atoms_tot)))
        assert not allowed_atoms

        n_atoms_tot = len(self.amide_acetal.get_atoms())
        allowed_atoms = f.filter_atoms([self.amide_acetal], list(range(n_atoms_tot)))
        assert not allowed_atoms

    def test_library_rules_allylic_sp3(self):
        f = AtomRuleBasedFilter({'H': AllylicSp3X(1)})
        n_atoms_tot = len(self.furfuryl_alcohol.get_atoms())
        allowed_atoms = f.filter_atoms([self.furfuryl_alcohol], list(range(n_atoms_tot)))
        assert allowed_atoms == [11, 12]

        n_atoms_tot = len(self.amide_acetal.get_atoms())
        allowed_atoms = f.filter_atoms([self.amide_acetal], list(range(n_atoms_tot)))
        assert not allowed_atoms

    def test_library_rules_ammonium(self):
        f = AtomRuleBasedFilter({'N': AmmoniumX(0)})
        for structure in [self.amide_acetal, self.structure2]:
            n_atoms_tot = len(structure.get_atoms())
            allowed_atoms = f.filter_atoms([structure], list(range(n_atoms_tot)))
            assert not allowed_atoms

    def test_library_rules_amin(self):
        f = AtomRuleBasedFilter({'N': AminX(0)})
        n_atoms_tot = len(self.amide_acetal.get_atoms())
        allowed_atoms = f.filter_atoms([self.amide_acetal], list(range(n_atoms_tot)))
        assert allowed_atoms == [6]

        n_atoms_tot = len(self.structure2.get_atoms())
        allowed_atoms = f.filter_atoms([self.structure2], list(range(n_atoms_tot)))
        assert allowed_atoms == [4]

    def test_library_rules_carboxyl_h(self):
        f = AtomRuleBasedFilter({'H': CarboxylH()})
        n_atoms_tot = len(self.proline_propanal_product.get_atoms())
        allowed_atoms = f.filter_atoms([self.proline_propanal_product], list(range(n_atoms_tot)))
        assert allowed_atoms == [9]

    def test_library_rules_acetal(self):
        f = AtomRuleBasedFilter({'C': AcetalX(0)})
        n_atoms_tot = len(self.proline_propanal_product.get_atoms())
        allowed_atoms = f.filter_atoms([self.proline_propanal_product], list(range(n_atoms_tot)))
        assert allowed_atoms == [17]

        n_atoms_tot = len(self.amide_acetal.get_atoms())
        allowed_atoms = f.filter_atoms([self.amide_acetal], list(range(n_atoms_tot)))
        assert allowed_atoms == [1]

        f = AtomRuleBasedFilter({'C': AcetalX(1)})
        allowed_atoms = f.filter_atoms([self.amide_acetal], list(range(n_atoms_tot)))
        assert allowed_atoms == [0]

        f = AtomRuleBasedFilter({'H': AcetalX(2)})
        allowed_atoms = f.filter_atoms([self.amide_acetal], list(range(n_atoms_tot)))
        assert allowed_atoms == [2, 3, 5]

    def test_library_rules_default_oc(self):
        f = AtomRuleBasedFilter(DefaultOrganicChemistry())
        n_atoms_tot = len(self.proline_propanal_product.get_atoms())
        allowed_atoms = f.filter_atoms([self.proline_propanal_product], list(range(n_atoms_tot)))
        assert allowed_atoms == [2, 3, 8, 9, 13, 14, 15, 16, 17, 20, 22, 23]

        n_atoms_tot = len(self.amide_acetal.get_atoms())
        allowed_atoms = f.filter_atoms([self.amide_acetal], list(range(n_atoms_tot)))
        assert allowed_atoms == [1, 2, 3, 4, 5, 6, 16]

        n_atoms_tot = len(self.furfuryl_alcohol.get_atoms())
        allowed_atoms = f.filter_atoms([self.furfuryl_alcohol], list(range(n_atoms_tot)))
        assert allowed_atoms == [0, 1, 2, 3, 4, 5, 6, 7, 11, 12]

    def test_substructure_filter(self):
        rr = resources_root_path()
        grubbs = json.load(open(os.path.join(rr, "grubbs.json"), "r"))
        cat = db.Structure(db.ID(), self.manager.get_collection("structures"))
        cat.create(os.path.join(rr, "grubbs.xyz"), 0, 1, db_setup.get_fake_model(), db.Label.MINIMUM_OPTIMIZED)
        cat.set_graph("masm_cbor_graph", grubbs["masm_cbor_graph"])
        cat.set_graph("masm_idx_map", grubbs["masm_idx_map"])
        cat.set_aggregate(db.ID())

        ring_ids = [
            5, 6, 7, 8, 9, 10,
            35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52,
            122, 123, 124, 125, 126, 127,
            152, 153, 154, 155, 156, 157,
            158, 159, 160, 161, 162, 163,
            164, 165, 166, 167, 168, 169,
        ]

        f = SubStructureFilter(os.path.join(rr, 'substructures'), exclude_mode=False)
        n_atoms_tot = len(cat.get_atoms())
        allowed_atoms = f.filter_atoms([cat, cat], list(range(2 * n_atoms_tot)))
        assert len(allowed_atoms) == 48
        assert allowed_atoms == ring_ids
        f = SubStructureFilter(os.path.join(rr, 'substructures'), exclude_mode=False)
        n_atoms_tot = len(cat.get_atoms())
        allowed_atoms = f.filter_atoms([cat, cat], list(range(n_atoms_tot)))
        assert len(allowed_atoms) == 24
        assert allowed_atoms == ring_ids[:24]

        f = SubStructureFilter(os.path.join(rr, 'substructures'), exclude_mode=True)
        n_atoms_tot = len(cat.get_atoms())
        allowed_atoms = f.filter_atoms([cat, cat], list(range(2 * n_atoms_tot)))
        assert len(allowed_atoms) == (2 * n_atoms_tot - 48)
        assert allowed_atoms == list(set(range(2 * n_atoms_tot)).difference(set(ring_ids)))

        f = SubStructureFilter(os.path.join(rr, 'substructures'), exclude_mode=True)
        n_atoms_tot = len(cat.get_atoms())
        allowed_atoms = f.filter_atoms([cat, cat], list(range(n_atoms_tot)))
        assert len(allowed_atoms) == (n_atoms_tot - 24)
        assert allowed_atoms == list(set(range(n_atoms_tot)).difference(set(ring_ids)))

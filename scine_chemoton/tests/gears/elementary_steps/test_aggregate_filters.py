#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""


# Standard library imports
import os
import numpy as np
import unittest

# Third party imports
import scine_database as db
import scine_molassembler as masm
import scine_utilities as utils
from scine_database.insert_concentration import insert_concentration_for_compound
from scine_database import test_database_setup as db_setup

# Local application tests imports
from ....gears import HoldsCollections
from ...resources import resources_root_path
from ....utilities.calculation_creation_helpers import finalize_calculation
from ....utilities.masm import mol_to_cbor

# Local application imports
from scine_chemoton.filters.aggregate_filters import (
    AggregateFilter,
    AtomNumberFilter,
    CompoundCostPropertyFilter,
    ElementCountFilter,
    ElementSumCountFilter,
    MolecularWeightFilter,
    IdFilter,
    SelfReactionFilter,
    CatalystFilter,
    TrueMinimumFilter,
    AggregateFilterAndArray,
    AggregateFilterOrArray,
    SelectedAggregateIdFilter,
    OneAggregateIdFilter,
    ConcentrationPropertyFilter,
    ChargeCombinationFilter,
    SpecificChargeFilter,
    CompleteSubstructureFilter,
    LastKineticModelingFilter,
    HasStructureWithModel,
    StopDuringExploration,
    OnePotentialEnergySurface,
    ActivatedAggregateFilter,
)
from scine_chemoton.filters.reactive_site_filters import (
    ReactiveSiteFilter
)


class AggregateFiltersTests(unittest.TestCase, HoldsCollections):

    _required_collections = HoldsCollections.possible_attributes()

    def custom_setup(self, manager: db.Manager):
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties", "calculations"]
        self.initialize_collections(manager)
        self._manager = manager

    def tearDown(self) -> None:
        if hasattr(self, "_manager"):
            self._manager.wipe()

    def test_default_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_default_filter")
        self.custom_setup(manager)

        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        # Setup filter
        f = AggregateFilter()

        # Filter and check
        for i in ["water", "arginine", "cyclohexene"]:
            assert f.filter(test_compounds[i])
            for j in ["water", "arginine", "cyclohexene"]:
                assert f.filter(test_compounds[i], test_compounds[j])

    def test_filter_chain_failure_logic_and(self):
        with self.assertRaises(TypeError) as context:
            _ = (AggregateFilter() & ReactiveSiteFilter())
        self.assertTrue('AggregateFilter' in str(context.exception))

    def test_filter_chain_failure_logic_or(self):
        with self.assertRaises(TypeError) as context:
            _ = (AggregateFilter() | ReactiveSiteFilter())
        self.assertTrue('AggregateFilter' in str(context.exception))

    def test_filter_chain_failure_class_and(self):
        with self.assertRaises(TypeError) as context:
            AggregateFilterAndArray([AggregateFilter(), ReactiveSiteFilter()])
        self.assertTrue('AggregateFilterAndArray' in str(context.exception))

    def test_filter_chain_failure_class_or(self):
        with self.assertRaises(TypeError) as context:
            AggregateFilterOrArray([AggregateFilter(), ReactiveSiteFilter()])
        self.assertTrue('AggregateFilterOrArray' in str(context.exception))

    def test_filter_chain_derived_classes_or(self):
        _ = AggregateFilterOrArray([SelfReactionFilter(), IdFilter([])])
        _ = SelfReactionFilter() & IdFilter([])
        _ = SelfReactionFilter() and IdFilter([])

    def test_element_count_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_element_count_filter")
        self.custom_setup(manager)

        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        # Setup filter
        counts = {
            "O": 1,
            "C": 6,
            "H": 12,
        }
        f = ElementCountFilter(counts)
        f.initialize_collections(manager)

        expected_results = {
            ("water",): True,
            ("arginine",): False,
            ("cyclohexene",): True,
            ("water", "water"): True,
            ("water", "arginine"): False,
            ("water", "cyclohexene"): True,
            ("arginine", "water"): False,
            ("arginine", "arginine"): False,
            ("arginine", "cyclohexene"): False,
            ("cyclohexene", "water"): True,
            ("cyclohexene", "arginine"): False,
            ("cyclohexene", "cyclohexene"): True,
        }

        def check():
            for k, v in expected_results.items():
                if len(k) == 1:
                    assert v == f.filter(test_compounds[k[0]])
                elif len(k) == 2:
                    assert v == f.filter(test_compounds[k[0]], test_compounds[k[1]])
                else:
                    assert False

        check()
        counts["N"] = 1
        f = ElementCountFilter(counts, True)
        f.initialize_collections(manager)
        expected_results = {
            ("water",): False,
            ("arginine",): True,
            ("cyclohexene",): False,
            ("water", "water"): False,
            ("water", "arginine"): False,
            ("water", "cyclohexene"): False,
            ("arginine", "water"): False,
            ("arginine", "arginine"): True,
            ("arginine", "cyclohexene"): False,
            ("cyclohexene", "water"): False,
            ("cyclohexene", "arginine"): False,
            ("cyclohexene", "cyclohexene"): False,
        }
        check()

    def test_element_sum_count_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_element_sum_count_filter")
        self.custom_setup(manager)

        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        # Setup filter
        counts = {"O": 3, "H": 16, "N": 99, "C": 99}

        f = ElementSumCountFilter(counts)
        f.initialize_collections(manager)
        expected_results = {
            ("water",): True,
            ("arginine",): True,
            ("cyclohexene",): True,
            ("water", "water"): True,
            ("water", "arginine"): True,
            ("water", "cyclohexene"): True,
            ("arginine", "water"): True,
            ("arginine", "arginine"): False,
            ("arginine", "cyclohexene"): False,
            ("cyclohexene", "water"): True,
            ("cyclohexene", "arginine"): False,
            ("cyclohexene", "cyclohexene"): False,
        }

        def check():
            for k, v in expected_results.items():
                if len(k) == 1:
                    assert v == f.filter(test_compounds[k[0]])
                elif len(k) == 2:
                    assert v == f.filter(test_compounds[k[0]], test_compounds[k[1]])
                else:
                    assert False

        check()
        counts = {"O": 3, "H": 10, "C": 6, "N": 1}
        expected_results = {k: False for k, v in expected_results.items()}
        expected_results[("water", "arginine")] = True
        expected_results[("arginine", "water")] = True
        expected_results[("arginine", "arginine")] = True
        f = ElementSumCountFilter(counts, True)
        f.initialize_collections(manager)
        check()

    def test_molecular_weight_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_molecular_weight_filter")
        self.custom_setup(manager)

        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        # Setup filter
        f = MolecularWeightFilter(20.0)
        f.initialize_collections(manager)

        # Filter and check
        assert f.filter(test_compounds["water"])
        assert not f.filter(test_compounds["arginine"])
        assert not f.filter(test_compounds["cyclohexene"])
        assert f.filter(test_compounds["water"], test_compounds["water"])
        assert not f.filter(test_compounds["water"], test_compounds["arginine"])
        assert not f.filter(test_compounds["water"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["arginine"], test_compounds["water"])
        assert not f.filter(test_compounds["arginine"], test_compounds["arginine"])
        assert not f.filter(test_compounds["arginine"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["cyclohexene"], test_compounds["water"])
        assert not f.filter(test_compounds["cyclohexene"], test_compounds["arginine"])
        assert not f.filter(test_compounds["cyclohexene"], test_compounds["cyclohexene"])

    def test_id_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_id_filter")
        self.custom_setup(manager)

        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        id_list = []
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.USER_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

            if xyz in ("water", "arginine"):
                id_list.append(str(compound.get_id()))

        f = IdFilter(id_list)

        # Filter and check
        assert f.filter(test_compounds["water"])
        assert f.filter(test_compounds["arginine"])
        assert not f.filter(test_compounds["cyclohexene"])
        assert f.filter(test_compounds["water"], test_compounds["water"])
        assert f.filter(test_compounds["arginine"], test_compounds["arginine"])
        assert f.filter(test_compounds["water"], test_compounds["arginine"])
        assert not f.filter(test_compounds["water"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["arginine"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["cyclohexene"], test_compounds["cyclohexene"])

    def test_self_reaction_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_self_reaction_filter")
        self.custom_setup(manager)

        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        f = SelfReactionFilter()

        # Filter and check
        assert f.filter(test_compounds["water"])
        assert f.filter(test_compounds["arginine"])
        assert f.filter(test_compounds["cyclohexene"])
        assert f.filter(test_compounds["water"], test_compounds["arginine"])
        assert f.filter(test_compounds["water"], test_compounds["cyclohexene"])
        assert f.filter(test_compounds["arginine"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["cyclohexene"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["water"], test_compounds["water"])
        assert not f.filter(test_compounds["arginine"], test_compounds["arginine"])

    def test_true_minimum_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_true_minimum_filter")
        self.custom_setup(manager)

        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["arginine", "h4o2", "allylvinylether", "hydrogen"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())

            if xyz in ["arginine", "h4o2"]:
                data = np.loadtxt(os.path.join(rr, xyz + "_frequencies.dat"), dtype="float64")

                frequencies = db.VectorProperty()
                frequencies.link(self._properties)
                frequencies.create(structure.get_model(), "frequencies", data)
                frequencies.set_structure(structure.id())
                structure.set_property("frequencies", frequencies.id())

            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        f = TrueMinimumFilter(0.0)
        f.initialize_collections(manager)

        assert f.filter(test_compounds["arginine"])
        assert not f.filter(test_compounds["h4o2"])
        assert f.filter(test_compounds["hydrogen"])
        assert not f.filter(test_compounds["allylvinylether"])

        assert f.filter(test_compounds["arginine"], test_compounds["arginine"])
        assert not f.filter(test_compounds["arginine"], test_compounds["h4o2"])
        assert f.filter(test_compounds["arginine"], test_compounds["hydrogen"])
        assert not f.filter(test_compounds["h4o2"], test_compounds["arginine"])
        assert not f.filter(test_compounds["h4o2"], test_compounds["h4o2"])
        assert not f.filter(test_compounds["h4o2"], test_compounds["hydrogen"])
        assert f.filter(test_compounds["hydrogen"], test_compounds["arginine"])
        assert not f.filter(test_compounds["hydrogen"], test_compounds["h4o2"])
        assert f.filter(test_compounds["hydrogen"], test_compounds["hydrogen"])

        assert not f.filter(test_compounds["allylvinylether"], test_compounds["arginine"])
        assert not f.filter(test_compounds["allylvinylether"], test_compounds["h4o2"])

    def test_catalyst_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_catalyst_filter")
        self.custom_setup(manager)

        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        # Setup filter
        counts = {"C": 6, "H": 10}
        f = CatalystFilter(counts)
        f.initialize_collections(manager)

        # Filter and check
        assert f.filter(test_compounds["water"])
        assert f.filter(test_compounds["arginine"])
        assert f.filter(test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["water"], test_compounds["water"])
        assert not f.filter(test_compounds["water"], test_compounds["arginine"])
        assert f.filter(test_compounds["water"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["arginine"], test_compounds["water"])
        assert not f.filter(test_compounds["arginine"], test_compounds["arginine"])
        assert f.filter(test_compounds["arginine"], test_compounds["cyclohexene"])
        assert f.filter(test_compounds["cyclohexene"], test_compounds["water"])
        assert f.filter(test_compounds["cyclohexene"], test_compounds["arginine"])
        assert not f.filter(test_compounds["cyclohexene"], test_compounds["cyclohexene"])

    def test_one_compound_id_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_one_compound_id_filter")
        self.custom_setup(manager)

        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        id_list = []
        xyzs = ["water", "arginine", "cyclohexene", "h4o2"]
        for xyz in xyzs:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.USER_OPTIMIZED)
            structure.set_compound(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_compound(compound.id())
            test_compounds[xyz] = compound

            if xyz in ("water", "arginine"):
                id_list.append(str(compound.get_id()))

        f = OneAggregateIdFilter(id_list)

        # Filter and check
        # One compound case
        assert f.filter(test_compounds["water"])
        assert f.filter(test_compounds["arginine"])
        assert not f.filter(test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["h4o2"])
        # Two compound case
        for xyz in xyzs:
            assert f.filter(test_compounds["water"], test_compounds[xyz])
            assert f.filter(test_compounds["arginine"], test_compounds[xyz])
            assert f.filter(test_compounds[xyz], test_compounds["water"])
            assert f.filter(test_compounds[xyz], test_compounds["arginine"])
        assert not f.filter(test_compounds["cyclohexene"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["h4o2"], test_compounds["h4o2"])
        assert not f.filter(test_compounds["h4o2"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["cyclohexene"], test_compounds["h4o2"])

    def test_selected_compound_id_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_one_compound_id_filter")
        self.custom_setup(manager)

        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        reactive_ids = []
        selected_ids = []
        xyzs = ["water", "arginine", "cyclohexene", "h4o2"]
        for xyz in xyzs:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.USER_OPTIMIZED)
            structure.set_compound(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_compound(compound.id())
            test_compounds[xyz] = compound

            if xyz in ("water", "arginine"):
                reactive_ids.append(str(compound.get_id()))
            if xyz in ("h4o2"):
                selected_ids.append(str(compound.get_id()))

        f = SelectedAggregateIdFilter(reactive_ids, selected_ids)

        # Filter and check
        # One compound case
        assert f.filter(test_compounds["water"])
        assert f.filter(test_compounds["arginine"])
        assert not f.filter(test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["h4o2"])
        # Two compound case
        for xyz in xyzs:
            assert not f.filter(test_compounds["cyclohexene"], test_compounds[xyz])
            assert not f.filter(test_compounds[xyz], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["h4o2"], test_compounds["h4o2"])
        assert f.filter(test_compounds["h4o2"], test_compounds["water"])
        assert f.filter(test_compounds["water"], test_compounds["h4o2"])
        assert f.filter(test_compounds["water"], test_compounds["arginine"])
        assert f.filter(test_compounds["arginine"], test_compounds["water"])

    def test_and_filter_array(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_and_filter_array")
        self.custom_setup(manager)

        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        # Setup filter
        counts_one = {
            "O": 1,
            "C": 6,
            "H": 20,
        }
        counts_two = {"C": 6, "H": 10}
        f = CatalystFilter(counts_two) & ElementCountFilter(counts_one)
        f.initialize_collections(manager)

        # Filter and check
        assert f.filter(test_compounds["water"])
        assert not f.filter(test_compounds["arginine"])
        assert f.filter(test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["water"], test_compounds["water"])
        assert not f.filter(test_compounds["water"], test_compounds["arginine"])
        assert f.filter(test_compounds["water"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["arginine"], test_compounds["water"])
        assert not f.filter(test_compounds["arginine"], test_compounds["arginine"])
        assert not f.filter(test_compounds["arginine"], test_compounds["cyclohexene"])
        assert f.filter(test_compounds["cyclohexene"], test_compounds["water"])
        assert not f.filter(test_compounds["cyclohexene"], test_compounds["arginine"])
        assert not f.filter(test_compounds["cyclohexene"], test_compounds["cyclohexene"])

    def test_or_filter_array(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_or_filter_array")
        self.custom_setup(manager)

        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        # Setup filter
        counts_one = {
            "O": 1,
            "C": 6,
            "H": 20,
        }
        counts_two = {"C": 6, "H": 10}
        f = CatalystFilter(counts_two) | ElementCountFilter(counts_one)
        f.initialize_collections(manager)

        # Filter and check
        assert f.filter(test_compounds["water"])
        assert f.filter(test_compounds["arginine"])
        assert f.filter(test_compounds["cyclohexene"])
        assert f.filter(test_compounds["water"], test_compounds["water"])
        assert not f.filter(test_compounds["water"], test_compounds["arginine"])
        assert f.filter(test_compounds["water"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["arginine"], test_compounds["water"])
        assert not f.filter(test_compounds["arginine"], test_compounds["arginine"])
        assert f.filter(test_compounds["arginine"], test_compounds["cyclohexene"])
        assert f.filter(test_compounds["cyclohexene"], test_compounds["water"])
        assert f.filter(test_compounds["cyclohexene"], test_compounds["arginine"])
        assert f.filter(test_compounds["cyclohexene"], test_compounds["cyclohexene"])

    def test_atom_number_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_atom_number_filter")
        self.custom_setup(manager)

        # Add structure data
        rr = resources_root_path()
        test_compounds = []
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.USER_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds.append(compound)

        numbers = [2, 10, 20, 100]
        for i, number in enumerate(numbers):
            f = AtomNumberFilter(number)
            f.initialize_collections(manager)
            n_allowed = sum(int(f.filter(compound)) for compound in test_compounds)
            assert n_allowed == i

    def test_compound_cost_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_compound_cost_filter")
        self.custom_setup(manager)
        model = db.Model("FAKE", "FAKE", "F-AKE")

        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz, cc in zip(["water", "arginine", "cyclohexene", "h4o2"], [1.0, 10.0, 500.0, -1.0]):
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            # Add property
            if cc > 0.0:
                prop = db.NumberProperty.make("compound_cost", model, cc, self._properties)
                structure.add_property("compound_cost", prop.id())
                prop.set_structure(structure.id())
            # Construct compound
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        f = CompoundCostPropertyFilter(11.0)
        f.initialize_collections(manager)

        assert f.filter(test_compounds["water"])
        assert f.filter(test_compounds["arginine"])
        assert not f.filter(test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["h4o2"])
        assert f.filter(test_compounds["water"], test_compounds["water"])
        assert not f.filter(test_compounds["water"], test_compounds["arginine"])
        assert not f.filter(test_compounds["arginine"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["h4o2"], test_compounds["water"])

    def test_concentration_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_concentration_filter")
        self.custom_setup(manager)

        # Get collections
        compounds = manager.get_collection("compounds")

        c1_id, _ = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c2_id, _ = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c3_id, _ = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c4_id, _ = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c1 = db.Compound(c1_id, compounds)
        c2 = db.Compound(c2_id, compounds)
        c3 = db.Compound(c3_id, compounds)
        c4 = db.Compound(c4_id, compounds)

        f = ConcentrationPropertyFilter(["max_concentration", "start_concentration"], 1e-2, True)
        f.initialize_collections(manager)

        insert_concentration_for_compound(manager, 0.5, db.Model("FAKE", "", ""), c1_id, False, "max_concentration")
        insert_concentration_for_compound(manager, 0.5, db.Model("FAKE", "", ""), c2_id, False, "start_concentration")

        assert f.filter(c1)
        assert f.filter(c2)
        assert f.filter(c3)
        assert f.filter(c4)

        assert f.filter(c1, c2)
        assert not f.filter(c1, c3)
        assert not f.filter(c1, c4)
        assert not f.filter(c2, c3)
        assert not f.filter(c2, c4)
        assert not f.filter(c3, c4)

        f = ConcentrationPropertyFilter(["max_concentration", "start_concentration"], 3e-1, False)
        f.initialize_collections(manager)

        assert f.filter(c1)
        assert f.filter(c2)
        assert not f.filter(c3)
        assert not f.filter(c4)

        assert not f.filter(c1, c2)
        assert not f.filter(c1, c3)
        assert not f.filter(c1, c4)
        assert not f.filter(c2, c3)
        assert not f.filter(c2, c4)
        assert not f.filter(c3, c4)

        f = ConcentrationPropertyFilter(["max_concentration", "start_concentration"], 3e-1, False,
                                        variance_labels=["manual_activation", None])
        f.initialize_collections(manager)
        insert_concentration_for_compound(manager, 0.05, db.Model("FAKE", "", ""), c1_id, False, "manual_activation")

        assert f.filter(c1)
        assert f.filter(c2)
        assert not f.filter(c3)
        assert not f.filter(c4)

        assert f.filter(c1, c2)
        assert not f.filter(c1, c3)
        assert not f.filter(c1, c4)
        assert not f.filter(c2, c3)
        assert not f.filter(c2, c4)
        assert not f.filter(c3, c4)

        # Cleaning
        manager.wipe()

    def test_charge_combination_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_charge_filter")
        self.custom_setup(manager)

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")

        c1_id, _ = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c2_id, s2_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c3_id, s3_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c4_id, s4_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c1 = db.Compound(c1_id, compounds)
        c2 = db.Compound(c2_id, compounds)
        c3 = db.Compound(c3_id, compounds)
        c4 = db.Compound(c4_id, compounds)
        s2 = db.Structure(s2_id, structures)
        s3 = db.Structure(s3_id, structures)
        s4 = db.Structure(s4_id, structures)
        s2.charge = +1
        s3.charge = +3
        s4.charge = -7

        f = ChargeCombinationFilter()
        f.initialize_collections(manager)

        assert f.filter(c1)
        assert f.filter(c2)
        assert f.filter(c3)
        assert f.filter(c4)

        assert f.filter(c1, c2)
        assert f.filter(c1, c3)
        assert f.filter(c1, c4)

        assert not f.filter(c2, c3)
        assert not f.filter(c2, c2)
        assert not f.filter(c3, c2)
        assert not f.filter(c3, c3)

        assert not f.filter(c4, c4)
        assert f.filter(c4, c3)
        assert f.filter(c4, c2)
        # Cleaning
        manager.wipe()

    def test_specific_charge_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_specific_charge_filter")
        self.custom_setup(manager)

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")

        c1_id, _ = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c2_id, s2_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c3_id, s3_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c4_id, s4_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c1 = db.Compound(c1_id, compounds)
        c2 = db.Compound(c2_id, compounds)
        c3 = db.Compound(c3_id, compounds)
        c4 = db.Compound(c4_id, compounds)
        s2 = db.Structure(s2_id, structures)
        s3 = db.Structure(s3_id, structures)
        s4 = db.Structure(s4_id, structures)
        s2.charge = +1
        s3.charge = +3
        s4.charge = -7

        for allow in [True, False]:
            for both in [True, False]:
                f = SpecificChargeFilter(charges=[+1, -7], allow=allow, both_charges_must_be_valid=both)
                f.initialize_collections(manager)

                assert f.filter(c1) == (not allow)
                assert f.filter(c2) == allow
                assert f.filter(c3) == (not allow)
                assert f.filter(c4) == allow

                assert f.filter(c1, c1) == (not allow)
                assert f.filter(c2, c2) == allow
                assert f.filter(c3, c3) == (not allow)
                assert f.filter(c4, c4) == allow

                assert f.filter(c1, c2) == (both is not allow)
                assert f.filter(c1, c3) == (not allow)
                assert f.filter(c1, c4) == (both is not allow)
                assert f.filter(c2, c3) == (both is not allow)
                assert f.filter(c3, c2) == (both is not allow)
                assert f.filter(c4, c3) == (both is not allow)
                assert f.filter(c4, c2) == allow

        # Cleaning
        manager.wipe()

    def test_last_kinetic_modeling_job_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_last_kinetic_modeling_job_filter")
        self.custom_setup(manager)

        # Get collections
        calculations = manager.get_collection("calculations")
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")

        c1_id, s1_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c2_id, s2_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c3_id, s3_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c4_id, _ = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c1 = db.Compound(c1_id, compounds)
        c2 = db.Compound(c2_id, compounds)
        c3 = db.Compound(c3_id, compounds)
        c4 = db.Compound(c4_id, compounds)

        model = db.Model("FAKE", "FAKE", "F-AKE")
        insert_concentration_for_compound(manager, 1.0, model, c1_id)
        insert_concentration_for_compound(manager, 1.0, model, c2_id)

        job_order = "fake_kinetic_modeling"
        aggregate_key = "fake_aggregate_key"
        f = LastKineticModelingFilter(job_order, aggregate_key)
        f.initialize_collections(manager)

        assert f.filter(c1)
        assert f.filter(c2)
        assert not f.filter(c3)
        assert not f.filter(c4)

        assert f.filter(c1, c2)
        assert f.filter(c1, c1)
        assert f.filter(c2, c2)
        assert not f.filter(c1, c3)
        assert not f.filter(c1, c4)
        assert not f.filter(c2, c3)
        assert not f.filter(c2, c4)
        assert not f.filter(c3, c4)

        calculation = db.Calculation()
        calculation.link(calculations)
        calculation.create(model, db.Job(job_order), [s1_id, s2_id, s3_id])
        settings_dict = {aggregate_key: [c1_id.string(), c2_id.string(), c3_id.string()]}
        settings = utils.ValueCollection(settings_dict)
        calculation.set_settings(settings)
        finalize_calculation(calculation, structures)

        assert f.filter(c1)
        assert f.filter(c2)
        assert not f.filter(c3)
        assert not f.filter(c4)

        assert f.filter(c1, c2)
        assert f.filter(c1, c1)
        assert f.filter(c2, c2)
        assert not f.filter(c1, c3)
        assert not f.filter(c1, c4)
        assert not f.filter(c2, c3)
        assert not f.filter(c2, c4)
        assert not f.filter(c3, c4)

        calculation.set_status(db.Status.COMPLETE)

        assert f.filter(c1)
        assert f.filter(c2)
        assert f.filter(c3)
        assert not f.filter(c4)

        assert f.filter(c1, c2)
        assert f.filter(c1, c1)
        assert f.filter(c2, c2)
        assert f.filter(c1, c3)
        assert not f.filter(c1, c4)
        assert f.filter(c2, c3)
        assert not f.filter(c2, c4)
        assert not f.filter(c3, c4)

        # Cleaning
        manager.wipe()

    def test_substructure(self):
        manager = db_setup.get_clean_db("chemoton_test_substructure_aggregate_filter")
        self.custom_setup(manager)
        rr = resources_root_path()
        sub_filter = CompleteSubstructureFilter(rr)
        double_sub_filter = CompleteSubstructureFilter(rr, require_both_match_bimolecular=True)
        sub_filter.initialize_collections(manager)
        double_sub_filter.initialize_collections(manager)
        assert sub_filter._molecules

        model = db_setup.get_fake_model()

        def get_path(name: str) -> str:
            mol_path = os.path.join(rr, name + ".mol")
            if not os.path.exists(mol_path):
                mol_path = mol_path[:-4] + ".xyz"
            return mol_path

        # Add structure data
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene", "h4o2", "ga_complex", "ethene"]:
            structure = db.Structure()
            structure.link(self._structures)
            path = get_path(xyz)
            structure.create(path, 0, 1, model, db.Label.MINIMUM_OPTIMIZED)
            ac, bo = utils.io.read(path)
            if bo.empty():
                bo = utils.BondDetector.detect_bonds(ac)
            mols = masm.interpret.molecules(
                ac, bo, masm.interpret.BondDiscretization.RoundToNearest).molecules
            # todo replace once Molassembler has been updated
            # mols = masm.io.split(path)
            graph = ";".join(mol_to_cbor(mol) for mol in mols)
            structure.set_graph("masm_cbor_graph", graph)
            # Construct compound
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            test_compounds[xyz] = compound

        assert all(sub_filter.filter(c) for c in test_compounds.values())
        assert all(double_sub_filter.filter(c) for c in test_compounds.values())
        assert all(sub_filter.filter(ci, cj) for ci in test_compounds.values() for cj in test_compounds.values())
        assert all(double_sub_filter.filter(ci, cj) for ci in test_compounds.values() for cj in test_compounds.values())

        for sub_mol, valids in zip(["water", "ethene"], [["water", "h4o2"], ["ethene", "ga_complex"]]):
            sub_path = get_path(sub_mol)
            sub_filter = CompleteSubstructureFilter(sub_path)
            double_sub_filter = CompleteSubstructureFilter(sub_path, require_both_match_bimolecular=True)
            sub_filter.initialize_collections(manager)
            double_sub_filter.initialize_collections(manager)

            for ki, vi in test_compounds.items():
                for kj, vj in test_compounds.items():
                    if ki == kj:
                        if ki in valids:
                            assert sub_filter.filter(vi)
                            assert double_sub_filter.filter(vi)
                        else:
                            assert not sub_filter.filter(vi)
                            assert not double_sub_filter.filter(vi)
                    else:
                        if ki in valids and kj in valids:
                            assert sub_filter.filter(vi, vj)
                            assert double_sub_filter.filter(vi, vj)
                        elif ki in valids or kj in valids:
                            assert sub_filter.filter(vi, vj)
                            assert not double_sub_filter.filter(vi, vj)
                        else:
                            assert not sub_filter.filter(vi, vj)
                            assert not double_sub_filter.filter(vi, vj)

    def test_has_structure_with_model_filter_energy(self):
        manager = db_setup.get_random_db(n_compounds=7, n_flasks=2, n_reactions=4, max_r_per_c=4,
                                         name="chemoton_test_has_structure_with_model_filter_energy")
        self.custom_setup(manager)

        test_model = db.Model("My", "test", "model")
        f = HasStructureWithModel(test_model, check_only_energies=True)
        f.initialize_collections(manager)
        compounds = self._compounds.random_select_compounds(7)
        for c in compounds:
            c.link(self._compounds)
        flasks = self._flasks.random_select_flasks(2)
        for c in flasks:
            c.link(self._flasks)
        aggregates = compounds + flasks

        for a in aggregates:
            assert not f.filter(a)
            s_id = a.get_structures()[-1]
            structure = db.Structure(s_id, self._structures)
            db_setup.add_random_energy(structure, (-10, -11), self._properties, test_model)

        for a in aggregates:
            assert f.filter(a)

    def test_has_structure_with_model_filter_structure(self):
        manager = db_setup.get_random_db(n_compounds=7, n_flasks=2, n_reactions=4, max_r_per_c=4,
                                         name="chemoton_test_has_structure_with_model_filter_structure")
        self.custom_setup(manager)

        test_model = db.Model("My", "test", "model")
        f = HasStructureWithModel(test_model, check_only_energies=False)
        f.initialize_collections(manager)
        compounds = self._compounds.random_select_compounds(7)
        for c in compounds:
            c.link(self._compounds)
        flasks = self._flasks.random_select_flasks(2)
        for c in flasks:
            c.link(self._flasks)
        aggregates = compounds + flasks

        for a in aggregates:
            assert not f.filter(a)
            s_id = a.get_structures()[-1]
            structure = db.Structure(s_id, self._structures)
            structure.set_model(test_model)

        for a in aggregates:
            assert f.filter(a)

    def test_stop_during_exploration_filter(self):
        manager = db_setup.get_random_db(n_compounds=7, n_flasks=2, n_reactions=4, max_r_per_c=4,
                                         name="chemoton_test_stop_during_exploration_filter")
        self.custom_setup(manager)
        model = db_setup.get_fake_model()
        wait_for = ["some-order"]
        f = StopDuringExploration(model=model, orders_to_wait_for=wait_for)
        f.initialize_collections(manager)
        for c in self._compounds.iterate_all_compounds():
            c.link(self._compounds)
            assert f.filter(c)

        new_calculation = db.Calculation.make(model, db.Job(wait_for[0]), [db.ID()], self._calculations)
        for status in [db.Status.NEW, db.Status.HOLD, db.Status.PENDING]:
            new_calculation.set_status(status)
            for c in self._compounds.iterate_all_compounds():
                c.link(self._compounds)
                assert not f.filter(c)

        new_calculation.set_status(db.Status.COMPLETE)
        for c in self._compounds.iterate_all_compounds():
            c.link(self._compounds)
            assert f.filter(c)

        new_calculation = db.Calculation.make(model, db.Job("some-other-label"), [db.ID()], self._calculations)
        for status in [db.Status.NEW, db.Status.HOLD, db.Status.PENDING]:
            new_calculation.set_status(status)
            for c in self._compounds.iterate_all_compounds():
                c.link(self._compounds)
                assert f.filter(c)

        new_calculation = db.Calculation.make(db.Model("some", "other", "model"), db.Job(wait_for[0]), [db.ID()],
                                              self._calculations)
        for status in [db.Status.NEW, db.Status.HOLD, db.Status.PENDING]:
            new_calculation.set_status(status)
            for c in self._compounds.iterate_all_compounds():
                c.link(self._compounds)
                assert f.filter(c)

    def test_one_potential_energy_surface_filter(self):
        manager = db_setup.get_clean_db("chemoton_test_one_potential_energy_surface_filter")
        self.custom_setup(manager)

        rr = resources_root_path()
        test_compounds = []
        test_structures = []
        for xyz in ["water", "water", "h4o2", "furfuryl_alcohol"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds.append(compound)
            test_structures.append(structure)

        reaction = db.Reaction()
        reaction.link(self._reactions)
        reaction.create([test_compounds[2].id()], [test_compounds[0].id(), test_compounds[1].id()])
        test_compounds[0].add_reaction(reaction.id())
        test_compounds[1].add_reaction(reaction.id())

        f = OnePotentialEnergySurface.from_structure(test_structures[2])
        f.initialize_collections(manager)

        assert not f.filter(test_compounds[0])
        assert not f.filter(test_compounds[1])
        assert not f.filter(test_compounds[3])
        assert f.filter(test_compounds[2])
        assert f.filter(test_compounds[0], test_compounds[1])

        f = OnePotentialEnergySurface(total_charge=0, element_counts={"H": 4, "O": 2})
        f.initialize_collections(manager)
        test_structures[2].set_charge(+1)
        assert not f.filter(test_compounds[2])

    def test_activated_aggregate_filter(self):
        manager = db_setup.get_clean_db("chemoton_test_activated_aggregate_filter")
        self.custom_setup(manager)

        rr = resources_root_path()
        test_compounds = []
        test_structures = []
        for xyz in ["water", "h4o2", "furfuryl_alcohol"]:
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds.append(compound)
            test_structures.append(structure)
            compound.disable_exploration()

        filter = ActivatedAggregateFilter()
        for compound in test_compounds:
            assert not filter.filter(compound)
            assert not filter.filter(compound, test_compounds[0])
            compound.enable_exploration()

        for compound in test_compounds:
            assert filter.filter(compound)
            assert filter.filter(compound, test_compounds[0])

        test_compounds[0].disable_exploration()
        assert not filter.filter(test_compounds[1], test_compounds[0])

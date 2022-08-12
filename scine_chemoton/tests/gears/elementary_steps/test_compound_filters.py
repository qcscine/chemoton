#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""


# Standard library imports
import os
import numpy as np
import unittest

# Third party imports
import scine_database as db

# Local application tests imports
from ... import test_database_setup as db_setup
from ...resources import resources_root_path

# Local application imports
from ....gears.elementary_steps.compound_filters import (
    CompoundFilter,
    ElementCountFilter,
    ElementSumCountFilter,
    MolecularWeightFilter,
    IDFilter,
    SelfReactionFilter,
    CatalystFilter,
    TrueMinimumFilter,
    CompoundFilterAndArray,
    CompoundFilterOrArray,
    SelectedCompoundIDFilter,
    OneCompoundIDFilter
)
from ....gears.elementary_steps.reactive_site_filters import (
    ReactiveSiteFilter
)


class CompoundFiltersTests(unittest.TestCase):

    def test_default_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_default_filter")

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        # Setup filter
        f = CompoundFilter()

        # Filter and check
        for i in ["water", "arginine", "cyclohexene"]:
            assert f.filter(test_compounds[i])
            for j in ["water", "arginine", "cyclohexene"]:
                assert f.filter(test_compounds[i], test_compounds[j])

        # Cleaning
        manager.wipe()

    def test_filter_chain_failure_logic_and(self):
        with self.assertRaises(TypeError) as context:
            _ = (CompoundFilter() & ReactiveSiteFilter())
        self.assertTrue('CompoundFilter' in str(context.exception))

    def test_filter_chain_failure_logic_or(self):
        with self.assertRaises(TypeError) as context:
            _ = (CompoundFilter() | ReactiveSiteFilter())
        self.assertTrue('CompoundFilter' in str(context.exception))

    def test_filter_chain_failure_class_and(self):
        with self.assertRaises(TypeError) as context:
            CompoundFilterAndArray([CompoundFilter(), ReactiveSiteFilter()])
        self.assertTrue('CompoundFilterAndArray' in str(context.exception))

    def test_filter_chain_failure_class_or(self):
        with self.assertRaises(TypeError) as context:
            CompoundFilterOrArray([CompoundFilter(), ReactiveSiteFilter()])
        self.assertTrue('CompoundFilterOrArray' in str(context.exception))

    def test_filter_chain_derived_classes_or(self):
        _ = CompoundFilterOrArray([SelfReactionFilter(), IDFilter([])])
        _ = SelfReactionFilter() & IDFilter([])
        _ = SelfReactionFilter() and IDFilter([])

    def test_element_count_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_element_count_filter")

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        # Setup filter
        counts = {
            "O": 1,
            "C": 6,
            "H": 20,
        }
        f = ElementCountFilter(counts, structures)

        # Filter and check
        assert f.filter(test_compounds["water"])
        assert not f.filter(test_compounds["arginine"])
        assert f.filter(test_compounds["cyclohexene"])
        assert f.filter(test_compounds["water"], test_compounds["water"])
        assert not f.filter(test_compounds["water"], test_compounds["arginine"])
        assert f.filter(test_compounds["water"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["arginine"], test_compounds["water"])
        assert not f.filter(test_compounds["arginine"], test_compounds["arginine"])
        assert not f.filter(test_compounds["arginine"], test_compounds["cyclohexene"])
        assert f.filter(test_compounds["cyclohexene"], test_compounds["water"])
        assert not f.filter(test_compounds["cyclohexene"], test_compounds["arginine"])
        assert f.filter(test_compounds["cyclohexene"], test_compounds["cyclohexene"])

        # Cleaning
        manager.wipe()

    def test_element_sum_count_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_element_sum_count_filter")

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        # Setup filter
        counts = {"O": 3, "H": 16, "N": 99, "C": 99}

        f = ElementSumCountFilter(counts, structures)
        # Filter and check
        assert f.filter(test_compounds["water"])
        assert f.filter(test_compounds["arginine"])
        assert f.filter(test_compounds["cyclohexene"])

        assert not f.filter(test_compounds["arginine"], test_compounds["arginine"])
        assert f.filter(test_compounds["arginine"], test_compounds["water"])
        assert not f.filter(test_compounds["arginine"], test_compounds["cyclohexene"])
        assert f.filter(test_compounds["water"], test_compounds["arginine"])
        assert f.filter(test_compounds["water"], test_compounds["water"])
        assert f.filter(test_compounds["water"], test_compounds["cyclohexene"])
        assert not f.filter(test_compounds["cyclohexene"], test_compounds["arginine"])
        assert f.filter(test_compounds["cyclohexene"], test_compounds["water"])
        assert not f.filter(test_compounds["cyclohexene"], test_compounds["cyclohexene"])

        # Cleaning
        manager.wipe()

    def test_molecular_weight_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_molecular_weight_filter")

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        # Setup filter
        f = MolecularWeightFilter(20.0, structures)

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

        # Cleaning
        manager.wipe()

    def test_id_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_id_filter")

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        id_list = []
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.USER_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

            if xyz in ("water", "arginine"):
                id_list.append(compound.get_id())

        f = IDFilter(id_list)

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

        # Cleaning
        manager.wipe()

    def test_self_reaction_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_self_reaction_filter")

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(compounds)
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

        # Cleaning
        manager.wipe()

    def test_true_minimum_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_true_minimum_filter")

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        properties = manager.get_collection("properties")
        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["arginine", "h4o2", "allylvinylether", "hydrogen"]:
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())

            if xyz in ["arginine", "h4o2"]:
                data = np.loadtxt(os.path.join(rr, xyz + "_frequencies.dat"), dtype="float64")

                frequencies = db.VectorProperty()
                frequencies.link(properties)
                frequencies.create(structure.get_model(), "frequencies", data)
                frequencies.set_structure(structure.id())
                structure.set_property("frequencies", frequencies.id())

            compound = db.Compound()
            compound.link(compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        f = TrueMinimumFilter(structures, properties, 0.0)

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

        # Cleaning
        manager.wipe()

    def test_catalyst_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_catalyst_filter")

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())
            test_compounds[xyz] = compound

        # Setup filter
        counts = {"C": 6, "H": 10}
        f = CatalystFilter(counts, structures)

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

        # Cleaning
        manager.wipe()

    def test_atom_number_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_atom_number_filter")
        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_compound(db.ID())
            compound = db.Compound()
            compound.link(compounds)
            compound.create([structure.id()])
            structure.set_compound(compound.id())
            test_compounds[xyz] = compound

        # Setup filter
        counts = {"C": 6, "H": 10}
        f = CatalystFilter(counts, structures)

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

        # Cleaning
        manager.wipe()

    def test_one_compound_id_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_one_compound_id_filter")

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        id_list = []
        xyzs = ["water", "arginine", "cyclohexene", "h4o2"]
        for xyz in xyzs:
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.USER_OPTIMIZED)
            structure.set_compound(db.ID())
            compound = db.Compound()
            compound.link(compounds)
            compound.create([structure.id()])
            structure.set_compound(compound.id())
            test_compounds[xyz] = compound

            if xyz in ("water", "arginine"):
                id_list.append(compound.get_id())

        f = OneCompoundIDFilter(id_list)

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
        # Cleaning
        manager.wipe()

    def test_selected_compound_id_filter(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_one_compound_id_filter")

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        reactive_ids = []
        selected_ids = []
        xyzs = ["water", "arginine", "cyclohexene", "h4o2"]
        for xyz in xyzs:
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.USER_OPTIMIZED)
            structure.set_compound(db.ID())
            compound = db.Compound()
            compound.link(compounds)
            compound.create([structure.id()])
            structure.set_compound(compound.id())
            test_compounds[xyz] = compound

            if xyz in ("water", "arginine"):
                reactive_ids.append(compound.get_id())
            if xyz in ("h4o2"):
                selected_ids.append(compound.get_id())

        f = SelectedCompoundIDFilter(reactive_ids, selected_ids)

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
        # Cleaning
        manager.wipe()

    def test_and_filter_array(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_and_filter_array")

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(compounds)
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
        f = CatalystFilter(counts_two, structures) & ElementCountFilter(counts_one, structures)

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

        # Cleaning
        manager.wipe()

    def test_or_filter_array(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_or_filter_array")

        # Get collections
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        # Add structure data
        rr = resources_root_path()
        test_compounds = {}
        for xyz in ["water", "arginine", "cyclohexene"]:
            structure = db.Structure()
            structure.link(structures)
            structure.create(os.path.join(rr, xyz + ".xyz"), 0, 1)
            structure.set_label(db.Label.MINIMUM_OPTIMIZED)
            structure.set_aggregate(db.ID())
            compound = db.Compound()
            compound.link(compounds)
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
        f = CatalystFilter(counts_two, structures) | ElementCountFilter(counts_one, structures)

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

        # Cleaning
        manager.wipe()

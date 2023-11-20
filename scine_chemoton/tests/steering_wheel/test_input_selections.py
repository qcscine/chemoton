#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from typing import List
import os
import unittest

# Third party imports
import scine_database as db
import scine_utilities as utils
from scine_database import test_database_setup as db_setup

# Local application tests imports
from ..resources import resources_root_path

# Local application imports
from scine_chemoton.steering_wheel.selections.input_selections import ScineGeometryInputSelection, FileInputSelection
from scine_chemoton.steering_wheel.datastructures import SelectionResult


class FileInputTests(unittest.TestCase):

    def setUp(self) -> None:
        self.manager = db_setup.get_clean_db(f"chemoton_{self.__class__.__name__}")
        self.credentials = self.manager.get_credentials()
        self.structures = self.manager.get_collection("structures")
        rr = resources_root_path()
        self.ref0_path = os.path.join(rr, "proline_acid_propanal_product.xyz")
        self.ref1_path = os.path.join(rr, "proline_acid.xyz")
        self.ref2_path = os.path.join(rr, "propanal.xyz")
        self.model = db_setup.get_fake_model()

        self.ref0 = db.Structure(db.ID(), self.structures)
        self.ref1 = db.Structure(db.ID(), self.structures)
        self.ref2 = db.Structure(db.ID(), self.structures)
        self.ref0.create(self.ref0_path, 0, 1, self.model, db.Label.MINIMUM_OPTIMIZED)
        self.ref1.create(self.ref1_path, 0, 1, self.model, db.Label.MINIMUM_OPTIMIZED)
        self.ref2.create(self.ref2_path, 0, 1, self.model, db.Label.MINIMUM_OPTIMIZED)

        pbc = utils.PeriodicBoundaries(42.0)
        self.pmodel = db_setup.get_fake_model()
        self.pmodel.periodic_boundaries = str(pbc)

        self.pref0 = db.Structure(db.ID(), self.structures)
        self.pref1 = db.Structure(db.ID(), self.structures)
        self.pref2 = db.Structure(db.ID(), self.structures)
        self.pref0.create(self.ref0_path, 0, 1, self.pmodel, db.Label.MINIMUM_OPTIMIZED)
        self.pref1.create(self.ref1_path, 0, 1, self.pmodel, db.Label.MINIMUM_OPTIMIZED)
        self.pref2.create(self.ref2_path, 0, 1, self.pmodel, db.Label.MINIMUM_OPTIMIZED)

    def tearDown(self) -> None:
        self.manager.wipe()

    def _test_if_valid_input_and_selection(self, result: SelectionResult, expected_refs: List[db.Structure]) -> None:
        assert result.structures
        assert len(result.structures) == len(expected_refs)
        for ss in result.structures:
            structure = db.Structure(ss, self.structures)
            assert any(self._identical_structures(structure, ref) for ref in expected_refs)

    @staticmethod
    def _identical_structures(s1: db.Structure, s2: db.Structure) -> bool:
        fit = utils.QuaternionFit(s1.get_atoms().positions, s2.get_atoms().positions)
        return fit.get_rmsd() < 1e-12 \
            and s1.get_charge() == s2.get_charge() \
            and s1.get_multiplicity() == s2.get_multiplicity() \
            and s1.get_model() == s2.get_model()

    def test_empty_init_fails(self):
        with self.assertRaises(TypeError) as context:
            _ = ScineGeometryInputSelection(self.model, [])
        self.assertTrue('Received incorrect structure input' in str(context.exception))

        with self.assertRaises(TypeError) as context:
            _ = FileInputSelection(self.model, [])
        self.assertTrue('Received incorrect structure input' in str(context.exception))

    def test_wrong_init_fails(self):
        with self.assertRaises(TypeError) as context:
            _ = ScineGeometryInputSelection(self.model, self.ref0_path)
        self.assertTrue('Received incorrect structure input' in str(context.exception))

        with self.assertRaises(TypeError) as context:
            _ = ScineGeometryInputSelection(self.model, [self.ref0_path])
        self.assertTrue('Received incorrect structure input' in str(context.exception))

        with self.assertRaises(TypeError) as context:
            _ = ScineGeometryInputSelection(self.model, (self.ref0_path, 0, 1))
        self.assertTrue('Received incorrect structure input' in str(context.exception))

        with self.assertRaises(TypeError) as context:
            _ = ScineGeometryInputSelection(self.model, [(self.ref0_path, 0, 1)])
        self.assertTrue('Received incorrect structure input' in str(context.exception))

        with self.assertRaises(TypeError) as context:
            _ = ScineGeometryInputSelection(self.model, [(self.ref0_path, 0, 1), (self.ref1_path, 0, 1)])
        self.assertTrue('Received incorrect structure input' in str(context.exception))

        with self.assertRaises(TypeError) as context:
            _ = FileInputSelection(self.model, self.ref0.get_atoms())
        self.assertTrue('Received incorrect structure input' in str(context.exception))

        with self.assertRaises(TypeError) as context:
            _ = FileInputSelection(self.model, [self.ref0.get_atoms()])
        self.assertTrue('Received incorrect structure input' in str(context.exception))

        with self.assertRaises(TypeError) as context:
            _ = FileInputSelection(self.model, (self.ref0.get_atoms(), 0, 1))
        self.assertTrue('Received incorrect structure input' in str(context.exception))

        with self.assertRaises(TypeError) as context:
            _ = FileInputSelection(self.model, [(self.ref0.get_atoms(), 0, 1)])
        self.assertTrue('Received incorrect structure input' in str(context.exception))

        with self.assertRaises(TypeError) as context:
            _ = FileInputSelection(self.model, [(self.ref0.get_atoms(), 0, 1), (self.ref1.get_atoms(), 0, 1)])
        self.assertTrue('Received incorrect structure input' in str(context.exception))

    def test_result_access_fails(self):
        sele = ScineGeometryInputSelection(self.model, (self.ref0.get_atoms(), 0, 1))
        with self.assertRaises(PermissionError) as context:
            _ = sele.get_step_result()
        self.assertTrue('may not access the step_result member' in str(context.exception))

    def test_geometry_input(self):
        sele = ScineGeometryInputSelection(self.model, (self.ref0.get_atoms(), 0, 1))
        self._test_if_valid_input_and_selection(sele(self.credentials), [self.ref0])
        sele = ScineGeometryInputSelection(self.model, [(self.ref0.get_atoms(), 0, 1)])
        self._test_if_valid_input_and_selection(sele(self.credentials), [self.ref0])
        sele = ScineGeometryInputSelection(self.model, [(self.ref0.get_atoms(), 0, 1),
                                                        (self.ref1.get_atoms(), 0, 1),
                                                        (self.ref2.get_atoms(), 0, 1)])
        self._test_if_valid_input_and_selection(sele(self.credentials), [self.ref0, self.ref1, self.ref2])

        ps0 = utils.PeriodicSystem(utils.PeriodicBoundaries(self.pmodel.periodic_boundaries), self.ref0.get_atoms())
        ps1 = utils.PeriodicSystem(utils.PeriodicBoundaries(self.pmodel.periodic_boundaries), self.ref1.get_atoms())
        ps2 = utils.PeriodicSystem(utils.PeriodicBoundaries(self.pmodel.periodic_boundaries), self.ref2.get_atoms())

        sele = ScineGeometryInputSelection(self.pmodel, (ps0, 0, 1))
        self._test_if_valid_input_and_selection(sele(self.credentials), [self.pref0])
        sele = ScineGeometryInputSelection(self.pmodel, [(ps0, 0, 1)])
        self._test_if_valid_input_and_selection(sele(self.credentials), [self.pref0])
        sele = ScineGeometryInputSelection(self.pmodel, [(ps0, 0, 1), (ps1, 0, 1), (ps2, 0, 1)])
        self._test_if_valid_input_and_selection(sele(self.credentials), [self.pref0, self.pref1, self.pref2])

    def test_file_input(self):
        sele = FileInputSelection(self.model, (self.ref0_path, 0, 1))
        self._test_if_valid_input_and_selection(sele(self.credentials), [self.ref0])
        sele = FileInputSelection(self.model, [(self.ref0_path, 0, 1)])
        self._test_if_valid_input_and_selection(sele(self.credentials), [self.ref0])
        sele = FileInputSelection(self.model, [(self.ref0_path, 0, 1), (self.ref1_path, 0, 1), (self.ref2_path, 0, 1)])
        self._test_if_valid_input_and_selection(sele(self.credentials), [self.ref0, self.ref1, self.ref2])

        pbc = self.pmodel.periodic_boundaries

        sele = FileInputSelection(self.pmodel, (self.ref0_path, 0, 1, pbc))
        self._test_if_valid_input_and_selection(sele(self.credentials), [self.pref0])
        sele = FileInputSelection(self.pmodel, [(self.ref0_path, 0, 1, pbc)])
        self._test_if_valid_input_and_selection(sele(self.credentials), [self.pref0])
        sele = FileInputSelection(self.pmodel, [(self.ref0_path, 0, 1, pbc), (self.ref1_path, 0, 1, pbc),
                                                (self.ref2_path, 0, 1, pbc)])
        self._test_if_valid_input_and_selection(sele(self.credentials), [self.pref0, self.pref1, self.pref2])

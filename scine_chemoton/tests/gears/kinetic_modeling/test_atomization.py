#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import unittest
from json import dumps

# Third party imports
import scine_database as db

from ....gears import HoldsCollections
from scine_database import test_database_setup as db_setup
from ....gears.kinetic_modeling.atomization import ZeroEnergyReference, AtomEnergyReference, MultiModelEnergyReferences


class TestEnergyReferences(unittest.TestCase, HoldsCollections):
    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties", "calculations"]
        self.initialize_collections(manager)
        # pylint: disable=attribute-defined-outside-init
        self._model_a = db_setup.get_fake_model()
        self._model_b = db.Model("Second", "Fake", "Model")
        self._model_b.solvation = "MySolventModel"
        self._model_b.solvent = "fantasy"
        _, self._structure_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        self._atom_energies = {
            'O': -60.0,  # just made up numbers.
            'H': -0.5
        }
        # pylint: enable=attribute-defined-outside-init

    def tearDown(self) -> None:
        self._manager.wipe()

    def test_zero_energy_reference(self):
        manager = db_setup.get_clean_db("chemoton_zero_energy_reference")
        self.custom_setup(manager)
        ref = ZeroEnergyReference(self._model_a)
        assert ref.get_reference_energy(self._structure_id) == 0.0
        manager.wipe()

    def test_atom_energy_reference(self):
        manager = db_setup.get_clean_db("chemoton_atom_energy_reference")
        self.custom_setup(manager)
        ref = AtomEnergyReference(self._model_a, self._manager, atom_energies=self._atom_energies)
        sum_of_atoms = self._atom_energies['O'] + 2 * self._atom_energies['H']
        assert abs(ref.get_reference_energy(self._structure_id) - sum_of_atoms) < 1e-12
        manager.wipe()

    def test_atom_energy_calculation_set_up(self):
        manager = db_setup.get_clean_db("chemoton_atom_energy_calculation_reference")
        self.custom_setup(manager)
        ref = AtomEnergyReference(self._model_a, self._manager)
        ref.single_atom_multiplicities = {
            "H": 2,
            "B": 2,
            "C": 3,
            "N": 4,
            "O": 3,
            "F": 2
        }
        ref.set_up_atom_energy_calculations([self._structure_id])
        ref.set_up_atom_energy_calculations([self._structure_id])  # multiple calls should not do anything extra
        assert self._calculations.count(dumps({})) == 2
        assert self._structures.count(dumps({})) == 3
        for calculation in self._calculations.iterate_all_calculations():
            calculation.link(self._calculations)
            assert calculation.get_model() == self._model_a

        for structure in self._structures.iterate_all_structures():
            if structure.id() == self._structure_id:
                continue
            structure.link(self._structures)
            atoms = structure.get_atoms()
            assert len(atoms) == 1
            compound = db.Compound(db.ID(), self._compounds)
            compound.create([structure.id()])
            structure.set_aggregate(compound.id())

        assert self._compounds.count(dumps({})) == 3

        ref_with_other_model = AtomEnergyReference(self._model_b, self._manager)
        ref_with_other_model.set_up_atom_energy_calculations([self._structure_id])
        assert self._calculations.count(dumps({})) == 4
        assert self._structures.count(dumps({})) == 3
        manager.wipe()

    def test_multi_model_energy_references(self):
        manager = db_setup.get_clean_db("chemoton_multi_model_energy_reference")
        self.custom_setup(manager)
        ref_atom = AtomEnergyReference(self._model_a, self._manager, atom_energies=self._atom_energies)
        ref_zero = ZeroEnergyReference(self._model_b)
        multi_model_reference = MultiModelEnergyReferences([ref_atom, ref_zero])
        assert multi_model_reference.get_energy_reference(self._model_a) == ref_atom
        assert multi_model_reference.get_energy_reference(self._model_b) == ref_zero
        manager.wipe()

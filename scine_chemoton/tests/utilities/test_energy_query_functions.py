#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import unittest

import scine_database as db
from .. import test_database_setup as db_setup
from ...utilities.energy_query_functions import (
    get_min_free_energy_for_aggregate
)


class EnergyQueryFunctionsTest(unittest.TestCase):

    @staticmethod
    def add_energy(energy, label, structure, model, properties):
        energy_prop = db.NumberProperty.make(label, model, energy, properties)
        energy_prop.set_structure(structure.id())
        structure.add_property(label, energy_prop.id())
        return energy_prop

    @staticmethod
    def add_structure(compound, structures, atoms=None):
        if atoms is None:
            atoms = db.Structure(compound.get_centroid(), structures).get_atoms()
        s = db.Structure()
        s.link(structures)
        s.create(atoms, 0, 1)
        compound.add_structure(s.id())
        return s

    def test_get_min_free_energy_for_aggregate(self):
        """
        Idea of the test: Add electronic energies and gibbs energy corrections with different models.
        Check whether the correct energy is fetched upon calling the energy getter function.
        """
        manager = db_setup.get_clean_db("chemoton_test_get_min_free_energy_for_aggregate")
        structures = manager.get_collection("structures")
        compounds = manager.get_collection("compounds")
        properties = manager.get_collection("properties")

        energies = [0.0, 1.2, 3.4, -1.5, -100.9, -99.9]
        gibbs_correction_1 = [0.0, 0.0, 0.0, 0.0, +10.0, 1.0]
        gibbs_correction_2 = [0.0, 0.0, 0.0, 0.0, +10.0, +10.0]
        a_id, s_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        aggregate = db.Compound(a_id, compounds)
        e_model = db.Model("Electronic", "Energy", "Model")
        g_model = db.Model("Gibbs", "Correction", "Model")
        s0 = db.Structure(s_id, structures)
        e_label = "electronic_energy"
        g_label = "gibbs_energy_correction"
        full_label = "gibbs_free_energy"
        self.add_energy(0.0, e_label, s0, e_model, properties)
        self.add_energy(0.0, g_label, s0, e_model, properties)
        self.add_energy(0.0, g_label, s0, g_model, properties)
        self.add_energy(0.0, full_label, s0, e_model, properties)

        assert len(energies) == len(gibbs_correction_1)
        assert len(energies) == len(gibbs_correction_2)
        for e, g1, g2 in zip(energies, gibbs_correction_1, gibbs_correction_2):
            s = self.add_structure(aggregate, structures)
            self.add_energy(e, e_label, s, e_model, properties)
            self.add_energy(g1, g_label, s, e_model, properties)
            self.add_energy(g2, g_label, s, g_model, properties)
            self.add_energy(e + g1, full_label, s, e_model, properties)

        min_same_model = get_min_free_energy_for_aggregate(aggregate, e_model, e_model, structures, properties)
        assert min_same_model is not None
        assert abs(min_same_model - (energies[-1] + gibbs_correction_1[-1])) < 1e-12
        min_different_models = get_min_free_energy_for_aggregate(aggregate, e_model, g_model, structures, properties)
        assert min_different_models is not None
        assert abs(min_different_models
                   - (energies[len(energies) - 2] + gibbs_correction_2[len(gibbs_correction_2) - 2])) < 1e-12

        manager.wipe()

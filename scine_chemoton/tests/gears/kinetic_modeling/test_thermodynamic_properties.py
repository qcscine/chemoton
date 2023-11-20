#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
import unittest
import numpy as np

# Third party imports
import scine_database as db

# Local application tests imports
from scine_database import test_database_setup as db_setup
from ...resources import resources_root_path
from ....gears import HoldsCollections

# Local application imports
from ....gears.kinetic_modeling.thermodynamic_properties import (
    ThermodynamicProperties,
    ReferenceState,
    ThermodynamicPropertiesCache
)
from ....gears.kinetic_modeling.aggregate_wrapper import AggregateCache


class TestThermodynamicProperties(unittest.TestCase, HoldsCollections):
    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties", "calculations"]
        self.initialize_collections(manager)

        # pylint: disable=attribute-defined-outside-init
        self.electronic_energy = -75.64001590264
        self.model = db.Model("FAKE", "FAKE", "F-AKE")
        # References from Turbomole job.
        self.ref = ReferenceState(298.15, 101325.0, 1)
        self.reference_free_energy = -75.6488462958007
        self.reference_correction = -0.00883039316073564
        self.reference_zero_point_energy_correction = 0.007480021854688685
        # pylint: enable=attribute-defined-outside-init

    def tearDown(self) -> None:
        self._manager.wipe()

    @staticmethod
    def get_OH_hessian(model: db.Model, properties: db.Collection) -> db.DenseMatrixProperty:
        hessian = [[0.0098411811, -0.0204618598, 0.0573693912, -0.0098411811, 0.0204618598, -0.0573693912],
                   [-0.0204618598, 0.0425444569, -0.1192828812, 0.0204618598, -0.0425444569, 0.1192828812],
                   [0.0573693912, -0.1192828812, 0.3344361818, -0.0573693912, 0.1192828812, -0.3344361818],
                   [-0.0098411811, 0.0204618598, -0.0573693912, 0.0098411811, -0.0204618598, 0.0573693912],
                   [0.0204618598, -0.0425444569, 0.1192828812, -0.0204618598, 0.0425444569, -0.1192828812],
                   [-0.0573693912, 0.1192828812, -0.3344361818, 0.0573693912, -0.1192828812, 0.3344361818]]
        hessian_property = db.DenseMatrixProperty()
        hessian_property.link(properties)
        hessian_property.create(model, "hessian", np.asarray(hessian))
        return hessian_property

    @staticmethod
    def get_OH_structures(model: db.Model, energy: float, structures: db.Collection, properties: db.Collection):
        structure = db.Structure()  # type ignore
        structure.link(structures)
        structure.create(os.path.join(resources_root_path(), "hydroxide.xyz"), -1, 1)
        structure.set_label(db.Label.USER_OPTIMIZED)
        energy_property = db.NumberProperty()
        energy_property.link(properties)
        energy_property.create(model, "electronic_energy", energy)
        structure.add_property("electronic_energy", energy_property.id())
        energy_property.set_structure(structure.id())
        return structure, energy_property

    def set_up_compound(self):
        # pylint: disable=attribute-defined-outside-init
        self.structure, energy_property = self.get_OH_structures(self.model, self.electronic_energy, self._structures,
                                                                 self._properties)
        self.compound = db.Compound(db.ID(), self._compounds)  # type ignore
        self.compound.create([self.structure.id()])
        self.structure.set_aggregate(self.compound.id())

        hessian_property = self.get_OH_hessian(self.model, self._properties)
        self.structure.add_property("hessian", hessian_property.id())
        hessian_property.set_structure(self.structure.id())

        self.hessian_property = hessian_property  # type ignore
        self.energy_property = energy_property  # type ignore
        # pylint: enable=attribute-defined-outside-init

    def test_thermodynamic_properties(self):
        manager = db_setup.get_clean_db("chemoton_thermo_properties")
        self.custom_setup(manager)
        self.set_up_compound()

        therm = ThermodynamicProperties(self.hessian_property.id(), self.energy_property.id(), self._properties,
                                        self._structures)

        assert abs(therm.get_zero_point_energy_correction() - self.reference_zero_point_energy_correction) < 1e-6
        assert abs(therm.get_electronic_energy() - self.electronic_energy) < 1e-6
        assert abs(therm.get_gibbs_free_energy_correction(self.ref) - self.reference_correction) < 1e-6
        assert abs(therm.get_gibbs_free_energy(self.ref) - self.reference_free_energy) < 1e-6
        assert therm.get_reference_state() == self.ref

        cache = ThermodynamicPropertiesCache(self._structures, self._properties, self.model, self.model)
        cache.get_or_produce(self.structure.id())
        assert cache.minimum_values_need_update(self.ref, 1)
        assert abs(therm.get_gibbs_free_energy(self.ref) - cache.get_ensemble_gibbs_free_energy(self.ref, 1)) < 1e-9
        assert not cache.minimum_values_need_update(self.ref, 1)
        h = cache.get_ensemble_enthalpy(self.ref)
        s = cache.get_ensemble_entropy(self.ref)
        g = h - self.ref.temperature * s
        assert abs(g - cache.get_ensemble_gibbs_free_energy(self.ref)) < 1e-9
        cache.clear()
        assert cache.minimum_values_need_update(self.ref, 1)

    def test_aggregate_wrapper(self):
        manager = db_setup.get_clean_db("chemoton_aggregate_wrapper")
        self.custom_setup(manager)
        self.set_up_compound()

        aggregate_cache = AggregateCache(manager, self.model, self.model)
        aggregate = aggregate_cache.get_or_produce(self.compound.id())
        assert abs(aggregate.get_free_energy(self.ref) - self.reference_free_energy) < 1e-7
        test = aggregate_cache.get_or_produce(self.compound.id())
        assert test == aggregate
        assert aggregate.get_db_id() == self.compound.id()
        assert aggregate.get_aggregate_type() == db.CompoundOrFlask.COMPOUND
        assert aggregate.get_concentration_flux() == 0.0
        assert aggregate.get_starting_concentration() == 0.0

        other_energy = self.electronic_energy - 0.001
        for _ in range(8):
            structure = db.Structure()
            structure.link(self._structures)
            structure.create(os.path.join(resources_root_path(), "hydroxide.xyz"), -1, 1)
            structure.set_label(db.Label.USER_OPTIMIZED)
            self.compound.add_structure(structure.id())
            structure.set_aggregate(self.compound.id())
            energy_property = db.NumberProperty()
            energy_property.link(self._properties)
            energy_property.create(self.model, "electronic_energy", other_energy)
            structure.add_property("electronic_energy", energy_property.id())
            energy_property.set_structure(structure.id())

        assert abs(aggregate.get_free_energy(self.ref) - self.reference_free_energy) < 1e-7  # lowest free energy
        new_cache = AggregateCache(manager, self.model, self.model, only_electronic=True)
        new_aggregate = new_cache.get_or_produce(self.compound.id())
        assert abs(new_aggregate.get_free_energy(self.ref) - other_energy) < 1e-9  # lowest electronic energy

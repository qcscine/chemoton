#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import unittest
import numpy as np
import os

# Third party imports
import scine_database as db
import scine_utilities as utils
from scine_database import test_database_setup as db_setup

# Local application tests imports
from ....gears import HoldsCollections
from ...resources import resources_root_path

# Local application imports
from ....gears.kinetic_modeling.reaction_wrapper import Reaction
from ....gears.kinetic_modeling.thermodynamic_properties import ReferenceState


class TestReactionWrapper(unittest.TestCase, HoldsCollections):
    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties", "calculations"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    def test_reaction_wrapper(self):
        n_compounds = 10
        n_flasks = 3
        n_reactions = 10
        max_r_per_c = 10
        max_n_products_per_r = 2
        max_n_educts_per_r = 2
        max_s_per_c = 1
        max_steps_per_r = 1
        barrier_limits = (10, 20)
        n_inserts = 3
        manager = db_setup.get_random_db(
            n_compounds,
            n_flasks,
            n_reactions,
            max_r_per_c,
            "chemoton_test_reaction_wrapper",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)
        model = db.Model("FAKE", "FAKE", "F-AKE")
        ref = ReferenceState(298.15, 1e+5)
        hessian = np.zeros((9, 9))

        for structure in self._structures.iterate_all_structures():
            structure.link(self._structures)
            hessian_property = db.DenseMatrixProperty()
            hessian_property.link(self._properties)
            hessian_property.create(model, "hessian", hessian)
            structure.add_property("hessian", hessian_property.id())
            hessian_property.set_structure(structure.id())

        for reaction in self._reactions.iterate_all_reactions():
            reaction_wrapper = Reaction(reaction.id(), manager, model, model)
            # Check for NaN and inf
            g = reaction_wrapper.get_free_energy_of_activation(ref)
            k = reaction_wrapper.get_ts_theory_rate_constants(ref)
            r = reaction_wrapper.get_reaction_free_energy(ref)
            assert abs(g[0] - g[0]) < 1e-9
            assert abs(g[1] - g[1]) < 1e-9
            assert abs(k[0] - k[0]) < 1e-9
            assert abs(k[1] - k[1]) < 1e-9
            assert abs(r - r) < 1e-9

    def test_deterministic_reaction(self):
        manager = db_setup.get_clean_db("chemoton_test_reaction_wrapper")
        self.custom_setup(manager)
        structures = manager.get_collection("structures")
        properties = manager.get_collection("properties")
        elementary_steps = manager.get_collection("elementary_steps")
        reactions = manager.get_collection("reactions")

        f_id_1, s_id_1 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_OPTIMIZED)
        f_id_2, s_id_2 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_OPTIMIZED)
        s_1 = db.Structure(s_id_1, structures)
        s_2 = db.Structure(s_id_2, structures)
        e1 = -634.6730820353
        e2 = -634.6568134574
        ets = -634.6145146309
        lhs_barrier = ets - e1
        rhs_barrier = ets - e2
        rxn_energy = e2 - e1
        ref = ReferenceState(298.15, 1e+5)

        db_setup.add_random_energy(s_1, (e1 / utils.HARTREE_PER_KJPERMOL, e1 / utils.HARTREE_PER_KJPERMOL), properties)
        db_setup.add_random_energy(s_2, (e2 / utils.HARTREE_PER_KJPERMOL, e2 / utils.HARTREE_PER_KJPERMOL), properties)

        model = db_setup.get_fake_model()
        ts = db.Structure()
        ts.link(structures)
        ts.create(os.path.join(resources_root_path(), "water.xyz"), 0, 1)
        ts.set_label(db.Label.TS_OPTIMIZED)
        ts.set_model(model)
        db_setup.add_random_energy(ts, (ets / utils.HARTREE_PER_KJPERMOL, ets / utils.HARTREE_PER_KJPERMOL), properties)

        new_step = db.ElementaryStep()
        new_step.link(elementary_steps)
        new_step.create([s_id_1], [s_id_2])
        new_step.set_transition_state(ts.id())
        new_reaction = db.Reaction()
        new_reaction.link(reactions)
        new_reaction.create([f_id_1], [f_id_2], [db.CompoundOrFlask.FLASK], [db.CompoundOrFlask.FLASK])
        new_reaction.add_elementary_step(new_step.id())

        wrapper = Reaction(new_reaction.id(), manager, model, model, only_electronic=True)
        assert abs(wrapper.get_reaction_free_energy(ref) - rxn_energy) < 1e-9
        assert abs(wrapper.get_free_energy_of_activation(ref)[0] - lhs_barrier) < 1e-9
        assert abs(wrapper.get_free_energy_of_activation(ref)[1] - rhs_barrier) < 1e-9

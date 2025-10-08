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
from ....utilities.db_object_wrappers.reaction_wrapper import Reaction
from ....utilities.db_object_wrappers.wrapper_caches import MultiModelReactionCache, MultiModelCacheFactory
from ....utilities.db_object_wrappers.thermodynamic_properties import ReferenceState
from ....utilities.model_combinations import ModelCombination


def add_reaction(manager: db.Manager, e1, e2, ets):
    structures = manager.get_collection("structures")
    elementary_steps = manager.get_collection("elementary_steps")
    reactions = manager.get_collection("reactions")
    properties = manager.get_collection("properties")

    f_id_1, s_id_1 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_OPTIMIZED)
    f_id_2, s_id_2 = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.COMPLEX_OPTIMIZED)
    s_1 = db.Structure(s_id_1, structures)
    s_2 = db.Structure(s_id_2, structures)

    db_setup.add_random_energy(s_1, (e1 / utils.HARTREE_PER_KJPERMOL, e1 / utils.HARTREE_PER_KJPERMOL),
                               properties)
    db_setup.add_random_energy(s_2, (e2 / utils.HARTREE_PER_KJPERMOL, e2 / utils.HARTREE_PER_KJPERMOL),
                               properties)

    model = db_setup.get_fake_model()
    ts = db.Structure()
    ts.link(structures)
    ts.create(os.path.join(resources_root_path(), "water.xyz"), 0, 1)
    ts.set_label(db.Label.TS_OPTIMIZED)
    ts.set_model(model)
    db_setup.add_random_energy(ts, (ets / utils.HARTREE_PER_KJPERMOL, ets / utils.HARTREE_PER_KJPERMOL),
                               properties)

    new_step = db.ElementaryStep()
    new_step.link(elementary_steps)
    new_step.create([s_id_1], [s_id_2])
    new_step.set_transition_state(ts.id())
    new_reaction = db.Reaction()
    new_reaction.link(reactions)
    new_reaction.create([f_id_1], [f_id_2], [db.CompoundOrFlask.FLASK], [db.CompoundOrFlask.FLASK])
    new_reaction.add_elementary_step(new_step.id())
    return new_reaction, s_1, s_2, ts


class TestReactionWrapper(unittest.TestCase, HoldsCollections):
    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "reactions", "compounds", "flasks",
                                      "properties", "calculations"]
        self.initialize_collections(manager)
        MultiModelCacheFactory().clear()

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
        water_hessian = np.asarray([
            [0.36647218415, -0.19686744467, 0.075679967953, -0.32218740014, 0.20925410358,
             -0.021533989199, -0.044239967603, -0.012588937262, -0.054298790044],
            [-0.19690008026, 0.32717966817, 0.2349580946, 0.24190692867, -0.17611641081,
             -0.0073972973886, -0.045127077254, -0.15110334448, -0.22770579793],
            [0.075668504675, 0.23499423498, 0.3590979212, 0.019125469482, -0.035995673283,
             -0.028100877642, -0.09485785008, -0.19911075354, -0.33112480642],
            [-0.32222889145, 0.24200032545, 0.019194331212, 0.32225650417, -0.23487938382,
             -0.010313672577, 2.3432972687e-05, -0.0070332340347, -0.0088115815734],
            [0.2093629349, -0.17616252381, -0.03607348316, -0.23487823759, 0.1931001407,
             0.034721341318, 0.02556546834, -0.016969326212, 0.0012704472369],
            [-0.021468156598, -0.0074726773919, -0.028088096172, -0.010313982661, 0.034721696195,
             0.034212862852, 0.031807114724, -0.027327793528, -0.0061217826359],
            [-0.044339713113, -0.045017321907, -0.094779378773, 2.3412011546e-05, 0.025565775972,
             0.03180712902, 0.044220438895, 0.019566115664, 0.063055983244],
            [-0.01246207735, -0.1510452638, -0.19888745662, -0.0070339450406, -0.016968623427,
             -0.027327147494, 0.019566084555, 0.16808568367, 0.22644129844],
            [-0.054204818519, -0.22752122314, -0.33102435314, -0.0088120623893, 0.0012707943262,
             -0.0061214385116, 0.063055780638, 0.22644139438, 0.3372705704]
        ])

        for structure in self._structures.iterate_all_structures():
            structure.link(self._structures)
            hessian_property = db.DenseMatrixProperty()
            hessian_property.link(self._properties)
            hessian_property.create(model, "hessian", water_hessian)
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

            if not reaction_wrapper.barrierless(ref) and (len(reaction_wrapper.get_lhs_aggregates()) == 1
                                                          or len(reaction_wrapper.get_rhs_aggregates()) == 1):
                zero_energy = reaction_wrapper.get_transition_state_free_energy(utils.vacuum_zero_kelvin())
                micro_kf, micro_kb = reaction_wrapper.get_rrkm_rate_constants(
                    np.asarray([zero_energy + 0.001]), active_rotors=True, reference_state=ref)
                if micro_kb is not None:
                    assert abs(micro_kb[0] - micro_kb[0]) < 1e-9
                if micro_kf is not None:
                    assert abs(micro_kf[0] - micro_kf[0]) < 1e-9
            if reaction_wrapper.barrierless(ref):
                assert reaction_wrapper.get_transition_state_wavenumber(ref) is None
            else:
                wavenumber = reaction_wrapper.get_transition_state_wavenumber(ref)
                reference_wavenumber = 1157.0369229397152
                assert abs(wavenumber - reference_wavenumber) < 1e-2

    def test_deterministic_reaction(self):
        manager = db_setup.get_clean_db("chemoton_test_reaction_wrapper")
        self.custom_setup(manager)

        e1 = -634.6730820353
        e2 = -634.6568134574
        ets = -634.6145146309
        new_reaction, s_1, s_2, ts = add_reaction(manager, e1, e2, ets)
        lhs_barrier = ets - e1
        rhs_barrier = ets - e2
        rxn_energy = e2 - e1
        ref = ReferenceState(298.15, 1e+5)

        model = db_setup.get_fake_model()
        model_combi = ModelCombination(model, model)

        wrapper = Reaction(new_reaction.id(), manager, model, model, only_electronic=True)
        assert abs(wrapper.get_reaction_free_energy(ref) - rxn_energy) < 1e-9
        assert abs(wrapper.get_free_energy_of_activation(ref)[0] - lhs_barrier) < 1e-9
        assert abs(wrapper.get_free_energy_of_activation(ref)[1] - rhs_barrier) < 1e-9

        # Check if the correct values are returned if multiple models are provided (with different orders).
        other_model = db.Model("other", "fake", "model")
        other_combi = ModelCombination(other_model, other_model)
        e3 = e1 + 0.001
        e4 = e2 + 0.004
        ets2 = ets + 0.006
        lhs_barrier2 = ets2 - e3
        rhs_barrier2 = ets2 - e4
        rxn_energy2 = e4 - e3
        db_setup.add_random_energy(s_1, (e3 / utils.HARTREE_PER_KJPERMOL, e3 / utils.HARTREE_PER_KJPERMOL),
                                   self._properties, other_model)
        db_setup.add_random_energy(s_2, (e4 / utils.HARTREE_PER_KJPERMOL, e4 / utils.HARTREE_PER_KJPERMOL),
                                   self._properties, other_model)
        db_setup.add_random_energy(ts, (ets2 / utils.HARTREE_PER_KJPERMOL, ets2 / utils.HARTREE_PER_KJPERMOL),
                                   self._properties, other_model)
        multi_cache = MultiModelReactionCache(manager, [other_combi, model_combi], only_electronic=True)
        other_wrapper = multi_cache.get_or_produce(new_reaction.id())
        assert abs(other_wrapper.get_reaction_free_energy(ref) - rxn_energy2) < 1e-9
        assert abs(other_wrapper.get_free_energy_of_activation(ref)[0] - lhs_barrier2) < 1e-9
        assert abs(other_wrapper.get_free_energy_of_activation(ref)[1] - rhs_barrier2) < 1e-9

        third_model = db.Model("third", "fake", "model")
        third_combi = ModelCombination(third_model, third_model)
        e5 = e3 * 0.98
        e6 = e4 * 1.01
        ets3 = ets2 * 0.95
        lhs_barrier3 = ets3 - e5
        rhs_barrier3 = ets3 - e6
        rxn_energy3 = e6 - e5
        db_setup.add_random_energy(s_1, (e5 / utils.HARTREE_PER_KJPERMOL, e5 / utils.HARTREE_PER_KJPERMOL),
                                   self._properties, third_model)
        multi_cache = MultiModelReactionCache(manager, [third_combi, other_combi, model_combi], only_electronic=True)
        other_wrapper = multi_cache.get_or_produce(new_reaction.id())
        # There should be no values available from third_model. Therefore, the values for other_model should be used.
        assert abs(other_wrapper.get_reaction_free_energy(ref) - rxn_energy2) < 1e-9
        assert abs(other_wrapper.get_free_energy_of_activation(ref)[0] - lhs_barrier2) < 1e-9
        assert abs(other_wrapper.get_free_energy_of_activation(ref)[1] - rhs_barrier2) < 1e-9

        db_setup.add_random_energy(s_2, (e6 / utils.HARTREE_PER_KJPERMOL, e6 / utils.HARTREE_PER_KJPERMOL),
                                   self._properties, third_model)
        other_wrapper = multi_cache.get_or_produce(new_reaction.id())
        # There still be no values available from third_model. Therefore, the values for other_model should be used.
        assert abs(other_wrapper.get_reaction_free_energy(ref) - rxn_energy2) < 1e-9
        assert abs(other_wrapper.get_free_energy_of_activation(ref)[0] - lhs_barrier2) < 1e-9
        assert abs(other_wrapper.get_free_energy_of_activation(ref)[1] - rhs_barrier2) < 1e-9

        db_setup.add_random_energy(ts, (ets3 / utils.HARTREE_PER_KJPERMOL, ets3 / utils.HARTREE_PER_KJPERMOL),
                                   self._properties, third_model)
        other_wrapper = multi_cache.get_or_produce(new_reaction.id())
        # Now, there should be values available from third_model.
        assert abs(other_wrapper.get_reaction_free_energy(ref) - rxn_energy3) < 1e-9
        assert abs(other_wrapper.get_free_energy_of_activation(ref)[0] - lhs_barrier3) < 1e-9
        assert abs(other_wrapper.get_free_energy_of_activation(ref)[1] - rhs_barrier3) < 1e-9

        multi_cache = MultiModelReactionCache(manager, [model_combi, other_combi, third_combi], only_electronic=True)
        # Use values for the first model.
        other_wrapper = multi_cache.get_or_produce(new_reaction.id())
        assert abs(other_wrapper.get_reaction_free_energy(ref) - rxn_energy) < 1e-9
        assert abs(other_wrapper.get_free_energy_of_activation(ref)[0] - lhs_barrier) < 1e-9
        assert abs(other_wrapper.get_free_energy_of_activation(ref)[1] - rhs_barrier) < 1e-9

        fourth_model = db.Model("fourth", "fake", "model")
        high_energy = 0.9 * e1
        low_energy = e1
        diff = low_energy - high_energy
        db_setup.add_random_energy(s_1, (high_energy / utils.HARTREE_PER_KJPERMOL,
                                         high_energy / utils.HARTREE_PER_KJPERMOL), self._properties, fourth_model)
        db_setup.add_random_energy(s_2, (low_energy / utils.HARTREE_PER_KJPERMOL,
                                         low_energy / utils.HARTREE_PER_KJPERMOL), self._properties, fourth_model)
        db_setup.add_random_energy(ts, (low_energy / utils.HARTREE_PER_KJPERMOL,
                                        low_energy / utils.HARTREE_PER_KJPERMOL), self._properties, fourth_model)

        other_wrapper = Reaction(new_reaction.id(), manager, fourth_model, fourth_model, only_electronic=True)
        assert abs(other_wrapper.get_reaction_free_energy(ref) - diff) < 1e-9
        assert abs(other_wrapper.get_free_energy_of_activation(ref)[0]) < 1e-9
        assert abs(other_wrapper.get_free_energy_of_activation(ref)[1] + diff) < 1e-9

    def test_reaction_analysis_flag(self):
        manager = db_setup.get_clean_db("chemoton_test_test_reaction_analysis_flag")
        self.custom_setup(manager)
        e1 = -76.847
        e2 = -76.849
        ets = -76.845
        new_reaction, _, _, _ = add_reaction(manager, e1, e2, ets)
        model = db_setup.get_fake_model()
        wrapper = Reaction(new_reaction.id(), manager, model, model, only_electronic=True)
        assert wrapper.analyze()
        assert wrapper.explore()
        new_reaction.disable_analysis()
        assert not wrapper.analyze()
        assert wrapper.explore()
        new_reaction.disable_exploration()
        assert not wrapper.explore()
        new_reaction.enable_exploration()
        new_reaction.enable_analysis()
        aggregate = wrapper.get_lhs_aggregates()[0]
        aggregate.get_db_object().disable_analysis()
        assert not aggregate.analyze()
        assert not wrapper.analyze()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import unittest

import numpy as np
import pytest
from random import random

import scine_database as db
import scine_utilities as utils
from scine_database import test_database_setup as db_setup

from ...gears import HoldsCollections
from ...utilities.uncertainties import (
    ModelCombinationBasedUncertaintyEstimator,
    ConstantUncertainty,
    AtomWiseUncertainty,
    StandardDeviationUncertainty
)
from ...utilities.model_combinations import ModelCombination
from ...utilities.db_object_wrappers.aggregate_cache import AggregateCache
from ...utilities.db_object_wrappers.reaction_cache import ReactionCache


class TestReactionWrapper(unittest.TestCase, HoldsCollections):
    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    def test_model_based_uncertainties(self):
        manager = db_setup.get_clean_db("chemoton_test_model_based_uncertainties")
        self.custom_setup(manager)
        a_id, _ = db_setup.insert_single_empty_structure_aggregate(self._manager, db.Label.COMPLEX_OPTIMIZED)
        model_combination_1 = ModelCombination(db_setup.get_fake_model())
        model_combination_2 = ModelCombination(db.Model("some", "other", "model"))
        model_combination_3 = ModelCombination(db.Model("some", "third", "model"))
        cache = AggregateCache(self._manager, model_combination_1.electronic_model, model_combination_1.hessian_model,
                               only_electronic=True)
        cache2 = AggregateCache(self._manager, model_combination_2.electronic_model, model_combination_2.hessian_model,
                                only_electronic=True)
        cache3 = AggregateCache(self._manager, model_combination_3.electronic_model, model_combination_3.hessian_model,
                                only_electronic=True)
        m1_lower = 500.0
        m1_upper = 300.0
        m2_lower = 700.0
        m2_upper = 1300.0
        uq_1 = ConstantUncertainty((m1_lower, m1_upper))
        uq_2 = ConstantUncertainty((m2_lower, m2_upper))
        estimator = ModelCombinationBasedUncertaintyEstimator(
            [(model_combination_2, uq_2, uq_2), (model_combination_1, uq_1, uq_1)])
        a1 = cache.get_or_produce(a_id)
        uq = estimator.get_uncertainty(a1)
        assert (uq.lower(a1) - m1_lower) < 1e-12
        assert (uq.upper(a1) - m1_upper) < 1e-12
        a2 = cache2.get_or_produce(a_id)
        uq = estimator.get_uncertainty(a2)
        assert (uq.lower(a2) - m2_lower) < 1e-12
        assert (uq.upper(a2) - m2_upper) < 1e-12

        with pytest.raises(RuntimeError):
            estimator.get_uncertainty(cache3.get_or_produce(a_id))

    def test_atom_wise_uncertainty(self):
        manager = db_setup.get_clean_db("chemoton_test_atom_wise_uncertainty")
        self.custom_setup(manager)
        a_id, _ = db_setup.insert_single_empty_structure_aggregate(self._manager, db.Label.COMPLEX_OPTIMIZED)
        model_combination_1 = ModelCombination(db_setup.get_fake_model())
        cache = AggregateCache(self._manager, model_combination_1.electronic_model, model_combination_1.hessian_model,
                               only_electronic=True)
        aggregate = cache.get_or_produce(a_id)
        uq_h_l = 100.0
        uq_h_u = 50.0
        uq_o_l = 200.0
        uq_o_u = 100.0
        uq_1 = AtomWiseUncertainty(atom_wise_uncertainties={"H": (uq_h_l, uq_h_u),
                                                            "O": (uq_o_l, uq_o_u)})
        assert abs(uq_1.lower(aggregate) - (2 * uq_h_l + uq_o_l)) < 1e-9
        assert abs(uq_1.upper(aggregate) - (2 * uq_h_u + uq_o_u)) < 1e-9

    def test_standard_deviation_uncertainty(self):
        from ..gears.kinetic_modeling.test_kinetic_modeling import add_reaction
        manager = db_setup.get_clean_db("chemoton_test_standard_deviation_uncertainty")
        self.custom_setup(manager)
        elementary_steps = manager.get_collection("elementary_steps")
        structures = manager.get_collection("structures")
        properties = manager.get_collection("properties")
        n_energies = 5
        fall_back = 10e+3
        models = [db.Model("one", "model", str(i)) for i in range(n_energies)]
        combs = [ModelCombination(models[-1])]

        other_models = [db.Model("other", "model", str(i)) for i in range(n_energies)]
        lhs_energies = [0.0 for _ in models]
        rhs_energies = [random() * 1e-3 for _ in models]
        ts_energies = [rhs_energy + random() * 1e-3 for rhs_energy in rhs_energies]
        standard_deviation = np.std(np.asarray(ts_energies) * utils.KJPERMOL_PER_HARTREE * 1e+3)
        reaction, lhs, rhs = add_reaction(manager, ts_energies[-1], [lhs_energies[-1]], [rhs_energies[-1]],
                                          model=models[-1])
        initial_ts = db.Structure(db.ElementaryStep(reaction.get_elementary_steps()[0], elementary_steps)
                                  .get_transition_state(), structures)
        lhs_structure_ids = [a.get_centroid() for a in lhs]
        rhs_structure_ids = [a.get_centroid() for a in rhs]
        for i, (model, ets, elhs, erhs) in enumerate(zip(models, ts_energies, lhs_energies, rhs_energies)):
            if i == len(models) - 1:
                break
            combs.append(ModelCombination(model, models[-1]))
            all_structures = lhs_structure_ids + rhs_structure_ids + [initial_ts.id()]
            all_energies = [elhs, erhs, ets]
            for s_id, e in zip(all_structures, all_energies):
                e = e * utils.KJPERMOL_PER_HARTREE
                db_setup.add_random_energy(db.Structure(s_id, structures), (e, e), properties, model=model)

        uq = StandardDeviationUncertainty(combs, fall_back)
        cache = ReactionCache(manager, models[0], models[-1])
        wrapper = cache.get_or_produce(reaction.id())
        assert abs(uq.lower(wrapper) - standard_deviation) < 1e-2  # value in J/mol
        assert abs(uq.upper(wrapper) - standard_deviation) < 1e-2  # value in J/mol

        uq = StandardDeviationUncertainty([ModelCombination(m) for m in other_models], fall_back)
        assert abs(uq.lower(wrapper) - fall_back) < 1e-9

        minimum = 1e+4
        uq = StandardDeviationUncertainty(combs, fall_back, minimum_uncertainty=minimum)
        assert uq.lower(wrapper) >= minimum  # value in J/mol
        assert uq.upper(wrapper) >= minimum  # value in J/mol

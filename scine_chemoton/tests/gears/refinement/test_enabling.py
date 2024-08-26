#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import unittest
from typing import Union

# Third party imports
import scine_database as db

# Local application tests imports
from scine_database import test_database_setup as db_setup
from ....gears import HoldsCollections
from ....gears.network_refinement.enabling import (
    EnableAllAggregates,
    AggregateEnabling,
    EnableAllReactions,
    ReactionEnabling,
    EnableAllStructures,
    EnableAllSteps,
    EnableCalculationResults,
    EnableJobSpecificCalculations,
    ApplyToAllStepsInReaction,
    ApplyToAllStructuresInAggregate,
    StructureEnabling,
    EnableStructureByModel,
    FilteredStepEnabling
)
from scine_chemoton.filters.elementary_step_filters import ConsistentEnergyModelFilter


class TestEnabling(unittest.TestCase, HoldsCollections):
    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "reactions", "calculations", "compounds", "flasks",
                                      "structures", "properties"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    def set_up_database(self):
        n_compounds = 10
        n_flasks = 6
        n_reactions = 16
        max_r_per_c = 10
        max_n_products_per_r = 3
        max_n_educts_per_r = 2
        max_s_per_c = 3
        max_steps_per_r = 1
        barrier_limits = (0.1, 100.0)
        n_inserts = 3
        manager = db_setup.get_random_db(
            n_compounds,
            n_flasks,
            n_reactions,
            max_r_per_c,
            "chemoton_test_enabling",
            max_n_products_per_r,
            max_n_educts_per_r,
            max_s_per_c,
            max_steps_per_r,
            barrier_limits,
            n_inserts,
        )
        self.custom_setup(manager)
        self.disable_all()

    def disable_all(self):
        for a in self._compounds.iterate_all_compounds():
            a.link(self._compounds)
            a.disable_exploration()
            a.disable_analysis()
        for a in self._flasks.iterate_all_flasks():
            a.link(self._flasks)
            a.disable_exploration()
            a.disable_analysis()
        for a in self._structures.iterate_all_structures():
            a.link(self._structures)
            a.disable_exploration()
            a.disable_analysis()
        for a in self._elementary_steps.iterate_all_elementary_steps():
            a.link(self._elementary_steps)
            a.disable_exploration()
            a.disable_analysis()
        for a in self._reactions.iterate_all_reactions():
            a.link(self._reactions)
            a.disable_exploration()
            a.disable_analysis()

    @staticmethod
    def assert_is_enabled(ob, expected_result: bool = True):
        assert ob.exists()
        assert ob.explore() is expected_result
        assert ob.analyze() is expected_result

    def check_all_aggregates(self, expected_result: bool = True):
        for a in self._compounds.iterate_all_compounds():
            a.link(self._compounds)
            assert a.analyze() is expected_result
        for f in self._flasks.iterate_all_flasks():
            f.link(self._flasks)
            assert f.analyze() is expected_result

    def check_all_structures(self, expected_result: bool = True):
        for a in self._structures.iterate_all_structures():
            a.link(self._structures)
            self.assert_is_enabled(a, expected_result)

    def check_all_steps(self, expected_result: bool = True):
        for a in self._elementary_steps.iterate_all_elementary_steps():
            a.link(self._elementary_steps)
            self.assert_is_enabled(a, expected_result)

    def check_all_reactions(self, expected_result: bool = True):
        for a in self._reactions.iterate_all_reactions():
            a.link(self._reactions)
            self.assert_is_enabled(a, expected_result)

    def set_up_calculation_with_results(self):
        model = db_setup.get_fake_model()
        calculation = db.Calculation.make(model, db.Job("fake_job"), [], self._calculations)
        results = calculation.get_results()
        structure_ids = [structure.id() for structure in self._structures.random_select_structures(4)]
        step_ids = [step.id() for step in self._elementary_steps.random_select_elementary_steps(3)]
        results.set_structures(structure_ids)
        results.set_elementary_steps(step_ids)
        calculation.set_results(results)
        calculation.set_status(db.Status.COMPLETE)
        return calculation

    def test_enable_all_aggregates(self):
        self.set_up_database()
        aggregate_policy = EnableAllAggregates()
        for a in self._compounds.iterate_all_compounds():
            a.link(self._compounds)
            aggregate_policy.process(a)
        for a in self._flasks.iterate_all_flasks():
            a.link(self._flasks)
            aggregate_policy.process(a)
        self.check_all_aggregates(True)

    def test_enable_no_aggregates(self):
        self.set_up_database()
        aggregate_policy = AggregateEnabling()
        for a in self._compounds.iterate_all_compounds():
            a.link(self._compounds)
            aggregate_policy.process(a)
        for a in self._flasks.iterate_all_flasks():
            a.link(self._flasks)
            aggregate_policy.process(a)
        self.check_all_aggregates(False)

    def test_enable_all_reactions(self):
        self.set_up_database()
        policy = EnableAllReactions()
        for a in self._reactions.iterate_all_reactions():
            a.link(self._reactions)
            policy.process(a)
        self.check_all_reactions(True)

    def test_enable_all_structures(self):
        self.set_up_database()
        policy = EnableAllStructures(aggregate_enabling_policy=EnableAllAggregates())
        policy.initialize_collections(self._manager)
        for a in self._structures.iterate_all_structures():
            a.link(self._structures)
            policy.process(a)
        self.check_all_structures(True)
        self.check_all_aggregates(expected_result=True)

    def test_enable_all_steps(self):
        self.set_up_database()
        policy = EnableAllSteps(reaction_enabling_policy=EnableAllReactions())
        policy.initialize_collections(self._manager)
        for a in self._elementary_steps.iterate_all_elementary_steps():
            a.link(self._elementary_steps)
            policy.process(a)
        self.check_all_steps(True)
        self.check_all_reactions(expected_result=True)

    def test_enable_all_steps_but_no_reactions(self):
        self.set_up_database()
        policy = EnableAllSteps(reaction_enabling_policy=ReactionEnabling())
        policy.initialize_collections(self._manager)
        for a in self._elementary_steps.iterate_all_elementary_steps():
            a.link(self._elementary_steps)
            policy.process(a)
        self.check_all_steps(True)
        self.check_all_reactions(expected_result=False)

    def check_calculation_results(self, calculation: db.Calculation, expected_result=True,
                                  expected_result_aggregates=False, expected_result_reactions=False):
        for a_id in calculation.get_results().get_elementary_steps():
            a = db.ElementaryStep(a_id, self._elementary_steps)
            self.assert_is_enabled(a, expected_result)
            reaction = db.Reaction(a.get_reaction(), self._reactions)
            self.assert_is_enabled(reaction, expected_result_reactions)
        for a_id in calculation.get_results().get_structures():
            s = db.Structure(a_id, self._structures)
            self.assert_is_enabled(s, expected_result)
            if s.has_aggregate():  # the structure may be a TS
                aggregate: Union[db.Compound, db.Flask] = db.Compound(s.get_aggregate(), self._compounds)
                if not aggregate.exists():
                    aggregate = db.Flask(s.get_aggregate(), self._flasks)
                assert aggregate.analyze() is expected_result_aggregates

    def test_enable_all_calculation_results(self):
        self.set_up_database()
        calculation = self.set_up_calculation_with_results()
        policy = EnableCalculationResults(step_enabling_policy=EnableAllSteps(EnableAllReactions()),
                                          structure_enabling_policy=EnableAllStructures(EnableAllAggregates()))
        policy.initialize_collections(self._manager)
        policy.process(calculation)
        self.check_calculation_results(calculation, True, True, True)

    def test_enable_job_specific_calculation_results(self):
        self.set_up_database()
        calculation = self.set_up_calculation_with_results()
        order = calculation.get_job().order
        random_structure = self._structures.random_select_structures(1)[0]
        random_structure.link(self._structures)
        random_structure.add_calculation(order, calculation.id())
        calculation.set_structures([random_structure.id()])
        model = calculation.get_model()
        wrong_model = db.Model("wrong", "model", "given")
        policy_1 = EnableJobSpecificCalculations(calculation.get_model(), "wrong_order",
                                                 step_enabling_policy=EnableAllSteps(EnableAllReactions()),
                                                 structure_enabling_policy=EnableAllStructures(EnableAllAggregates()))
        policy_1.initialize_collections(self._manager)
        policy_1.process_calculations_of_structures([random_structure.id()])
        self.check_calculation_results(calculation, False, False, False)

        policy_2 = EnableJobSpecificCalculations(wrong_model, order,
                                                 step_enabling_policy=EnableAllSteps(EnableAllReactions()),
                                                 structure_enabling_policy=EnableAllStructures(EnableAllAggregates()))
        policy_2.initialize_collections(self._manager)
        policy_2.process_calculations_of_structures([random_structure.id()])
        self.check_calculation_results(calculation, False, False, False)

        policy_3 = EnableJobSpecificCalculations(model, order,
                                                 step_enabling_policy=EnableAllSteps(EnableAllReactions()),
                                                 structure_enabling_policy=EnableAllStructures(EnableAllAggregates()))
        policy_3.initialize_collections(self._manager)
        calculation.disable_analysis()
        policy_3.process_calculations_of_structures([random_structure.id()])
        self.check_calculation_results(calculation, True, True, True)
        policy_3.process_calculations_of_structures([random_structure.id()])
        self.check_calculation_results(calculation, True, True, True)

    def test_apply_to_all_structures_in_aggregate(self):
        self.set_up_database()
        random_compounds = self._compounds.random_select_compounds(3)
        random_flasks = self._flasks.random_select_flasks(2)

        policy = ApplyToAllStructuresInAggregate(structure_enabling_policy=StructureEnabling())
        policy.initialize_collections(self._manager)
        for compound in random_compounds:
            compound.link(self._compounds)
            policy.process(compound)
            assert compound.analyze()
        for flask in random_flasks:
            flask.link(self._flasks)
            policy.process(flask)
            assert flask.analyze()

        self.check_all_structures(False)
        self.check_all_reactions(False)
        self.check_all_steps(False)
        self.disable_all()

        policy = ApplyToAllStructuresInAggregate(structure_enabling_policy=EnableAllStructures(AggregateEnabling()))
        policy.initialize_collections(self._manager)
        for compound in random_compounds:
            compound.link(self._compounds)
            policy.process(compound)
            assert compound.analyze()
            for s_id in compound.get_structures():
                structure = db.Structure(s_id, self._structures)
                self.assert_is_enabled(structure)
        for flask in random_flasks:
            flask.link(self._flasks)
            policy.process(flask)
            assert flask.analyze()
            for s_id in flask.get_structures():
                structure = db.Structure(s_id, self._structures)
                self.assert_is_enabled(structure)

        self.check_all_reactions(False)
        self.check_all_steps(False)
        self.disable_all()

    def test_apply_to_all_steps_in_reaction(self):
        self.set_up_database()
        random_reactions = self._reactions.random_select_reactions(4)

        policy = ApplyToAllStepsInReaction(step_enabling_policy=StructureEnabling())
        policy.initialize_collections(self._manager)
        for reaction in random_reactions:
            reaction.link(self._reactions)
            policy.process(reaction)
            self.assert_is_enabled(reaction)

        self.check_all_structures(False)
        self.check_all_aggregates(False)
        self.check_all_steps(False)
        self.disable_all()

        policy = ApplyToAllStepsInReaction(step_enabling_policy=EnableAllSteps(ReactionEnabling()))
        policy.initialize_collections(self._manager)
        for reaction in random_reactions:
            reaction.link(self._reactions)
            policy.process(reaction)
            self.assert_is_enabled(reaction)
            for step_id in reaction.get_elementary_steps():
                step = db.ElementaryStep(step_id, self._elementary_steps)
                self.assert_is_enabled(step)

        self.check_all_structures(False)
        self.check_all_aggregates(False)
        self.disable_all()

    def test_enable_structures_by_model(self):
        self.set_up_database()

        test_model = db.Model("My", "test", "model")
        policy = EnableStructureByModel(test_model, check_only_energy=True)
        policy.initialize_collections(self._manager)

        for a in self._structures.iterate_all_structures():
            a.link(self._structures)
            policy.process(a)
            db_setup.add_random_energy(a, (-10.0, 11.0), self._properties, test_model)
        self.check_all_structures(expected_result=False)

        for a in self._structures.iterate_all_structures():
            a.link(self._structures)
            policy.process(a)
        self.check_all_structures(expected_result=True)
        self.disable_all()

        policy = EnableStructureByModel(test_model, check_only_energy=False)
        policy.initialize_collections(self._manager)
        self.check_all_structures(expected_result=False)
        for a in self._structures.iterate_all_structures():
            a.link(self._structures)
            policy.process(a)
            a.set_model(test_model)
        self.check_all_structures(expected_result=False)

        for a in self._structures.iterate_all_structures():
            a.link(self._structures)
            policy.process(a)
        self.check_all_structures(expected_result=True)

    def test_filtered_step_enabling(self):
        self.set_up_database()

        test_model = db.Model("My", "test", "model")
        policy = FilteredStepEnabling(step_filter=ConsistentEnergyModelFilter(test_model))
        policy.initialize_collections(self._manager)

        random_step = self._elementary_steps.random_select_elementary_steps(1)[0]
        random_step.link(self._elementary_steps)

        policy.process(random_step)
        self.assert_is_enabled(random_step, False)
        reactants = random_step.get_reactants(db.Side.BOTH)
        s_ids = reactants[0] + reactants[1]
        if random_step.has_transition_state():
            s_ids.append(random_step.get_transition_state())

        energy_property_ids = []
        for s_id in s_ids:
            energy_property_ids.append(db_setup.add_random_energy(db.Structure(s_id, self._structures), (-10.0, 11.0),
                                       self._properties, test_model))
        policy.process(random_step)
        self.assert_is_enabled(random_step, True)
        self.disable_all()
        for p_id in energy_property_ids:
            property = db.Property(p_id, self._properties)
            property.set_model(db.Model("some", "third", "model"))
        policy.process(random_step)
        self.assert_is_enabled(random_step, False)

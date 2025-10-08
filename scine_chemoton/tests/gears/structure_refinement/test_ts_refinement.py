#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from json import dumps
import unittest

# Third party imports
import scine_database as db

# Local application tests imports
from scine_database.queries import model_query
from scine_database import test_database_setup as db_setup


# Local application imports
from ....engine import Engine
from ....gears.network_refinement.structure_refinement.ts_refinement import TSRefinement
from ....gears import HoldsCollections
from scine_chemoton.utilities.place_holder_model import (
    ModelNotSetError,
)


class TSRefinementTests(unittest.TestCase, HoldsCollections):

    def custom_setup(self, manager: db.Manager) -> None:
        self._required_collections = ["manager", "elementary_steps", "structures", "calculations",
                                      "reactions", "compounds", "flasks", "properties"]
        self.initialize_collections(manager)

    def tearDown(self) -> None:
        self._manager.wipe()

    def test_ts_loop(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_ts_loop")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        refine_model = db.Model("BFAKE", "BFAKE", "BF-AKE")

        # set up 2 compounds
        _, s1_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        _, s2_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)

        # set up step between compounds
        step = db.ElementaryStep()
        step.link(self._elementary_steps)
        step.create([s1_id], [s2_id])

        # set up TS and energies
        s1 = db.Structure(s1_id, self._structures)
        s2 = db.Structure(s2_id, self._structures)
        db_setup.add_random_energy(s1, (0.0, 1.0), self._properties)
        db_setup.add_random_energy(s2, (50.0, 51.0), self._properties)

        ts = db.Structure(db_setup.insert_single_empty_structure_aggregate(manager, db.Label.TS_OPTIMIZED)[1],
                          self._structures)
        db_setup.add_random_energy(ts, (70.0, 71.0), self._properties)
        step.set_transition_state(ts.get_id())

        ts_refinement_gear = TSRefinement()

        ts_refinement_engine = Engine(manager.get_credentials(), fork=False)
        ts_refinement_engine.set_gear(ts_refinement_gear)

        # Check no model
        with self.assertRaises(ModelNotSetError) as context:
            ts_refinement_engine.run(single=True)

        # Check same model
        ts_refinement_gear.options.model = model
        ts_refinement_gear.options.refine_model = model
        with self.assertRaises(RuntimeError) as context:
            ts_refinement_engine.run(single=True)
        self.assertTrue("Model and refine_model must be different!" in str(context.exception))

        ts_refinement_gear.options.refine_model = refine_model

        ts_refinement_gear.options.use_reactive_atoms = True
        ts_refinement_engine.run(single=True)
        assert self._calculations.count(dumps({})) == 0

        # Add reactive atoms to TS
        reactive_atoms = [0, 2]
        prop = db.VectorProperty()
        prop.link(self._properties)
        prop.create(model, "reactive_atoms", reactive_atoms)
        ts.add_property(prop.get_property_name(), prop.get_id())

        # Ensure cache is working
        for _ in range(3):
            ts_refinement_engine.run(single=True)

        assert ts_refinement_gear.options.structure_model == model
        # DB Checks
        assert self._calculations.count(dumps({})) == 1
        calc = self._calculations.find(dumps({"$and": model_query(refine_model)}))
        calc.link(self._calculations)

        target_job = ts_refinement_gear.options.ts_job
        target_settings = ts_refinement_gear.options.ts_job_settings
        target_settings["tsopt_automatic_mode_selection"] = reactive_atoms
        assert calc.get_job().order == target_job.order
        assert calc.get_settings() == target_settings

    def test_barrierless_loop_dissociation(self):
        # LHS and RHS for Dissociation
        for side in ['lhs', 'rhs']:
            # Connect to test DB
            manager = db_setup.get_clean_db("chemoton_test_barrierless_loop_dissociation")
            self.custom_setup(manager)

            # Add structure data
            model = db.Model("FAKE", "FAKE", "F-AKE")
            refine_model = db.Model("BFAKE", "BFAKE", "BF-AKE")

            # set up 2 compounds
            _, s1_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
            _, s2_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)

            # set up step between compounds
            step = db.ElementaryStep()
            step.link(self._elementary_steps)
            if side == 'lhs':
                step.create([s1_id], [s2_id, s2_id])
            else:
                step.create([s2_id, s2_id], [s1_id])
            step.set_type(db.ElementaryStepType.BARRIERLESS)

            # set up TS and energies
            # s0 = db.Structure(s0_id, self._structures)
            s1 = db.Structure(s1_id, self._structures)
            s2 = db.Structure(s2_id, self._structures)
            # db_setup.add_random_energy(s0, (66.0, 67.0), self._properties)
            db_setup.add_random_energy(s1, (0.0, 1.0), self._properties)
            db_setup.add_random_energy(s2, (50.0, 51.0), self._properties)

            ts_refinement_gear = TSRefinement()
            ts_refinement_gear.options.model = model
            ts_refinement_gear.options.refine_model = refine_model

            ts_refinement_engine = Engine(manager.get_credentials(), fork=False)
            ts_refinement_engine.set_gear(ts_refinement_gear)
            with self.assertRaises(RuntimeError) as context:
                ts_refinement_engine.run(single=True)
            self.assertTrue("Could not find calculation that created barrierless step" in str(context.exception))

            # Add Diss Calculation which has to be refined
            diss_job_order = "scine_dissociation_cut"
            calculation = db.Calculation(db.ID(), self._calculations)
            calculation.create(s1.get_model(), db.Job(diss_job_order), [s1_id])
            s1.add_calculation(calculation.job.order, calculation.id())
            calculation.set_status(db.Status.COMPLETE)
            fake_results = db.Results()
            fake_results.add_elementary_step(step.id())
            fake_results.structure_ids = [s2_id]
            calculation.set_results(fake_results)
            # Ensure cache is working
            for _ in range(3):
                ts_refinement_engine.run(single=True)

            assert self._calculations.count(dumps({})) == 2  # Initial Diss and Refined Diss
            ref_calc = self._calculations.find(dumps({"$and": model_query(refine_model)}))
            ref_calc.link(self._calculations)
            assert ref_calc.get_job().order == ts_refinement_gear.options.dissociation_job_rerun[1].order
            assert ref_calc.get_structures() == [s1_id]

    def test_barrierless_loop_bspline(self):
        manager = db_setup.get_clean_db("chemoton_test_barrierless_loop_bspline")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        refine_model = db.Model("BFAKE", "BFAKE", "BF-AKE")

        # set up 2 compounds
        _, s1_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        _, s2_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)

        # set up step between compounds
        step = db.ElementaryStep()
        step.link(self._elementary_steps)
        step.create([s1_id], [s2_id])
        step.set_type(db.ElementaryStepType.BARRIERLESS)

        # set up TS and energies
        # s0 = db.Structure(s0_id, self._structures)
        s1 = db.Structure(s1_id, self._structures)
        s2 = db.Structure(s2_id, self._structures)
        # db_setup.add_random_energy(s0, (66.0, 67.0), self._properties)
        db_setup.add_random_energy(s1, (0.0, 1.0), self._properties)
        db_setup.add_random_energy(s2, (50.0, 51.0), self._properties)

        ts_refinement_gear = TSRefinement()
        ts_refinement_gear.options.model = model
        ts_refinement_gear.options.refine_model = refine_model

        ts_refinement_engine = Engine(manager.get_credentials(), fork=False)
        ts_refinement_engine.set_gear(ts_refinement_gear)

        bspline_job_order = "scine_bspline_optimization"
        calculation = db.Calculation(db.ID(), self._calculations)
        calculation.create(s1.get_model(), db.Job(bspline_job_order), [s1_id, s2_id])
        s1.add_calculation(calculation.job.order, calculation.id())
        calculation.set_status(db.Status.COMPLETE)
        fake_results = db.Results()
        fake_results.add_elementary_step(step.id())
        calculation.set_results(fake_results)
        # Ensure cache is working
        for _ in range(3):
            ts_refinement_engine.run(single=True)

        assert self._calculations.count(dumps({})) == 2  # Initial Diss and Refined Diss
        ref_calc = self._calculations.find(dumps({"$and": model_query(refine_model)}))
        ref_calc.link(self._calculations)
        assert ref_calc.get_job().order == ts_refinement_gear.options.bspline_job_rerun.order
        assert ref_calc.get_structures() == [s1_id, s2_id]

    def test_reaction_loop(self):
        # Connect to test DB
        manager = db_setup.get_clean_db("chemoton_test_ts_loop")
        self.custom_setup(manager)

        # Add structure data
        model = db.Model("FAKE", "FAKE", "F-AKE")
        refine_model = db.Model("BFAKE", "BFAKE", "BF-AKE")

        # set up 2 compounds
        c1_id, s1_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)
        c2_id, s2_id = db_setup.insert_single_empty_structure_aggregate(manager, db.Label.MINIMUM_OPTIMIZED)

        s1 = db.Structure(s1_id, self._structures)
        s2 = db.Structure(s2_id, self._structures)
        db_setup.add_random_energy(s1, (0.0, 1.0), self._properties)
        db_setup.add_random_energy(s2, (50.0, 51.0), self._properties)

        # set up reaction
        reaction = db.Reaction()
        reaction.link(self._reactions)
        reaction.create([c1_id], [c2_id])
        compound_1 = db.Compound(c1_id, self._compounds)
        compound_2 = db.Compound(c2_id, self._compounds)
        compound_1.set_reactions([reaction.get_id()])
        compound_2.set_reactions([reaction.get_id()])

        # Add three elementary steps with different energies
        for ts_lower_energy in [60.0, 62.0, 69.0]:
            step = db.ElementaryStep()
            step.link(self._elementary_steps)
            step.create([s1_id], [s2_id])

            ts = db.Structure(db_setup.insert_single_empty_structure_aggregate(manager, db.Label.TS_OPTIMIZED)[1],
                              self._structures)
            db_setup.add_random_energy(ts, (ts_lower_energy, ts_lower_energy), self._properties)
            step.set_transition_state(ts.get_id())
            reaction.add_elementary_step(step.id())

        ts_refinement_gear = TSRefinement()
        ts_refinement_gear.options.model = model
        ts_refinement_gear.options.refine_model = refine_model
        ts_refinement_gear.options.lowest_per_reaction = True
        ts_refinement_gear.options.ts_energy_window = 0.0

        ts_refinement_engine = Engine(manager.get_credentials(), fork=False)
        ts_refinement_engine.set_gear(ts_refinement_gear)

        ts_refinement_engine.run(single=True)
        assert self._calculations.count(dumps({})) == 1  # Only min step should be refined

        ts_refinement_gear.clear_cache()
        ts_refinement_gear.options.ts_energy_window = 2.0
        ts_refinement_engine.run(single=True)
        assert self._calculations.count(dumps({})) == 2

        ts_refinement_gear.clear_cache()
        ts_refinement_gear.options.ts_energy_window = 10.0
        ts_refinement_engine.run(single=True)
        assert self._calculations.count(dumps({})) == 3

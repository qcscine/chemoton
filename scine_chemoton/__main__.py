#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
import pkg_resources
import psutil
import sys
import signal
import time
from typing import List

# Third party imports
import scine_database as db

# Local application imports
from scine_chemoton.utilities.insert_initial_structure import insert_initial_structure
from scine_chemoton.default_settings import default_nt_settings, default_cutting_settings, default_opt_settings
from scine_chemoton.engine import Engine, EngineHandler
from scine_chemoton.gears.scheduler import Scheduler
from scine_chemoton.gears.thermo import BasicThermoDataCompletion
from scine_chemoton.gears.compound import BasicAggregateHousekeeping
from scine_chemoton.gears.reaction import BasicReactionHousekeeping
from scine_chemoton.gears.network_refinement.calculation_based_refinement import CalculationBasedRefinement
from scine_chemoton.gears.kinetics import (
    MinimalConnectivityKinetics,
    # BasicBarrierHeightKinetics,
)
from scine_chemoton.gears.conformers.brute_force import BruteForceConformers
from scine_chemoton.gears.elementary_steps.minimal import MinimalElementarySteps
from scine_chemoton.gears.elementary_steps.trial_generator.bond_based import BondBased
from scine_chemoton.gears.elementary_steps.trial_generator.fast_dissociations import (
    FastDissociations
)
from scine_chemoton.filters.further_exploration_filters import FurtherExplorationFilter
from scine_chemoton.filters.aggregate_filters import AggregateFilter
from scine_chemoton.filters.reactive_site_filters import ReactiveSiteFilter
from scine_chemoton.utilities import yes_or_no_question


def main() -> None:
    # Prepare clean database
    manager = db.Manager()
    db_name = "default"
    ip = os.environ.get('TEST_MONGO_DB_IP', '127.0.0.1')
    port = os.environ.get('TEST_MONGO_DB_PORT', '27017')
    credentials = db.Credentials(ip, int(port), db_name)
    manager.set_credentials(credentials)
    manager.connect()
    if not manager.has_collection("calculations"):
        manager.init()

    # model = db.Model('pm6', 'pm6', '')
    model = db.Model("dftb3", "dftb3", "")
    # model = db.Model("gfn2", "gfn2", "")
    # model = db.Model('dft', 'wb97x_v', 'def2-svp')
    # model = db.Model('dft', 'pbe-d3bj', 'def2-svp')
    model.spin_mode = "unrestricted"
    model.program = "sparrow"

    wipe = True
    if len(sys.argv) > 1 and sys.argv[1].upper() == "CONTINUE":
        wipe = False

    if wipe:
        wipe = yes_or_no_question(f"Are you sure you want to wipe the database '{db_name}'")

    if wipe:
        manager.wipe()
        manager.init()
        time.sleep(1.0)
        # Load initial data
        methanol = pkg_resources.resource_filename("scine_chemoton", os.path.join("resources", "methanol.xyz"))
        formaldehyde = pkg_resources.resource_filename("scine_chemoton", os.path.join("resources", "formaldehyde.xyz"))

        _structure, _calculation = insert_initial_structure(
            manager,
            methanol,
            0,
            1,
            model,
            settings=default_opt_settings()
        )
        _structure, _calculation = insert_initial_structure(
            manager,
            formaldehyde,
            0,
            1,
            model,
            settings=default_opt_settings()
        )

    # check for existing chemoton process as a sanity check
    # Iterate over all running process
    for proc in psutil.process_iter():
        try:
            name = proc.name()
            process_id = proc.pid
            if "Chemoton" in name:
                continue_explore = yes_or_no_question(f"Detected a running Chemoton process '{name}' "
                                                      f"with id {process_id}. Do you want to continue")
                if not continue_explore:
                    sys.exit(1)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    # ================= #
    #   Start engines   #
    # ================= #

    engine_list: List[Engine] = []

    # 1) Conformer Engine
    conformer_gear = BruteForceConformers()
    conformer_gear.options.model = model
    conformer_gear.options.conformer_job = db.Job("conformers")
    conformer_gear.options.minimization_job = db.Job("scine_geometry_optimization")
    conformer_gear.options.minimization_settings = default_opt_settings()
    conformer_engine = Engine(credentials)
    conformer_engine.set_gear(conformer_gear)
    engine_list.append(conformer_engine)

    # 2) Compound generation and sorting
    compound_gear = BasicAggregateHousekeeping()
    compound_gear.options.model = model
    compound_engine = Engine(credentials)
    compound_engine.set_gear(compound_gear)
    engine_list.append(compound_engine)

    # 3) Thermo-chemical data completion
    thermo_gear = BasicThermoDataCompletion()
    thermo_gear.options.model = model
    thermo_gear.options.job = db.Job("scine_hessian")
    thermo_engine = Engine(credentials)
    thermo_engine.set_gear(thermo_gear)
    engine_list.append(thermo_engine)

    # 4) Reaction Exploration
    #  Set the settings for the elementary step exploration.
    #  These are the main settings for the general exploration
    # 4.1) Starting with the settings for the elementary step trial calculation (here an NT2 calculation)
    nt_job = db.Job("scine_react_complex_nt2")
    nt_settings = default_nt_settings()

    # 4.2) Choose the reaction types to be probed
    dissociations_gear = MinimalElementarySteps()
    dissociations_gear.aggregate_filter = AggregateFilter()
    dissociations_gear.options.enable_unimolecular_trials = True
    dissociations_gear.options.enable_bimolecular_trials = False
    dissociations_gear.trial_generator = FastDissociations()
    dissociations_gear.trial_generator.reactive_site_filter = ReactiveSiteFilter()
    dissociations_gear.trial_generator.further_exploration_filter = FurtherExplorationFilter()
    dissociations_gear.trial_generator.options.model = model
    dissociations_gear.trial_generator.options.job = db.Job("scine_dissociation_cut")
    dissociations_gear.trial_generator.options.cutting_job_settings = default_cutting_settings()
    dissociations_gear.trial_generator.options.min_bond_dissociations = 1
    dissociations_gear.trial_generator.options.max_bond_dissociations = 1
    dissociations_gear.trial_generator.options.enable_further_explorations = False
    dissociations_gear.trial_generator.options.always_further_explore_dissociative_reactions = True
    dissociations_gear.trial_generator.options.further_job = nt_job
    dissociations_gear.trial_generator.options.further_job_settings = nt_settings
    dissociations_engine = Engine(credentials)
    dissociations_engine.set_gear(dissociations_gear)
    engine_list.append(dissociations_engine)

    elementary_step_gear = MinimalElementarySteps()
    elementary_step_gear.trial_generator = BondBased()
    elementary_step_gear.trial_generator.options.model = model
    elementary_step_gear.options.enable_bimolecular_trials = True
    elementary_step_gear.options.enable_unimolecular_trials = True
    # 4.2.1) Minimalistic settings for bimolecular trial reaction coordinates and reactive complex generation
    #        Set-up one trial reaction coordinate consisting only of one intermolecular bond formation
    #        per trial calculation
    #        NOTE: The number of trial calculations scales steeply with the modification numbers chosen here.
    #              See elementary_step_gear.trial_generator.estimate_n_bimolecular_trials(...) to get an estimate of
    #              how many trials are to be expected from your options for given structures
    #              without enumerating them explicitly.
    #        NOTE: The modification numbers only specify which kind of changes are included in the trial reaction
    #              coordinates. This does not imply that the eventually resulting elementary steps include
    #              the same changes.
    elementary_step_gear.trial_generator.options.bimolecular_options.min_bond_modifications = 1
    elementary_step_gear.trial_generator.options.bimolecular_options.max_bond_modifications = 1
    elementary_step_gear.trial_generator.options.bimolecular_options.min_inter_bond_formations = 1
    elementary_step_gear.trial_generator.options.bimolecular_options.max_inter_bond_formations = 1
    elementary_step_gear.trial_generator.options.bimolecular_options.min_intra_bond_formations = 0
    elementary_step_gear.trial_generator.options.bimolecular_options.max_intra_bond_formations = 0
    elementary_step_gear.trial_generator.options.bimolecular_options.min_bond_dissociations = 0
    elementary_step_gear.trial_generator.options.bimolecular_options.max_bond_dissociations = 0
    elementary_step_gear.trial_generator.options.bimolecular_options.\
        complex_generator.options.number_rotamers = 1
    elementary_step_gear.trial_generator.options.bimolecular_options.\
        complex_generator.options.number_rotamers_two_on_two = 1
    elementary_step_gear.trial_generator.options.bimolecular_options.\
        complex_generator.options.multiple_attack_points = False
    # 4.2.2) Minimalistic settings for unimolecular additions
    #        Set-up trial reaction coordinates consisting of either one bond formation or one bond dissociation
    #        per trial calculation
    #        NOTE: The number of trial calculations scales steeply with the modification numbers chosen here.
    #              See elementary_step_gear.trial_generator.estimate_n_unimolecular_trials(...) to get an estimate of
    #              how many trials are to be expected from your options for a given structure,
    #              without enumerating them explicitly.
    #        NOTE: The modification numbers only specify which kind of changes are included in the trial reaction
    #              coordinates. This does not imply that the eventually resulting elementary steps include
    #              the same changes.
    elementary_step_gear.trial_generator.options.unimolecular_options.min_bond_modifications = 1
    elementary_step_gear.trial_generator.options.unimolecular_options.max_bond_modifications = 1
    elementary_step_gear.trial_generator.options.unimolecular_options.min_bond_formations = 0
    elementary_step_gear.trial_generator.options.unimolecular_options.max_bond_formations = 1
    elementary_step_gear.trial_generator.options.unimolecular_options.min_bond_dissociations = 0
    elementary_step_gear.trial_generator.options.unimolecular_options.max_bond_dissociations = 0
    # 4.3) Apply the basic calculation settings to all different reactions types in the gear
    #      Note: These settings could be different for different reaction types, resulting in better performance.
    elementary_step_gear.trial_generator.options.bimolecular_options.job = nt_job
    elementary_step_gear.trial_generator.options.bimolecular_options.job_settings = nt_settings
    elementary_step_gear.trial_generator.options.bimolecular_options.minimal_spin_multiplicity = False
    elementary_step_gear.trial_generator.options.unimolecular_options.job = nt_job
    #      Associative job settings are applied when at least one bond formation is included in the trial coordinate
    elementary_step_gear.trial_generator.options.unimolecular_options.job_settings_associative = nt_settings
    #      Disconnective job settings are applied when there are no associative components in the trial coordinate and
    #      it would result in splitting the reactant into two or more molecules
    elementary_step_gear.trial_generator.options.unimolecular_options.job_settings_disconnective = nt_settings
    #      Dissociative job settings are applied when there are no associative components in the trial coordinate but it
    #      would not result in splitting the reactant into two or more molecules
    elementary_step_gear.trial_generator.options.unimolecular_options.job_settings_dissociative = nt_settings
    # 4.4) Add filters (default ones, filter nothing)
    elementary_step_gear.trial_generator.reactive_site_filter = ReactiveSiteFilter()
    elementary_step_gear.aggregate_filter = AggregateFilter()
    # Run
    elementary_step_engine = Engine(credentials)
    elementary_step_engine.set_gear(elementary_step_gear)
    engine_list.append(elementary_step_engine)

    # Sorting elementary steps into reactions
    reaction_gear = BasicReactionHousekeeping()
    reaction_engine = Engine(credentials)
    reaction_engine.set_gear(reaction_gear)
    engine_list.append(reaction_engine)

    # Improve the network with a better model or find more connections with additional double ended searches
    refinement_gear = CalculationBasedRefinement()
    refinement_engine = Engine(credentials)
    refinement_gear.options.refinements = {
        "refine_single_points": False,  # SP for all minima and TS
        "refine_optimizations": False,  # optimize all minima and TS (+ validation)
        "double_ended_refinement": False,  # find TS of existing reactions of different model with double ended search
        "double_ended_new_connections": False,  # find more unimolecular reactions in the network
        "refine_single_ended_search": False,  # redo previously successful single ended reaction searches with new model
        "refine_structures_and_irc": False,  # redo irc and structure opt. from the old transition state.
    }
    pre_refinement_model = db.Model("PM6", "PM6", "")
    post_refinement_model = db.Model("DFT", "", "")
    refinement_gear.options.model = pre_refinement_model
    refinement_gear.options.post_refine_model = post_refinement_model
    refinement_engine.set_gear(refinement_gear)
    engine_list.append(refinement_engine)

    # Driving exploration based on kinetics
    kinetics_gear = MinimalConnectivityKinetics()  # activate all compounds
    """
    kinetics_gear = BasicBarrierHeightKinetics()  # activate compound if accessible via reaction with low enough barrier
    kinetics_gear.options.restart = True
    kinetics_gear.options.model = model  # model from which you want to take the energies from
    kinetics_gear.options.max_allowed_barrier = 1000.0  # kJ/mol
    kinetics_gear.options.enforce_free_energies = False  # only consider free energies for barrier height
    """
    kinetics_engine = Engine(credentials)
    kinetics_engine.set_gear(kinetics_gear)
    engine_list.append(kinetics_engine)

    # Calculation scheduling
    scheduling_gear = Scheduler()
    scheduling_gear.options.job_counts = {
        "scine_single_point": 500,
        "scine_geometry_optimization": 500,
        "scine_ts_optimization": 500,
        "scine_bond_orders": 500,
        "scine_hessian": 200,
        "scine_react_complex_nt2": 100,
        "scine_dissociation_cut": 100,
        "conformers": 20,
        "final_conformer_deduplication": 20,
        "graph": 1000,
    }
    scheduling_engine = Engine(credentials)
    scheduling_engine.set_gear(scheduling_gear)
    engine_list.append(scheduling_engine)

    # takes care of graceful clean up of forked processes
    handler = EngineHandler(engine_list, signals=[signal.SIGINT, signal.SIGTERM])
    handler.run()
    handler.wait_for_stop_signal()


if __name__ == "__main__":
    main()

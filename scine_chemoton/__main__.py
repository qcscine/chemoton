#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
import os
import pkg_resources
import time
import sys

# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application imports
from scine_chemoton.utilities.insert_initial_structure import insert_initial_structure
from scine_chemoton.engine import Engine
from scine_chemoton.gears.scheduler import Scheduler
from scine_chemoton.gears.thermo import BasicThermoDataCompletion
from scine_chemoton.gears.compound import BasicAggregateHousekeeping
from scine_chemoton.gears.reaction import BasicReactionHousekeeping
from scine_chemoton.gears.refinement import NetworkRefinement
from scine_chemoton.gears.kinetics import (
    MinimalConnectivityKinetics,
    # BasicBarrierHeightKinetics,
)
from scine_chemoton.gears.conformers.brute_force import BruteForceConformers
from scine_chemoton.gears.elementary_steps.minimal import MinimalElementarySteps
from scine_chemoton.gears.elementary_steps.trial_generator.bond_based import BondBased
from scine_chemoton.gears.elementary_steps.trial_generator.fast_dissociations import (
    FastDissociations,
    FurtherExplorationFilter
)
from scine_chemoton.gears.elementary_steps.compound_filters import CompoundFilter
from scine_chemoton.gears.elementary_steps.reactive_site_filters import ReactiveSiteFilter

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
if len(sys.argv) > 1:
    if sys.argv[1].upper() == "CONTINUE":
        wipe = False

if wipe:
    inp = input("Are you sure you want to wipe the database '" + db_name + "'? (y/n): ")
    while True:
        if inp.strip().lower() in ["y", "yes"]:
            break
        if inp.strip().lower() in ["n", "no"]:
            wipe = False
            break
        inp = input("Did not recognize answer, please answer 'y', if '" + db_name + "' should be wiped: ")

if wipe:
    manager.wipe()
    manager.init()
    time.sleep(1.0)
    # Load initial data
    methanol = pkg_resources.resource_filename("scine_chemoton", "tests/resources/methanol.xyz")
    formaldehyde = pkg_resources.resource_filename("scine_chemoton", "tests/resources/formaldehyde.xyz")

    structure, calculation = insert_initial_structure(
        manager,
        methanol,
        0,
        1,
        model,
        settings=utils.ValueCollection(
            {
                "convergence_max_iterations": 1000,
                "bfgs_use_trust_radius": True,
            }
        ),
    )
    structure, calculation = insert_initial_structure(
        manager,
        formaldehyde,
        0,
        1,
        model,
        settings=utils.ValueCollection(
            {
                "convergence_max_iterations": 1000,
                "bfgs_use_trust_radius": True,
            }
        ),
    )

# ================= #
#   Start engines   #
# ================= #

# 1) Conformer Engine
conformer_gear = BruteForceConformers()
conformer_gear.options.model = model
conformer_gear.options.conformer_job = db.Job("conformers")
conformer_gear.options.minimization_job = db.Job("scine_geometry_optimization")
conformer_gear.options.minimization_settings = utils.ValueCollection(
    {
        "convergence_max_iterations": 1000,
        "bfgs_use_trust_radius": True,
        "geoopt_coordinate_system": "cartesianWithoutRotTrans",
    }
)
conformer_engine = Engine(credentials)
conformer_engine.set_gear(conformer_gear)
conformer_engine.run()

# 2) Compound generation and sorting
compound_gear = BasicAggregateHousekeeping()
compound_gear.options.model = model
compound_engine = Engine(credentials)
compound_engine.set_gear(compound_gear)
compound_engine.run()

# 3) Thermo-chemical data completion
thermo_gear = BasicThermoDataCompletion()
thermo_gear.options.model = model
thermo_gear.options.job = db.Job("scine_hessian")
thermo_engine = Engine(credentials)
thermo_engine.set_gear(thermo_gear)
thermo_engine.run()

# 4) Reaction Exploration
#  Set the settings for the elementary step exploration.
#  These are the main settings for the general exploration
# 4.1) Starting with the settings for the elementary step trial calculation (here an NT2 calculation)
nt_job = db.Job("scine_react_complex_nt2")
nt_settings = utils.ValueCollection(
    {
        # # # Settings for the nt task
        "nt_convergence_max_iterations": 600,
        "nt_nt_total_force_norm": 0.1,
        "nt_sd_factor": 1.0,
        "nt_nt_use_micro_cycles": True,
        "nt_nt_fixed_number_of_micro_cycles": True,
        "nt_nt_number_of_micro_cycles": 10,
        "nt_nt_filter_passes": 10,
        # # # Settings for the tsopt task
        "tsopt_convergence_max_iterations": 1000,
        "tsopt_convergence_step_max_coefficient": 2.0e-3,
        "tsopt_convergence_step_rms": 1.0e-3,
        "tsopt_convergence_gradient_max_coefficient": 2.0e-4,
        "tsopt_convergence_gradient_rms": 1.0e-4,
        "tsopt_convergence_requirement": 3,
        "tsopt_convergence_delta_value": 1e-6,
        "tsopt_optimizer": "bofill",
        "tsopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
        "tsopt_bofill_trust_radius": 0.2,
        # 'tsopt_bofill_follow_mode': 0, # uncomment to disable automatic mode selection
        # 'tsopt_dimer_calculate_hessian_once': True, # enable when using dimer
        # 'tsopt_dimer_trust_radius': 0.1, # enable when using dimer
        # # # Settings for the irc task
        "irc_convergence_max_iterations": 100,
        "irc_sd_factor": 2.0,
        "irc_irc_initial_step_size": 0.3,
        "irc_stop_on_error": False,
        "irc_convergence_step_max_coefficient": 2.0e-3,
        "irc_convergence_step_rms": 1.0e-3,
        "irc_convergence_gradient_max_coefficient": 2.0e-4,
        "irc_convergence_gradient_rms": 1.0e-4,
        "irc_convergence_delta_value": 1.0e-6,
        "irc_irc_coordinate_system": "cartesianWithoutRotTrans",
        # # # Settings for the optimisation after the irc on combined product
        "ircopt_convergence_max_iterations": 1000,
        "ircopt_convergence_step_max_coefficient": 2.0e-3,
        "ircopt_convergence_step_rms": 1.0e-3,
        "ircopt_convergence_gradient_max_coefficient": 2.0e-4,
        "ircopt_convergence_gradient_rms": 1.0e-4,
        "ircopt_convergence_requirement": 3,
        "ircopt_convergence_delta_value": 1e-6,
        "ircopt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
        "ircopt_bfgs_use_trust_radius": True,
        "ircopt_bfgs_trust_radius": 0.2,
        # # # Settings for the optimisation after irc
        "opt_convergence_max_iterations": 1000,
        "opt_convergence_step_max_coefficient": 2.0e-3,
        "opt_convergence_step_rms": 1.0e-3,
        "opt_convergence_gradient_max_coefficient": 2.0e-4,
        "opt_convergence_gradient_rms": 1.0e-4,
        "opt_convergence_requirement": 3,
        "opt_convergence_delta_value": 1e-6,
        "opt_geoopt_coordinate_system": "cartesianWithoutRotTrans",
        "opt_bfgs_use_trust_radius": True,
        "opt_bfgs_trust_radius": 0.4,
    }
)
# 4.2) Choose the reaction types to be probed
dissociations_gear = MinimalElementarySteps()
dissociations_gear.options.enable_unimolecular_trials = True
dissociations_gear.options.enable_bimolecular_trials = False
dissociations_gear.trial_generator = FastDissociations()
dissociations_gear.trial_generator.options.model = model
dissociations_gear.compound_filter = CompoundFilter()
dissociations_gear.trial_generator.reactive_site_filter = ReactiveSiteFilter()
dissociations_gear.trial_generator.options.cutting_job = db.Job("scine_dissociation_cut")
dissociations_gear.trial_generator.options.cutting_job_settings = utils.ValueCollection({})
dissociations_gear.trial_generator.options.min_bond_dissociations = 1
dissociations_gear.trial_generator.options.max_bond_dissociations = 1
dissociations_gear.trial_generator.options.enable_further_explorations = False
dissociations_gear.trial_generator.options.always_further_explore_dissociative_reactions = True
dissociations_gear.trial_generator.options.further_exploration_filter = FurtherExplorationFilter()
dissociations_gear.trial_generator.options.further_job = nt_job
dissociations_gear.trial_generator.options.further_job_settings = nt_settings
dissociations_engine = Engine(credentials)
dissociations_engine.set_gear(dissociations_gear)
dissociations_engine.run()

elementary_step_gear = MinimalElementarySteps()
elementary_step_gear.trial_generator = BondBased()
elementary_step_gear.trial_generator.options.model = model
elementary_step_gear.options.enable_bimolecular_trials = True
elementary_step_gear.options.enable_unimolecular_trials = True
# 4.2.1) Minimalistic settings for bimolecular trial reaction coordinates and reactive complex generation
#        Set-up one trial reaction coordinate consting only of one intermolecular bond formation per trial calculation
#        NOTE: The number of trial calculations scales steeply with the modification numbers chosen here.
#              See elementary_step_gear.trial_generator.estimate_n_bimolecular_trials(...) to get an estimate of
#              how many trials are to be expected from your options for given structures
#              without enumerating them explicitly.
#        NOTE: The modification numbers only specify which kind of changes are included in the trial reaction
#              coordinates. This does not imply that the eventually resulting elementary steps include the same changes.
elementary_step_gear.trial_generator.options.bimolecular.min_bond_modifications = 1
elementary_step_gear.trial_generator.options.bimolecular.max_bond_modifications = 1
elementary_step_gear.trial_generator.options.bimolecular.min_inter_bond_formations = 1
elementary_step_gear.trial_generator.options.bimolecular.max_inter_bond_formations = 1
elementary_step_gear.trial_generator.options.bimolecular.min_intra_bond_formations = 0
elementary_step_gear.trial_generator.options.bimolecular.max_intra_bond_formations = 0
elementary_step_gear.trial_generator.options.bimolecular.min_bond_dissociations = 0
elementary_step_gear.trial_generator.options.bimolecular.max_bond_dissociations = 0
elementary_step_gear.trial_generator.options.bimolecular.complex_generator.options.number_rotamers = 1
elementary_step_gear.trial_generator.options.bimolecular.complex_generator.options.number_rotamers_two_on_two = 1
elementary_step_gear.trial_generator.options.bimolecular.complex_generator.options.multiple_attack_points = False
# 4.2.2) Minimalistic settings for unimolecular additions
#        Set-up trial reaction coordinates consisting of either one bond formation or one bond dissociation per trial
#        calculation
#        NOTE: The number of trial calculations scales steeply with the modification numbers chosen here.
#              See elementary_step_gear.trial_generator.estimate_n_unimolecular_trials(...) to get an estimate of
#              how many trials are to be expected from your options for a given structure,
#              without enumerating them explicitly.
#        NOTE: The modification numbers only specify which kind of changes are included in the trial reaction
#              coordinates. This does not imply that the eventually resulting elementary steps include the same changes.
elementary_step_gear.trial_generator.options.unimolecular.min_bond_modifications = 1
elementary_step_gear.trial_generator.options.unimolecular.max_bond_modifications = 1
elementary_step_gear.trial_generator.options.unimolecular.min_bond_formations = 0
elementary_step_gear.trial_generator.options.unimolecular.max_bond_formations = 1
elementary_step_gear.trial_generator.options.unimolecular.min_bond_dissociations = 0
elementary_step_gear.trial_generator.options.unimolecular.max_bond_dissociations = 0
# 4.3) Apply the basic calculation settings to all different reactions types in the gear
#      Note: These settings could be different for different reaction types, resulting in better performance.
elementary_step_gear.trial_generator.options.bimolecular.job = nt_job
elementary_step_gear.trial_generator.options.bimolecular.job_settings = nt_settings
elementary_step_gear.trial_generator.options.bimolecular.minimal_spin_multiplicity = False
elementary_step_gear.trial_generator.options.unimolecular.job = nt_job
#      Associative job settings are applied when at least one bond formation is included in the trial coordinate
elementary_step_gear.trial_generator.options.unimolecular.job_settings_associative = nt_settings
#      Disconnective job settings are applied when there are no associative components in the trial coordinate and it
#      would result in splitting the reactant into two or more molecules
elementary_step_gear.trial_generator.options.unimolecular.job_settings_disconnective = nt_settings
#      Dissociative job settings are applied when there are no associative components in the trial coordinate but it
#      would not result in splitting the reactant into two or more molecules
elementary_step_gear.trial_generator.options.unimolecular.job_settings_dissociative = nt_settings
# 4.4) Add filters (default ones, filter nothing)
elementary_step_gear.trial_generator.reactive_site_filter = ReactiveSiteFilter()
elementary_step_gear.compound_filter = CompoundFilter()
# Run
elementary_step_engine = Engine(credentials)
elementary_step_engine.set_gear(elementary_step_gear)
elementary_step_engine.run()

# Sorting elementary steps into reactions
reaction_gear = BasicReactionHousekeeping()
reaction_engine = Engine(credentials)
reaction_engine.set_gear(reaction_gear)
reaction_engine.run()

# Improve the network with a better model or find more connections with additional double ended searches
refinement_gear = NetworkRefinement()
refinement_engine = Engine(credentials)
refinement_gear.options.refinements = {
    "refine_single_points": False,  # SP for all minima and TS
    "refine_optimizations": False,   # optimize all minima and TS (+ validation)
    "double_ended_refinement": False,  # find TS of existing reactions of different model with double ended search
    "double_ended_new_connections": False,  # find more unimolecular reactions in the network
    "refine_single_ended_search": False,  # redo previously successful single ended reaction searches with new model
    "refine_structures_and_irc": False,  # redo irc and structure opt. from the old transition state.
}
pre_refinement_model = db.Model("PM6", "PM6", "")
post_refinement_model = db.Model("DFT", "", "")
refinement_gear.options.pre_refine_model = pre_refinement_model
refinement_gear.options.post_refine_model = post_refinement_model
refinement_engine.set_gear(refinement_gear)
refinement_engine.run()

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
kinetics_engine.run()

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
scheduling_engine = Engine(credentials, fork=False)
scheduling_engine.set_gear(scheduling_gear)
scheduling_engine.run()

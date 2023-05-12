#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

import scine_utilities as utils


def default_nt_settings() -> utils.ValueCollection:
    return utils.ValueCollection({
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
        # # # Settings for the optimisation of the reactive complex
        "rcopt_convergence_max_iterations": 1000,
    })

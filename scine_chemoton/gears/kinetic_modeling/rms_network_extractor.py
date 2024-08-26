#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

import scine_database as db
import scine_utilities as utils

from scine_chemoton.gears.kinetic_modeling.rms_kinetic_modeling import RMSKineticModelingJobFactory
from scine_chemoton.gears.kinetic_modeling.kinetic_modeling import KineticModeling
from scine_chemoton.gears.kinetic_modeling.atomization import MultiModelEnergyReferences


class ReactionNetworkData:
    """
    This class extracts the input values for a RMS kinetic modeling run from the database.

    Parameters
    ----------
    manager : db.Manager
        The database manager.
    options : KineticModeling.Options
        The kinetic modeling options. Note that the model combinations and the reference state must be set
        explicitly and may not be None.
    use_zero_flux_truncation : bool
        If true, reaction previously modeled in a kinetic modeling run which have negligible concentration flux
        will be neglected. By default, True.
    """

    def __init__(self, manager: db.Manager, options: KineticModeling.Options,
                 use_zero_flux_truncation: bool = True):
        assert options.model_combinations_reactions
        assert options.reference_state
        rms_job_factory = RMSKineticModelingJobFactory(options.model_combinations,
                                                       options.model_combinations_reactions,
                                                       manager, MultiModelEnergyReferences([]))
        rms_job_factory.reference_state = options.reference_state
        rms_job_factory.max_barrier = options.max_barrier
        rms_job_factory.min_flux_truncation = options.min_flux_truncation
        rms_job_factory.use_zero_flux_truncation = use_zero_flux_truncation

        reactions, aggregates = rms_job_factory.get_reactions()
        self.reference_state = rms_job_factory.reference_state
        hartree_to_j_per_mol = utils.KJPERMOL_PER_HARTREE * 1e+3
        a_values = aggregates.values()
        enthalpies = [a.get_enthalpy(self.reference_state) for a in a_values]
        entropies = [a.get_entropy(self.reference_state) for a in a_values]
        ea = [r.get_free_energy_of_activation(self.reference_state)[0] for r in reactions]
        rxn_e = [r.get_reaction_free_energy(self.reference_state) for r in reactions]
        assert None not in rxn_e
        assert None not in ea
        assert None not in entropies
        assert None not in enthalpies
        self.aggregate_ids = [a.get_db_id().string() for a in a_values]
        self.reaction_ids = [r.get_db_id().string() for r in reactions]
        self.enthalpies = [e * hartree_to_j_per_mol for e in enthalpies]  # type: ignore
        self.entropies = [e * hartree_to_j_per_mol for e in entropies]  # type: ignore
        self.ea = [e * hartree_to_j_per_mol for e in ea]  # type: ignore
        self.rxn_e = [e * hartree_to_j_per_mol for e in rxn_e]  # type: ignore
        self.prefactors = [r.get_arrhenius_prefactor(self.reference_state) for r in reactions]
        self.exponents = [0 for _ in reactions]
        self.aggregates = aggregates
        self.reactions = reactions
        assert None not in entropies
        assert None not in ea
        rms_job_factory.assert_non_negative_barriers(self.enthalpies, self.entropies, self.ea, self.reaction_ids,
                                                     self.aggregate_ids, self.reference_state.temperature)

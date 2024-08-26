#!/usr/bin/env python3
from __future__ import annotations
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from abc import abstractmethod
from typing import List, Optional, Any, Dict

import numpy as np
import scine_database as db
from scine_utilities import ValueCollection

from . import NetworkExpansion, thermochemistry_job_wrapper
from ..datastructures import ProtocolEntry, GearOptions, NetworkExpansionResult

from scine_chemoton.default_settings import default_nt_settings, default_cutting_settings
from scine_chemoton.gears.elementary_steps import ElementaryStepGear
from scine_chemoton.gears.elementary_steps.selected_structures import SelectedStructuresElementarySteps
from scine_chemoton.gears.elementary_steps.minimal import MinimalElementarySteps
from scine_chemoton.gears.elementary_steps.trial_generator.bond_based import BondBased
from scine_chemoton.gears.elementary_steps.trial_generator.fast_dissociations import (
    FastDissociations
)
from scine_chemoton.filters.further_exploration_filters import \
    ReactionCoordinateMaxDissociationEnergyFilter


class ChemicalReaction(NetworkExpansion):
    """
    The base class for all network expansions that should implement a basic reaction step.
    Still abstract and not to be used directly.
    """

    class Options(NetworkExpansion.Options):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.general_react_job = db.Job('scine_react_complex_nt2')
            self.general_react_job_settings = ValueCollection({})

    options: ChemicalReaction.Options  # required for mypy checks, so it knows which options object to check

    def __init__(self, model: db.Model,  # pylint: disable=keyword-arg-before-vararg
                 gear_options: Optional[GearOptions] = None,
                 status_cycle_time: float = 60.0,
                 include_thermochemistry: bool = False,
                 general_settings: Optional[Dict[str, Any]] = None,
                 add_default_chemoton_nt_settings: bool = True,
                 exact_settings_check: bool = False,
                 react_flasks: bool = False,
                 *args, **kwargs):
        super().__init__(model, gear_options, status_cycle_time, include_thermochemistry, general_settings,
                         *args, **kwargs)
        self._exact_settings_check = exact_settings_check
        self._add_default_chemoton_nt_settings = add_default_chemoton_nt_settings
        self._react_flasks = react_flasks
        if add_default_chemoton_nt_settings:
            self.options.general_react_job_settings.update(default_nt_settings().as_dict())

    @thermochemistry_job_wrapper
    def _relevant_puffin_jobs(self) -> List[str]:
        return [self.options.general_react_job.order, *self._aggregation_necessary_jobs()]

    @abstractmethod
    def _set_protocol(self, credentials: db.Credentials) -> None:
        pass

    def _execute(self, n_already_executed_protocol_steps: int) -> NetworkExpansionResult:
        """
        Convenience implementation of execution.
        First basic execute and then query for all modified reactions and add their reactants and products
        to the result.
        """
        self._basic_execute(n_already_executed_protocol_steps)
        result = NetworkExpansionResult()
        self._add_modified_reactions_to_results(result, include_reactants=True)
        return result

    def _basic_elementary_step_gear_setup(self, credentials: db.Credentials, trial_generator: BondBased = BondBased()) \
            -> ElementaryStepGear:
        """
        Convenience method that can be used by implementations.
        Sets up an ElementaryStepGear with the given TrialGenerator based on the holding options.
        """
        self._add_basic_chemoton_gears_to_protocol(credentials)

        if self._selection.structures:
            elementary_step_gear = SelectedStructuresElementarySteps()
            elementary_step_gear.options.selected_structures = self._selection.structures
        else:
            elementary_step_gear = MinimalElementarySteps()  # type: ignore
        elementary_step_gear.trial_generator = trial_generator
        if self._react_flasks:
            elementary_step_gear.options.looped_collection = "flasks"
        else:
            elementary_step_gear.options.looped_collection = "compounds"

        # unimolecular job + settings
        elementary_step_gear.trial_generator.options.unimolecular_options.job = self.options.general_react_job
        if isinstance(self.options.general_settings, ValueCollection):
            elementary_step_gear.trial_generator.options.base_job_settings.update(
                self.options.general_settings.as_dict()
            )
        elif isinstance(self.options.general_settings, dict):
            elementary_step_gear.trial_generator.options.base_job_settings.update(self.options.general_settings)
        for settings in [
            # leave out cutting job settings, because they are different
            "job_settings_associative",
            "job_settings_dissociative",
            "job_settings_disconnective",
            "further_job_settings",
        ]:
            if hasattr(elementary_step_gear.trial_generator.options.unimolecular_options, settings):
                existing_settings = getattr(elementary_step_gear.trial_generator.options.unimolecular_options, settings)
                existing_settings.update(self.options.general_react_job_settings.as_dict())
                setattr(elementary_step_gear.trial_generator.options.unimolecular_options, settings, existing_settings)

        # bimolecular job + settings
        elementary_step_gear.trial_generator.options.bimolecular_options.job = self.options.general_react_job
        elementary_step_gear.trial_generator.options.bimolecular_options.job_settings.update(
            self.options.general_react_job_settings.as_dict()
        )

        elementary_step_gear.options.run_one_cycle_with_settings_enhancement = self._exact_settings_check
        elementary_step_gear.disable_caching()

        return elementary_step_gear


class Dissociation(ChemicalReaction):
    """
    Represents an unimolecular dissociation into two or more separate compounds.
    """

    class Options(ChemicalReaction.Options):

        def __init__(self,
                     model: db.Model,
                     status_cycle_time: float,
                     include_thermochemistry: bool,
                     gear_options: Optional[GearOptions],
                     general_settings: Optional[Dict[str, Any]],
                     max_bond_dissociations: int,
                     further_explore_dissociations_barrier: float,
                     *args, **kwargs):
            super().__init__(model, status_cycle_time, include_thermochemistry, gear_options, general_settings,
                             *args, **kwargs)
            self.max_bond_dissociations = max_bond_dissociations
            self.dissociation_job = db.Job('scine_dissociation_cut')
            self.further_explore_dissociations_barrier = further_explore_dissociations_barrier

    options: Dissociation.Options  # required for mypy checks, so it knows which options object to check

    def __init__(self, model: db.Model,  # pylint: disable=keyword-arg-before-vararg
                 gear_options: Optional[GearOptions] = None,
                 status_cycle_time: float = 60.0,
                 include_thermochemistry: bool = False,
                 general_settings: Optional[Dict[str, Any]] = None,
                 add_default_chemoton_nt_settings: bool = True,
                 exact_settings_check: bool = False,
                 react_flasks: bool = False,
                 max_bond_dissociations: int = 1,
                 further_explore_dissociations_barrier: float = 200.0,
                 *args, **kwargs):
        super().__init__(model, gear_options, status_cycle_time, include_thermochemistry, general_settings,
                         add_default_chemoton_nt_settings, exact_settings_check, react_flasks,
                         max_bond_dissociations, further_explore_dissociations_barrier, *args, **kwargs)

    @thermochemistry_job_wrapper
    def _relevant_puffin_jobs(self) -> List[str]:
        return [self.options.dissociation_job.order] + super()._relevant_puffin_jobs()

    def _set_protocol(self, credentials: db.Credentials) -> None:
        dissociations_gear = self._basic_elementary_step_gear_setup(credentials)

        dissociations_gear.options.enable_unimolecular_trials = True
        dissociations_gear.options.enable_bimolecular_trials = False
        # safe general settings before overwrite the trial generator
        general_settings = dissociations_gear.trial_generator.options.base_job_settings
        dissociations_gear.trial_generator = FastDissociations()
        dissociations_gear.trial_generator.options.base_job_settings = general_settings

        if self._add_default_chemoton_nt_settings:
            dissociations_gear.trial_generator.options.cutting_job_settings = default_cutting_settings()
        dissociations_gear.trial_generator.options.min_bond_dissociations = 1
        dissociations_gear.trial_generator.options.max_bond_dissociations = self.options.max_bond_dissociations
        dissociations_gear.trial_generator.options.enable_further_explorations = True
        dissociations_gear.trial_generator.options.always_further_explore_dissociative_reactions = True
        further_job = self.options.general_react_job
        dissociations_gear.trial_generator.options.unimolecular_options.further_job = further_job
        dissociations_gear.trial_generator.options.unimolecular_options.further_job_settings.update(
            self.options.general_react_job_settings.as_dict()
        )
        dissociations_gear.trial_generator.further_exploration_filter = ReactionCoordinateMaxDissociationEnergyFilter(
            max_dissociation_energy=self.options.further_explore_dissociations_barrier,
            energy_type=self.energy_type,
            model=self.options.model,
            job_order=further_job.order,
        )

        self.protocol.append(ProtocolEntry(credentials, dissociations_gear, n_runs=1,
                                           wait_for_calculation_finish=True, fork=False))
        self._extra_manual_cycles_to_avoid_race_condition(credentials, aggregate_reactions=True)


class Association(Dissociation):
    """
    Represents the bimolecular association of two compounds.
    """

    class Options(Dissociation.Options):

        def __init__(self,
                     model: db.Model,
                     status_cycle_time: float,
                     include_thermochemistry: bool,
                     gear_options: Optional[GearOptions],
                     general_settings: Optional[Dict[str, Any]],
                     max_bond_dissociations: int,
                     further_explore_dissociations_barrier: float,
                     max_bond_associations: int,
                     max_intra_associations: int, *args, **kwargs):
            super().__init__(model, status_cycle_time, include_thermochemistry, gear_options, general_settings,
                             max_bond_dissociations, further_explore_dissociations_barrier,
                             *args, **kwargs)
            self.max_bond_associations = max_bond_associations
            self.max_intra_associations = max_intra_associations

    options: Association.Options  # required for mypy checks, so it knows which options object to check

    def __init__(self, model: db.Model,  # pylint: disable=keyword-arg-before-vararg
                 gear_options: Optional[GearOptions] = None,
                 status_cycle_time: float = 60.0,
                 include_thermochemistry: bool = False,
                 general_settings: Optional[Dict[str, Any]] = None,
                 add_default_chemoton_nt_settings: bool = True,
                 exact_settings_check: bool = False,
                 react_flasks: bool = False,
                 max_bond_associations: int = 1, max_bond_dissociations: int = 0,
                 max_intra_associations: int = 0, *args, **kwargs):
        super().__init__(model, gear_options, status_cycle_time, include_thermochemistry, general_settings,
                         add_default_chemoton_nt_settings, exact_settings_check,
                         react_flasks,  # react_flasks
                         max_bond_dissociations,  # max_bond_dissociations
                         np.inf,  # further_explore_dissociations_barrier
                         max_bond_associations,  # max_bond_associations
                         max_intra_associations,  # max_intra_associations
                         *args, **kwargs)

    def _set_protocol(self, credentials: db.Credentials) -> None:
        elementary_step_gear = self._basic_elementary_step_gear_setup(credentials)

        elementary_step_gear.options.enable_unimolecular_trials = False
        elementary_step_gear.options.enable_bimolecular_trials = True

        if isinstance(elementary_step_gear.trial_generator, BondBased):
            elementary_step_gear.trial_generator.options.bimolecular_options.min_bond_modifications = 1
            elementary_step_gear.trial_generator.options.bimolecular_options.max_bond_modifications = \
                self.options.max_bond_associations + self.options.max_bond_dissociations

            elementary_step_gear.trial_generator.options.bimolecular_options.min_intra_bond_formations = 0
            elementary_step_gear.trial_generator.options.bimolecular_options.max_intra_bond_formations = \
                self.options.max_intra_associations

            elementary_step_gear.trial_generator.options.bimolecular_options.max_bond_dissociations = \
                self.options.max_bond_dissociations

            elementary_step_gear.trial_generator.options.bimolecular_options.min_inter_bond_formations = 1
            elementary_step_gear.trial_generator.options.bimolecular_options.max_inter_bond_formations = \
                self.options.max_bond_associations

        self.protocol.append(ProtocolEntry(credentials, elementary_step_gear, n_runs=1,
                                           wait_for_calculation_finish=True, fork=False))
        self._extra_manual_cycles_to_avoid_race_condition(credentials, aggregate_reactions=True)


class Rearrangement(Association):
    """
    Represents the unimolecular rearrangement of a compound into another compound
    or multiple separate compounds.
    """

    options: Rearrangement.Options

    def _set_protocol(self, credentials: db.Credentials) -> None:
        elementary_step_gear = self._basic_elementary_step_gear_setup(credentials)
        elementary_step_gear.options.enable_unimolecular_trials = True
        elementary_step_gear.options.enable_bimolecular_trials = False

        formations = max([self.options.max_bond_associations, self.options.max_intra_associations])

        if isinstance(elementary_step_gear.trial_generator, BondBased):
            elementary_step_gear.trial_generator.options.unimolecular_options.min_bond_modifications = 1
            elementary_step_gear.trial_generator.options.unimolecular_options.max_bond_dissociations = \
                self.options.max_bond_dissociations
            elementary_step_gear.trial_generator.options.unimolecular_options.min_bond_formations = 0
            elementary_step_gear.trial_generator.options.unimolecular_options.min_bond_dissociations = 0
            elementary_step_gear.trial_generator.options.unimolecular_options.max_bond_formations = \
                formations
            elementary_step_gear.trial_generator.options.unimolecular_options.max_bond_modifications = \
                formations + self.options.max_bond_dissociations

        self.protocol.append(ProtocolEntry(credentials, elementary_step_gear, n_runs=1,
                                           wait_for_calculation_finish=True, fork=False))
        self._extra_manual_cycles_to_avoid_race_condition(credentials, aggregate_reactions=True)

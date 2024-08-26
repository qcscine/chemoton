#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
# Standard library imports
from time import sleep
from json import dumps
from typing import Optional, List
# Third party imports
import scine_database as db
import scine_utilities as utils

# Local application imports
from .. import Gear
from .prepare_kinetic_modeling_job import KineticModelingJobFactory
from .rms_kinetic_modeling import RMSKineticModelingJobFactory
from .kinetx_kinetic_modeling import KinetxKineticModelingJobFactory
from ...utilities.db_object_wrappers.thermodynamic_properties import ReferenceState, PlaceHolderReferenceState
from ...utilities.model_combinations import ModelCombination
from .atomization import (
    AtomEnergyReference, ZeroEnergyReference, MultiModelEnergyReferences, PlaceHolderMultiModelEnergyReferences
)
from ...utilities.uncertainties import UncertaintyEstimator, ZeroUncertainty
from scine_chemoton.utilities.place_holder_model import (
    construct_place_holder_model,
    PlaceHolderModelType
)


class KineticModeling(Gear):
    """
    This gear sets up kinetic modeling jobs as soon as no other jobs are pending/waiting.

    Attributes
    ----------
    options : KineticModeling.Options
        The options for the KineticModeling gear.
    """
    class Options(Gear.Options):
        """
        The options for the KineticModeling Gear.
        """

        __slots__ = (
            "model_combinations_reactions",
            "model_combinations",
            "job",
            "max_barrier",
            "min_flux_truncation",
            "reference_state",
            "job_settings",
            "only_electronic",
            "energy_references",
            "energy_reference_type",
            "uncertainty_estimator",
            "flux_variance_label"
        )

        def __init__(self) -> None:
            super().__init__()
            self.job = db.Job('kinetx_kinetic_modeling')
            """
            int
                Set up a kinetic modeling job if at least this number of new
                elementary steps were added to the network and are eligible according
                to reaction-barrier cutoff, and electronic structure model.
            """
            self.model_combinations_reactions = [ModelCombination(construct_place_holder_model())]
            """
            List[ModelCombinations]
                The hierarchy of model combinations for reaction barriers.
            """
            self.model_combinations: List[ModelCombination] = [ModelCombination(construct_place_holder_model())]
            """
            List[ModelCombinations]
                The hierarchy of model combinations
            """
            self.max_barrier: float = 300.0  # in kJ/mol
            """
            float
                The maximum barrier to consider in the kinetic modeling.
            """
            self.min_flux_truncation: float = 1e-9
            """
            float
                Minimum flux of all aggregates in a reaction in a previous kinetic modeling job. If the flux is lower
                than this threshold, the reaction is excluded from the kinetic modeling.
            """
            self.reference_state: ReferenceState = PlaceHolderReferenceState()
            """
            ReferenceState
                The thermodynamic reference state defined by temperature (in K) and pressure (in Pa).
            """
            self.job_settings = utils.ValueCollection({})
            """
            ValueCollection
                The job settings. See get_default_settings(job) for default settings.
            """
            self.only_electronic = False
            """
            bool
                If true, only the electronic energies are used for the rate constant calculations.
            """
            self.energy_references: MultiModelEnergyReferences = PlaceHolderMultiModelEnergyReferences()
            """
            MultiModelEnergyReferences
                Optional atom energy calculators for each model combination. If none are provided, they are constructed
                on the fly.
            """
            self.energy_reference_type = "zero"
            """
            str
                The energy reference type. zero -> absolute energies are used. atom -> Atomization energies are used.
            """
            self.uncertainty_estimator: UncertaintyEstimator = ZeroUncertainty()
            """
            UncertaintyEstimator
                The uncertainty estimator for the thermodynamic parameters (only used in RMS kinetic modeling).
            """
            self.flux_variance_label: str = ""
            """
            str
                If not empty, the minimum flux truncation also considers the flux's variance.
            """

    options: Options

    def __init__(self) -> None:
        super().__init__()
        self._required_collections = ["calculations", "elementary_steps", "reactions"]
        self._job_factory: Optional[KineticModelingJobFactory] = None
        self._last_n_enabled_reactions: int = -1
        self._model_is_required = False

    def reset_job_factory(self):
        self._job_factory = None

    def _get_job_factory(self) -> KineticModelingJobFactory:
        if (not self.options.model_combinations
                or isinstance(self.options.model_combinations[0].electronic_model, PlaceHolderModelType)):
            self.options.model_combinations = [ModelCombination(self.options.model)]
        if (not self.options.model_combinations_reactions
                or isinstance(self.options.model_combinations_reactions[0].electronic_model, PlaceHolderModelType)):
            self.options.model_combinations_reactions = self.options.model_combinations
        if self._job_factory is None:
            if self.options.job == RMSKineticModelingJobFactory.get_job():
                self._job_factory = RMSKineticModelingJobFactory(self.options.model_combinations,
                                                                 self.options.model_combinations_reactions,
                                                                 self._manager, self._get_energy_references(),
                                                                 self.options.uncertainty_estimator,
                                                                 self.options.only_electronic)
            elif self.options.job == KinetxKineticModelingJobFactory.get_job():
                self._job_factory = KinetxKineticModelingJobFactory(self.options.model_combinations,
                                                                    self.options.model_combinations_reactions,
                                                                    self._manager, self.options.only_electronic)
            else:
                raise RuntimeError("Error: The given kinetic modeling job is not supported. Options are:\n"
                                   + RMSKineticModelingJobFactory.get_job().order + "\n"
                                   + KinetxKineticModelingJobFactory.get_job().order)
        if isinstance(self.options.reference_state, PlaceHolderReferenceState):
            # We already check in the beginning of the function if the list model_combinations is empty.
            model = self.options.model_combinations[0].electronic_model
            self.options.reference_state = ReferenceState(float(model.temperature), float(model.pressure))
        self._job_factory.reference_state = self.options.reference_state
        self._job_factory.max_barrier = self.options.max_barrier
        self._job_factory.min_flux_truncation = self.options.min_flux_truncation
        self._job_factory.flux_variance_label = self.options.flux_variance_label
        return self._job_factory

    @staticmethod
    def get_default_settings(job: db.Job) -> utils.ValueCollection:
        if job.order == RMSKineticModelingJobFactory.get_job().order:
            return RMSKineticModelingJobFactory.get_default_settings()
        elif job.order == KinetxKineticModelingJobFactory.get_job().order:
            return KinetxKineticModelingJobFactory.get_default_settings()
        else:
            raise RuntimeError("Error: The given kinetic modeling job is not supported. Options are:\n"
                               + RMSKineticModelingJobFactory.get_job().order + "\n"
                               + KinetxKineticModelingJobFactory.get_job().order)

    def start_conditions_are_met(self) -> bool:
        # Run the kinetic modeling if all other calculations are done, independent on the elementary-step interval.
        if self._get_n_queuing_calculations() > 0:
            return False
        if not self._reactions_are_consistently_enabled():
            return False
        return True

    def _reactions_are_consistently_enabled(self):
        n_enabled_reactions = self._reactions.count(dumps({"exploration_disabled": False}))
        if self._last_n_enabled_reactions < 0:
            self._last_n_enabled_reactions = n_enabled_reactions
        consistent = n_enabled_reactions == self._last_n_enabled_reactions
        self._last_n_enabled_reactions = n_enabled_reactions
        return consistent

    def _loop_impl(self):
        # reset the number of enabled reactions. Ensure consistency for the two checks below.
        self._last_n_enabled_reactions = -1
        if not self.start_conditions_are_met() or self.have_to_stop_at_next_break_point():
            return
        sleep(self.options.cycle_time)  # make sure that there is not just one gear lagging behind.
        if not self.start_conditions_are_met() or self.have_to_stop_at_next_break_point():
            return
        self._get_job_factory().create_kinetic_modeling_job(self.options.job_settings)

    def _get_n_queuing_calculations(self) -> int:
        selection = {
            "$and": [
                {"$or": [{"status": "new"}, {"status": "hold"}, {"status": "pending"}]},
            ]
        }
        return self._calculations.count(dumps(selection))

    def _reaction_gear_finished(self) -> bool:
        selection = {
            "reaction": ""
        }
        return self._elementary_steps.count(dumps(selection)) == 0

    def _build_energy_reference(self, model: db.Model) -> ZeroEnergyReference:
        if self.options.energy_reference_type == "zero":
            return ZeroEnergyReference(model)
        if self.options.energy_reference_type == "atom":
            return AtomEnergyReference(model, self._manager)
        raise RuntimeError("Error: Unknown energy reference type. Options are: zero (absolute energies), atom"
                           " (atomization energies).")

    def _get_energy_references(self) -> MultiModelEnergyReferences:
        combs = self.options.model_combinations
        if self.options.model_combinations_reactions is not None:
            combs += self.options.model_combinations_reactions
        if isinstance(self.options.energy_references, PlaceHolderMultiModelEnergyReferences):
            refs = [self._build_energy_reference(c.electronic_model) for c in combs]
            self.options.energy_references = MultiModelEnergyReferences(refs)
        for c in combs:
            if not self.options.energy_references.has_reference(c.electronic_model):
                raise RuntimeError("Error: Energy reference missing for model " + str(c.electronic_model))
        return self.options.energy_references

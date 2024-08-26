#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""


from typing import Union, Optional, List, Set
from json import dumps

from scine_database.queries import select_calculation_by_structures, get_common_calculation_ids, stop_on_timeout
from scine_database.energy_query_functions import get_energy_for_structure

from scine_chemoton.gears import HoldsCollections, HasName
from scine_chemoton.filters.elementary_step_filters import ElementaryStepFilter

# Third party imports
import scine_database as db


class DBObjectEnabling(HasName):
    """
    These classes allow the enabling of previously disabled database objects like structures, calculation etc. and are
    intended to be used in combination with the classes implemented in disabling.py. See disabling.py for more
    information.
    """

    def __init__(self) -> None:
        super().__init__()
        self._remove_chemoton_from_name()

    @staticmethod
    def enable(db_object: Union[db.Structure, db.Compound, db.Flask, db.ElementaryStep, db.Reaction, db.Calculation]):
        """
        Enabling a database object.

        Parameters
        ----------
        db_object : Union[db.Structure, db.Compound, db.Flask, db.ElementaryStep, db.Reaction, db.Calculation]
            The database object to be enabled.
        """
        db_object.enable_exploration()
        db_object.enable_analysis()


class ReactionEnabling(DBObjectEnabling):
    """
    Base class for reaction enabling. By default, this class does nothing.
    """

    def process(self, _: db.Reaction):
        """
        Process a reaction. By default, do nothing.
        """
        return


class PlaceHolderReactionEnabling(ReactionEnabling):
    """
    A place-holder reaction enabling policy that can be used instead of None default arguments and replaced later.
    """

    def process(self, _: db.Reaction):
        raise NotImplementedError


class EnableAllReactions(ReactionEnabling):
    """
    Enable all reactions.
    """

    def process(self, reaction: db.Reaction):
        """
        Process a reaction. Enable all reactions.

        Parameters
        ----------
        reaction : db.Reaction
            The reaction to enable.
        """
        self.enable(reaction)


class AggregateEnabling(DBObjectEnabling):
    """
    Base class for aggregate enabling.
    """

    def process(self, _: Union[db.Compound, db.Flask]):
        """
        Process an aggregate. By default, do nothing
        """
        return

    def __eq__(self, other):
        """
        Equality check.

        Parameter
        ---------
        other : Any
            The other object.
        """
        return isinstance(other, self.__class__)


class EnableAllAggregates(AggregateEnabling):
    """
    Enable all aggregates.
    """

    def process(self, aggregate: Union[db.Compound, db.Flask]):
        """
        Process an aggregate by enabling it.

        Parameters
        ----------
        aggregate : Union[db.Compound, db.Flask]
            The aggregate.
        """
        aggregate.enable_analysis()


class PlaceHolderAggregateEnabling(AggregateEnabling):
    """
    A place-holder for the aggregate enabling logic that can be used instead of a None default argument.
    """

    def process(self, _: Union[db.Compound, db.Flask]):
        raise NotImplementedError


class StructureEnabling(HoldsCollections, DBObjectEnabling):
    """
    Base class for structure enabling. This base class does nothing on its own.
    """

    def process(self, _: db.Structure):
        """
        Process a structure. By default, do nothing.

        Parameters
        ----------
        _ : db.Structure
            The structure.
        """
        return

    def enable_structure(self, structure: db.Structure):
        """
        Enable a structure.

        Parameters
        ----------
        structure : db.Structure
            The structure.
        """
        assert isinstance(structure, db.Structure)
        if structure.get_label() == db.Label.DUPLICATE:
            DBObjectEnabling.enable(db.Structure(structure.is_duplicate_of(), self._structures))
            return
        DBObjectEnabling.enable(structure)


class EnableAllStructures(StructureEnabling):
    """
    Enable all structures. By default, also their associated aggregates are enabled.

    Parameters
    ----------
    aggregate_enabling_policy : AggregateEnabling
        The policy to apply to the associated aggregates.
    """

    def __init__(self, aggregate_enabling_policy: AggregateEnabling = EnableAllAggregates()) -> None:
        super().__init__()
        self._required_collections = ["compounds", "flasks", "structures"]
        self._aggregate_enabling_policy = aggregate_enabling_policy

    def process(self, structure: db.Structure):
        """
        Process a structure by enabling it and applying the aggregate policy to its aggregate.

        Parameters
        ----------
        structure : db.Structure
            The structure.
        """
        self.enable_structure(structure)
        aggregate = self._get_aggregate(structure)
        if aggregate is not None:
            self._aggregate_enabling_policy.process(aggregate)  # type: ignore

    def _get_aggregate(self, structure: db.Structure) -> Optional[Union[db.Compound, db.Flask]]:
        if not structure.has_aggregate():
            return None
        a_id: db.ID = structure.get_aggregate()
        aggregate: Union[db.Compound, db.Flask] = db.Compound(a_id, self._compounds)
        if not aggregate.exists():
            aggregate = db.Flask(a_id, self._flasks)
        return aggregate


class EnableStructureByModel(StructureEnabling):
    """
    Enable a structure if its electronic structure model matches the given model, or it has an energy with this
    model.

    Parameters
    ----------
    model : db.Mode
        The electronic structure model to match.
    check_only_energy : bool
        Whether to check only if the structure has an electronic energy with the given model.
    """

    def __init__(self, model: db.Model, check_only_energy: bool = False) -> None:
        super().__init__()
        self._model = model
        self._required_collections = ["structures", "properties"]
        self._check_only_energy = check_only_energy

    def process(self, structure: db.Structure):
        """
        Process a structure.

        Parameters
        ----------
        structure : db.Structure
            The structure.
        """
        if self._check_only_energy and get_energy_for_structure(structure, "electronic_energy", self._model,
                                                                self._structures, self._properties) is not None:
            self.enable_structure(structure)
        elif structure.get_model() == self._model:
            self.enable_structure(structure)


class StepEnabling(HoldsCollections, DBObjectEnabling):
    """
    Base class for elementary step enabling.
    """

    def process(self, _: db.ElementaryStep):
        """
        Process an elementary step. By default, do nothing.
        """
        return


class EnableAllSteps(StepEnabling):
    """
    Enables all elementary steps. By default, also the associated reactions are enabled.

    Parameters
    ----------
    reaction_enabling_policy : ReactionEnabling
        The enabling policy to be applied to the elementary step's reaction. By default, all reactions are enabled.
    """

    def __init__(self, reaction_enabling_policy: ReactionEnabling = EnableAllReactions()) -> None:
        super().__init__()
        self._reaction_enabling_policy = reaction_enabling_policy
        self._required_collections = ["reactions"]
        assert reaction_enabling_policy

    def process(self, step: db.ElementaryStep):
        """
        Process an elementary step.

        Parameters
        ----------
        step : db.ElementaryStep
            The elementary step process.
        """
        self.enable(step)
        if step.has_reaction():
            reaction = db.Reaction(step.get_reaction(), self._reactions)
            self._reaction_enabling_policy.process(reaction)  # type: ignore


class FilteredStepEnabling(StepEnabling):
    """
    Enable elementary steps that parse the given elementary step filter.

    Parameters
    ----------
    step_filter : ElementaryStepFilter
        The filter.
    reaction_enabling_policy : ReactionEnabling
        The reaction enabling policy to be applied to the step's reaction.
    """

    def __init__(self, step_filter: ElementaryStepFilter,
                 reaction_enabling_policy: ReactionEnabling = ReactionEnabling()) -> None:
        super().__init__()
        self._required_collections = ["reactions"]
        self._reaction_enabling_policy = reaction_enabling_policy
        self._filter = step_filter
        assert reaction_enabling_policy

    def initialize_collections(self, manager: db.Manager):
        super().initialize_collections(manager)
        self._filter.initialize_collections(manager)

    def process(self, step: db.ElementaryStep):
        """
        Process an elementary step.

        Parameters
        ----------
        step : db.ElementaryStep
            The elementary step process.
        """
        if self._filter.filter(step):
            self.enable(step)
            if step.has_reaction():
                reaction = db.Reaction(step.get_reaction(), self._reactions)
                self._reaction_enabling_policy.process(reaction)  # type: ignore


class ApplyToAllStepsInReaction(HoldsCollections, ReactionEnabling):
    """
    Apply the given step enabling policy to all steps in the reaction.

    Parameters
    ----------
    step_enabling_policy : StepEnabling
        The step enabling policy.
    """

    def __init__(self, step_enabling_policy: StepEnabling) -> None:
        super().__init__()
        self._step_policy = step_enabling_policy
        self._required_collections = ["elementary_steps"]

    def process(self, reaction: db.Reaction):
        """
        Process a reaction by enabling it and applying the elementary step policy.

        Parameters
        ----------
        reaction : db.Reaction
            The reaction.
        """
        self.enable(reaction)
        for step_id in reaction.get_elementary_steps():
            self._step_policy.process(db.ElementaryStep(step_id, self._elementary_steps))

    def initialize_collections(self, manager: db.Manager):
        self._step_policy.initialize_collections(manager)
        super().initialize_collections(manager)


class ApplyToAllStructuresInAggregate(HoldsCollections, AggregateEnabling):
    """
    Apply a structure enabling policy to all structures of a given aggregate.

    Parameters
    ----------
    structure_enabling_policy : StructureEnabling
        The structure enabling policy.
    """

    def __init__(self, structure_enabling_policy: StructureEnabling) -> None:
        super().__init__()
        self._structure_policy = structure_enabling_policy
        self._required_collections = ["structures"]

    def process(self, aggregate: Union[db.Compound, db.Flask]):
        """
        Process an aggregate by applying the structure enabling policy and enabling the aggregate.
        """
        aggregate.enable_analysis()
        for s_id in aggregate.get_structures():
            self._structure_policy.process(db.Structure(s_id, self._structures))

    def initialize_collections(self, manager: db.Manager):
        self._structure_policy.initialize_collections(manager)
        super().initialize_collections(manager)


class EnableCalculationResults(HoldsCollections, DBObjectEnabling):
    """
    Apply the given step enabling and structure enabling policies to all elementary steps and structures resulting
    from a given calculation.

    Parameters
    ----------
    step_enabling_policy : StepEnabling
        The step enabling policy.
    structure_enabling_policy : StructureEnabling
        The structure enabling policy.
    """

    def __init__(self, step_enabling_policy: StepEnabling = StepEnabling(),
                 structure_enabling_policy: StructureEnabling = StructureEnabling()) -> None:
        super().__init__()
        self._required_collections = ["elementary_steps", "structures", "properties"]
        self._step_enabling_policy = step_enabling_policy
        self._structure_enabling_policy = structure_enabling_policy

    def initialize_collections(self, manager: db.Manager) -> None:
        super().initialize_collections(manager)
        self._step_enabling_policy.initialize_collections(manager)
        self._structure_enabling_policy.initialize_collections(manager)

    def process(self, calculation: db.Calculation):
        """
        Process a calculation by enabling it and applying the given step and structure enabling policies.
        """
        self.enable(calculation)
        if calculation.get_status() == db.Status.COMPLETE:
            results = calculation.get_results()
            for step_id in results.get_elementary_steps():
                step = db.ElementaryStep(step_id, self._elementary_steps)
                self._step_enabling_policy.process(step)
            for s_id in results.get_structures():
                structure = db.Structure(s_id, self._structures)
                self._structure_enabling_policy.process(structure)
            for p_id in results.get_properties():
                prop = db.Property(p_id, self._properties)
                prop.enable_analysis()


class PlaceHolderCalculationEnabling(EnableCalculationResults):
    """
    A place-holder calculation enabling policy that can be used instead of None default arguments and replaced later.
    """

    def process(self, _: db.Calculation):
        raise NotImplementedError


class EnableJobSpecificCalculations(EnableCalculationResults):
    """
    Apply the given step enabling and structure enabling policies to all elementary steps and structures resulting
    from a given calculation with given job order and electronic structure model.

    Parameters
    ----------
    model : db.Model
        The electronic structure model.
    job_order : str
        The job order.
    step_enabling_policy : StepEnabling
        The step enabling policy.
    structure_enabling_policy : StructureEnabling
        The structure enabling policy.
    """

    def __init__(self, model: db.Model, job_order: str, step_enabling_policy: StepEnabling = StepEnabling(),
                 structure_enabling_policy: StructureEnabling = StructureEnabling()) -> None:
        super().__init__(step_enabling_policy, structure_enabling_policy)
        self._required_collections = ["elementary_steps", "structures", "calculations", "properties"]
        self._model = model
        self._job_order = job_order
        self._structure_cache: Set[str] = set()

    def process_calculations_of_structures(self, structure_ids: List[db.ID]) -> None:
        """
        Process all calculations of a given list of structures.

        Parameters
        ----------
        structure_ids : List[db.ID]
            The list of structure ids.
        """
        sorted_structure_ids = ';'.join(sorted([s_id.string() for s_id in structure_ids]))
        if sorted_structure_ids in self._structure_cache:
            return
        calc_str_ids: Set[str] = get_common_calculation_ids(self._job_order, structure_ids, self._model,
                                                            self._structures, self._calculations)
        # Check the clearly tabulated calculations first and stop if they were already analyzed before.
        for str_id in calc_str_ids:
            calculation = db.Calculation(db.ID(str_id), self._calculations)
            structure_str_ids = ';'.join(sorted([s_id.string() for s_id in calculation.get_structures()]))
            if sorted_structure_ids != structure_str_ids or calculation.get_job().order != self._job_order:
                continue
            if calculation.analyze() and calculation.explore():
                return
        # Slow loop over all calculations
        selection = select_calculation_by_structures(self._job_order, structure_ids, self._model)
        for calculation in stop_on_timeout(self._calculations.iterate_calculations(dumps(selection))):
            calculation.link(self._calculations)
            self.process(calculation)
        self._structure_cache.add(sorted_structure_ids)

    def process(self, calculation: db.Calculation):
        """
        Process the calculation.
        """
        if not calculation.exists() or calculation.analyze():
            return
        if calculation.get_job().order == self._job_order and calculation.get_model() == self._model:
            self.enable(calculation)
            super().process(calculation)

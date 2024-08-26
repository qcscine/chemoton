#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""


from typing import List, Union

from scine_chemoton.gears import HoldsCollections, HasName

# Third party imports
import scine_database as db


class DBObjectDisabling(HasName):
    """
    The idea of these disabling functions is to make the gears ignore structures, elementary steps etc. if desired.
    For instance, one could start refinement calculations for elementary steps of a reaction and disables the old steps
    and only rely on the refined steps.
    A second application is to disable all steps/structures etc. in the database and enable these objects in a step-wise
    manner again through the gears if the gears intend to set up the calculation that produced these results. In this
    way an exploration can be simulated without executing any new exploration calculations and new logic for enabling
    compounds for exploration can be tested.
    """

    def __init__(self) -> None:
        super().__init__()
        self._remove_chemoton_from_name()

    @staticmethod
    def disable(db_object: Union[db.Structure, db.Compound, db.Flask, db.ElementaryStep, db.Reaction]):
        """
        Disable the given DB object.

        Parameters
        ----------
        db_object : Union[db.Structure, db.Compound, db.Flask, db.ElementaryStep, db.Reaction]
            The DB object to disable.
        """
        db_object.disable_exploration()
        db_object.disable_analysis()

    def __eq__(self, other):
        return isinstance(other, self.__class__)


class StepDisabling(DBObjectDisabling, HoldsCollections):
    """
    Base class for handling the disabling of elementary steps after, e.g., step-refinement. Does nothing by default.

    The process function should decide on the fate of the given elementary step. As an additional argument a job order
    can be given.
    """

    def process(self, _: db.ElementaryStep, __: str = ""):
        """Do nothing."""
        return


class DisableAllSteps(StepDisabling):
    """
    Disable all elementary steps.

    Parameters
    ----------
    ignore_barrierless : bool
        Whether to ignore barrier-less steps. False by default.
    """

    def __init__(self, ignore_barrierless: bool = False) -> None:
        super().__init__()
        self._ignore_barrierless = ignore_barrierless

    def process(self, step: db.ElementaryStep, __: str = ""):
        """
        Process an elementary step

        Parameters
        ----------
        step : db.ElementaryStep
            The step to process.
        __ : str
            Optional string which may be used by other classes to decide whether disable.
        """
        if self._ignore_barrierless and step.get_type() == db.ElementaryStepType.BARRIERLESS:
            return
        self.disable(step)


class DisableAllStepsByModel(DisableAllSteps):
    """
    Disable all steps that match the given model.

    Parameters
    ----------
    model : db.Model
        The model.
    ignore_barrierless : bool
        Whether to ignore barrier-less steps. False by default.
    """

    def __init__(self, model: db.Model, ignore_barrierless: bool = False) -> None:
        super().__init__(ignore_barrierless)
        self._model = model
        self._required_collections = ["structures"]

    def process(self, step: db.ElementaryStep, __: str = ""):
        """
        Process an elementary step

        Parameters
        ----------
        step : db.ElementaryStep
            The step to process.
        __ : str
            Optional string which may be used by other classes to decide whether disable.
        """
        reactants = step.get_reactants(db.Side.BOTH)
        all_structures = reactants[0] + reactants[1]
        if step.has_transition_state():
            all_structures.append(step.get_transition_state())
        for s_id in all_structures:
            structure = db.Structure(s_id, self._structures)
            if structure.get_model() != self._model:
                return
        super().process(step)


class DisableStepByJob(StepDisabling):
    """
    Disable all elementary step associated to a job order within a given black-list, i.e., the elementary step was
    produced by a job with this order.

    Parameters
    ----------
    black_list : List[str]
        List of job orders for which steps are disabled.
    """

    def __init__(self, black_list: List[str]) -> None:
        super().__init__()
        self._black_list = black_list

    def process(self, step: db.ElementaryStep, job_order: str = ""):
        """
        Process an elementary step

        Parameters
        ----------
        step : db.ElementaryStep
            The step to process.
        job_order : str
            The job order. By default, empty.
        """
        if job_order in self._black_list:
            self.disable(step)


class ReactionDisabling(DBObjectDisabling, HoldsCollections):
    """
    Base class for handling  the disabling of reactions + their elementary steps after, e.g., step-refinement.

    The process function should decide on the fate of the given elementary step. As an additional argument a job order
    can be given.
    Apply a disabling policy to all elementary steps of a reaction.

    Parameters
    ----------
    step_disabling : StepDisabling
        The step disabling policy.
    """

    def __init__(self, step_disabling: StepDisabling = DisableAllSteps()) -> None:
        super().__init__()
        self._required_collections = ["elementary_steps"]
        self._step_disabling = step_disabling

    def initialize_collections(self, manager: db.Manager):
        super().initialize_collections(manager)
        self._step_disabling.initialize_collections(manager)

    def process(self, _: db.Reaction, __: str = ""):
        """Do nothing"""
        return

    def disable_reaction(self, reaction: db.Reaction):
        """
        Apply the step disabling policy. If all steps where disabled, the reaction is also disabled.

        Parameters
        ----------
        reaction : db.Reaction
            The reaction.
        """
        all_steps_disabled = True
        for step_id in reaction.get_elementary_steps():
            step = db.ElementaryStep(step_id, self._elementary_steps)
            self._step_disabling.process(step)
            all_steps_disabled = all_steps_disabled and not step.analyze()
        if all_steps_disabled:
            self.disable(reaction)


class DisableAllReactions(ReactionDisabling):
    """
    Disable all reactions and their steps.

    Parameters
    ----------
    step_disabling : StepDisabling
        The step disabling policy. By default, all steps are disabled.
    """

    def __init__(self, step_disabling: StepDisabling = DisableAllSteps()) -> None:
        super().__init__(step_disabling)

    def process(self, reaction: db.Reaction, __: str = ""):
        """
        Disable all reactions and steps.

        Parameters
        ----------
        reaction : db.Reaction
            The reaction.
        __ : str
            Optional job order. Not used here.
        """
        self.disable_reaction(reaction)


class DisableReactionByJob(ReactionDisabling):
    """
    Disable all reactions associated to a job order within a given black-list, i.e, an elementary step assigned to
    this reaction was produced by a job with the given order.

    Parameters
    ----------
    black_list : List[str]
        The list of job orders.
    """

    def __init__(self, black_list: List[str]) -> None:
        super().__init__()
        self._black_list = black_list

    def process(self, reaction: db.Reaction, job_order: str = ""):
        """
        Disable all reactions and steps.

        Parameters
        ----------
        reaction : db.Reaction
            The reaction.
        job_order : str
            The job order.
        """
        if job_order in self._black_list:
            self.disable_reaction(reaction)


class AggregateDisabling(DBObjectDisabling, HoldsCollections):
    """
    Base class for handling  the disabling of aggregates + their structures.

    The process function should decide on the fate of the given elementary step. As an additional argument a job
    order can be given.
    """

    def __init__(self) -> None:
        super().__init__()
        self._required_collections = ["structures"]

    def process(self, _: Union[db.Compound, db.Flask]):
        """
        Process an aggregate. By default, do nothing.
        """
        return

    def disable_aggregate(self, aggregate: Union[db.Compound, db.Flask]):
        """
        Disable an aggregate and all its structures.

        Parameters
        ----------
        aggregate : Union[db.Compound, db.Flask]
            The aggregate to disable.
        """
        self.disable(aggregate)
        for s_id in aggregate.get_structures():
            structure = db.Structure(s_id, self._structures)
            self.disable(structure)


class DisableAllAggregates(AggregateDisabling):
    """
    Disable all aggregates and their structures.

    Parameters
    ----------
    aggregate : Union[db.Compound, db.Flask]
        The aggregate.
    """

    def process(self, aggregate: Union[db.Compound, db.Flask]):
        self.disable_aggregate(aggregate)

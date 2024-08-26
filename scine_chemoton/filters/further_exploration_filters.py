#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""

from functools import wraps
from json import dumps
from typing import List, Tuple, Union, Dict, Optional, Callable

import scine_utilities as utils
import scine_database as db
from scine_database.queries import (
    stop_on_timeout,
    select_calculation_by_structures,
)
from scine_database.energy_query_functions import (
    get_energy_sum_of_elementary_step_side,
    get_energy_for_structure
)

from scine_chemoton.filters.reactive_site_filters import (
    ReactiveSiteFilter,
    ReactiveSiteFilterAndArray,
    ReactiveSiteFilterOrArray,
)


def single_structure_assertion(fun: Callable):
    """
    Makes sure that the first argument given to the function `fun` is either a list of db.Structure of length 1 or
    a single db.Structure and then calls the function with the list argument, throws TypeError otherwise
    """

    @wraps(fun)
    def _impl(self, structures: Union[List[db.Structure], db.Structure], *args):
        arg = structures
        if isinstance(structures, list):
            if len(structures) != 1:
                raise TypeError(f"The method {fun.__name__} of the filter {self.__class__.__name__} is only supported"
                                f"for single structure lists.")
        elif isinstance(structures, db.Structure):
            arg = [structures]
        else:
            raise TypeError(f"The method {fun.__name__} of the filter {self.__class__.__name__} received an invalid "
                            f"type {type(structures)} for its input argument")
        return fun(self, arg, *args)

    return _impl


class FurtherExplorationFilter(ReactiveSiteFilter):
    """
    Class to evaluate if detailed dissociation exploration trials should be setup.
    This base class does not filter anything out.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)  # required for multiple inheritance
        self._setting_key = 'dissociations'
        self._structure_property_key = 'dissociated_structures'

    def __and__(self, o):
        if not isinstance(o, FurtherExplorationFilter):
            raise TypeError("FurtherExplorationFilter expects FurtherExplorationFilter "
                            "(or derived class) to chain with.")
        return FurtherExplorationFilterAndArray([self, o])

    def __or__(self, o):
        if not isinstance(o, FurtherExplorationFilter):
            raise TypeError("FurtherExplorationFilter expects FurtherExplorationFilter "
                            "(or derived class) to chain with.")
        return FurtherExplorationFilterOrArray([self, o])

    def filter_atoms(self, structure_list: List[db.Structure],
                     atom_indices: List[int]) -> List[int]:
        """
        The blueprint for a filter function, checking a list of atoms
        regarding their reactivity as defined by the filter.

        Parameters
        ----------
        structure_list : List[db.Structure]
            The structures to be checked. Unused in this function.
        atom_indices : [List[int]]
            The list of atoms to consider. If several structures are listed
            atom indices are expected to refer to the entity of all structures
            in the order they are given in the structure list.
            For example, the first atom of the second structure has the index
            equalling the number of atoms of the first structure.

        Returns
        -------
        result : List[int]
            The list of all relevant atom indices after applying the filter.
        """
        return atom_indices

    def filter_atom_pairs(
            self, structure_list: List[db.Structure], pairs: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        The blueprint for a filter function, checking a list of atom pairs
        regarding their reactivity as defined by the filter.

        Parameters
        ----------
        structure_list : List[db.Structure]
            The structures to be checked. Unused in this implementation.
        pairs : List[Tuple[int, int]]
            The list of atom pairs to consider. If several structures are listed
            atom indices are expected to refer to the entity of all structures in
            the order they are given in the structure list.
            For example, the first atom of the second structure has the index
            equalling the number of atoms of the first structure.

        Returns
        -------
        result : List[Tuple[int, int]]
            The list of all relevant reactive atom pairs (given as atom index
            pairs) after applying the filter.
        """
        return pairs

    def filter_reaction_coordinates(
            self, structure_list: List[db.Structure], coordinates: List[List[Tuple[int, int]]]
    ) -> List[List[Tuple[int, int]]]:
        """
        The blueprint for a filter function, checking  a list of trial reaction
        coordinates each given as a tuple of reactive atom pairs for their
        reactivity as defined by the filter.

        Parameters
        ----------
        structure_list : List[db.Structure]
            The structures to be checked. Unused in this implementation.
        coordinates : List[List[Tuple[int, int]]]
            The list of trial reaction coordinates to consider.
            If several structures are listed atom indices are expected to refer
            to the entity of all structures in the order they are given in the
            structure list.
            For example, the first atom of the second structure has the index
            equalling the number of atoms of the first structure.

        Returns
        -------
        result : List[List[Tuple[int, int]]]
            The list of all relevant reaction coordinates given as tuples of
            reactive atom pairs after applying the filter.
        """
        return coordinates

    @staticmethod
    def dissociation_setting_from_reaction_coordinate(coordinate: List[Tuple[int, int]]) -> List[int]:
        """
        Transform given reaction coordinates into the format required in the dissociation setting for the Puffin job.

        Parameters
        ----------
        coordinate : List[Tuple[int, int]]
            reaction coordinate of dissociations

        Returns
        -------
        List[int]
            A flattened list with the dissociation indices after each other
        """
        return [item for pair in coordinate for item in pair]

    def _query_dissociation_energy(self, structure: db.Structure, dissociations_list: List[List[int]], energy_type: str,
                                   model: db.Model, job_order: str) -> List[Union[float, None]]:
        """
        Query the dissociation energy for the given structure and dissociation list.

        Parameters
        ----------
        structure : db.Structure
            The structure that is dissociated
        dissociations_list : List[List[int]]
            A list of possible dissociations
        energy_type : str
            The energy type to query in the properties such as 'electronic energy'
        model : db.Model
            The model
        job_order : str
            The job order that generated the calculations

        Returns
        -------
        List[Union[float, None]]
            A list of dissociation energies in kJ/mol or None if the dissociation was not found
        """
        # prepare data structures to only loop calculations once
        step_dissociation_energies: Dict[str, List[float]] = {}
        plain_dissociation_energies: Dict[str, List[float]] = {}
        for data in [step_dissociation_energies, plain_dissociation_energies]:
            for dissociations in dissociations_list:
                data[str(dissociations)] = []

        selection = select_calculation_by_structures(job_order, [structure.id()], model)
        for calculation in stop_on_timeout(self._calculations.iterate_calculations(dumps(selection))):
            calculation.link(self._calculations)
            dissociation = calculation.get_settings().get(self._setting_key)
            if dissociation is not None and dissociation in dissociations_list:
                results = calculation.get_results()
                if results.elementary_step_ids:
                    for step_id in results.elementary_step_ids:
                        step = db.ElementaryStep(step_id, self._elementary_steps)
                        if not step.explore():
                            continue
                        reactant_energy = get_energy_sum_of_elementary_step_side(step, db.Side.LHS, energy_type,
                                                                                 model, self._structures,
                                                                                 self._properties)
                        if reactant_energy is None:
                            continue
                        product_energy = get_energy_sum_of_elementary_step_side(step, db.Side.RHS, energy_type,
                                                                                model, self._structures,
                                                                                self._properties)
                        if product_energy is None:
                            continue
                        step_dissociation_energies[str(dissociation)].append(product_energy - reactant_energy)
                elif results.property_ids:
                    reactant_energy = get_energy_for_structure(structure, energy_type, model, self._structures,
                                                               self._properties)
                    if reactant_energy is None:
                        continue
                    try:  # if property is not present the db wrapper throws an exception
                        dissocations_property_id = results.get_property(self._structure_property_key, self._properties)
                        dissocations_property = db.StringProperty(dissocations_property_id, self._properties)
                    except BaseException:
                        continue
                    sids = dissocations_property.get_data().split(",")
                    product_energies = []
                    for sid in sids:
                        structure = db.Structure(db.ID(sid), self._structures)
                        product_energies.append(get_energy_for_structure(structure, energy_type, model,
                                                                         self._structures, self._properties))
                    if None in product_energies:
                        continue
                    plain_dissociation_energies[str(dissociation)].append(
                        sum(product_energies) - reactant_energy)  # type: ignore
        energies: List[Union[float, None]] = []
        for dissociation in dissociations_list:
            step_energies = step_dissociation_energies[str(dissociation)]
            if step_energies:
                energies.append(min(step_energies) * utils.KJPERMOL_PER_HARTREE)
                continue
            plain_energies = plain_dissociation_energies[str(dissociation)]
            if plain_energies:
                energies.append(min(plain_energies) * utils.KJPERMOL_PER_HARTREE)
                continue
            energies.append(None)
        return energies


class FurtherExplorationFilterAndArray(FurtherExplorationFilter, ReactiveSiteFilterAndArray):
    """
    An array of logically 'and' connected filters.
    """

    def __init__(self, filters: Optional[List[Union[FurtherExplorationFilter, ReactiveSiteFilter]]] = None) -> None:
        """
        Parameters
        ----------
        filters : Optional[List[Union[FurtherExplorationFilter, ReactiveSiteFilter]]]
            A list of filters to be combined.
        """
        super().__init__(filters=filters)
        assert filters is None or len(self._filters) == len(filters)
        for f in self._filters:
            if not isinstance(f, FurtherExplorationFilter) and not isinstance(f, ReactiveSiteFilter):
                raise TypeError("FurtherExplorationFilterAndArray expects FurtherExplorationFilter "
                                "(or derived class) to chain with.")
        self._join_names(self._filters)


class FurtherExplorationFilterOrArray(FurtherExplorationFilter, ReactiveSiteFilterOrArray):
    """
    An array of logically 'or' connected filters.
    """

    def __init__(self, filters: Optional[List[FurtherExplorationFilter]] = None) -> None:
        """
        Parameters
        ----------
        filters : Optional[List[Union[FurtherExplorationFilter, ReactiveSiteFilter]]]
            A list of filters to be combined.
        """
        super().__init__(filters=filters)
        assert filters is None or len(self._filters) == len(filters)
        for f in self._filters:
            if not isinstance(f, FurtherExplorationFilter) and not isinstance(f, ReactiveSiteFilter):
                raise TypeError("FurtherExplorationFilterOrArray expects FurtherExplorationFilter "
                                "(or derived class) to chain with.")
        self._join_names(self._filters)


class AllBarrierLessDissociationsFilter(FurtherExplorationFilter):
    """
    Filter out all barrierless dissociations.
    """

    def __init__(self, model: db.Model, job_order: str) -> None:
        """
        The model and job that generated the calculations.

        Parameters
        ----------
        model : db.Model
            The model
        job_order : str
            The job order that generated the calculations.
        """
        super().__init__()
        self.model = model
        self.job_order = job_order

    @single_structure_assertion
    def filter_reaction_coordinates(self, structure_list: List[db.Structure],
                                    coordinates: List[List[Tuple[int, int]]]) \
            -> List[List[Tuple[int, int]]]:
        structure = structure_list[0]
        dissociation_lists = [self.dissociation_setting_from_reaction_coordinate(coordinate)
                              for coordinate in coordinates]
        filtered = []
        # search for barrierless steps with structure on LHS
        # first go over reactions to query less
        aggregate_id = structure.get_aggregate()
        if not structure.has_graph("masm_cbor_graph"):
            return []
        aggregate_type = "flask" if ";" in structure.get_graph("masm_cbor_graph") else "compound"
        lhs = {"id": {"$oid": str(aggregate_id)}, "type": aggregate_type}
        selection = {
            "$and": [
                {"exploration_disabled": False},
                {"lhs": {"$size": 1, "$all": [lhs]}},
            ]
        }
        relevant_step_ids = []
        structure_id = structure.id()
        for reaction in self._reactions.query_reactions(dumps(selection)):
            step_ids = reaction.get_elementary_steps()
            for sid in step_ids:
                step = db.ElementaryStep(sid, self._elementary_steps)
                if not step.explore():
                    continue
                if step.get_type() != db.ElementaryStepType.BARRIERLESS:
                    # caveat: this relies on logic in reaction gear that we
                    # disable barrierless elementary steps if there is a regular one for the same reaction
                    continue
                if structure_id in step.get_reactants(db.Side.LHS)[0]:
                    relevant_step_ids.append(sid)
        step_sele = {"results.elementary_steps": {"$in": [{"$oid": str(sid)} for sid in relevant_step_ids]}}
        selection = select_calculation_by_structures(self.job_order, [structure.id()], self.model)
        selection["$and"].append(step_sele)
        for calculation in stop_on_timeout(self._calculations.iterate_calculations(dumps(selection))):
            calculation.link(self._calculations)
            calc_dissociation = calculation.get_settings().get(self._setting_key)
            if calc_dissociation is not None:
                for coordinate, dissociation in zip(coordinates, dissociation_lists):
                    if calc_dissociation == dissociation:
                        filtered.append(coordinate)
        return filtered


class ReactionCoordinateMaxDissociationEnergyFilter(FurtherExplorationFilter):
    """
    Filter out all dissociations with a dissociation energy above a given threshold.
    """

    def __init__(self, max_dissociation_energy: float, energy_type: str, model: db.Model, job_order: str) -> None:
        """
        Specify the limit and electronic structure method to use.

        Parameters
        ----------
        max_dissociation_energy : float
            The maximum dissociation energy in kJ/mol
        energy_type : str
            The energy type to query in the properties such as 'electronic energy'
        model : db.Model
            The model
        job_order : str
            The job order that generated the calculations.
        """
        super().__init__()
        self.max_dissociation_energy = max_dissociation_energy
        self.energy_type = energy_type
        self.model = model
        self.job_order = job_order

    @single_structure_assertion
    def filter_reaction_coordinates(self, structure_list: List[db.Structure],
                                    coordinates: List[List[Tuple[int, int]]]) \
            -> List[List[Tuple[int, int]]]:
        """
        Filter the given structures and reaction coordinates for barrierless dissociations.

        Returns
        -------
        List[List[Tuple[int, int]]]
            The filtered reaction coordinates
        """
        structure = structure_list[0]
        dissociation_lists = [self.dissociation_setting_from_reaction_coordinate(coordinate)
                              for coordinate in coordinates]
        energies = self._query_dissociation_energy(structure, dissociation_lists, self.energy_type, self.model,
                                                   self.job_order)
        if len(energies) != len(coordinates):
            raise RuntimeError("Something went wrong while fetching the dissociation energies for the coordinates "
                               f"{coordinates}")
        filtered = []
        for energy, coordinate in zip(energies, coordinates):
            if energy is None or energy > self.max_dissociation_energy:
                continue
            filtered.append(coordinate)
        return filtered

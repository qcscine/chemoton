#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
.. module:: queries
   :synopsis: Collection of functions that help generating queries
              based on Chemoton specific objects.
"""
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

# Standard library imports
from collections import Counter
from json import dumps
from typing import Any, Dict, List, Optional, Union

# Third party imports
import scine_database as db
import scine_utilities as utils


class stop_on_timeout:
    """
    Iterator class/function that gracefully stops database loops if the
    loop cursor times out.

    Parameters
    ----------
    loop :: Iterator
        The original iterator statement of a DB loop.

    Examples
    --------
    >>> def inner_loop():
    >>>     for i in range(10):
    >>>         if i%2 ==0:
    >>>             yield i
    >>>         else:
    >>>              raise RuntimeError('socket error or timeout , Failed at '+str(i))
    >>>
    >>> for i in stop_on_timeout(inner_loop()):
    >>>     print(i)
    """

    def __init__(self, loop):
        self.loop = loop

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.loop)
        except StopIteration:
            raise StopIteration  # pylint: disable=raise-missing-from
        except RuntimeError as e:
            if "socket error or timeout" in str(e):
                raise StopIteration  # pylint: disable=raise-missing-from
            else:
                raise e


def model_query(model: db.Model) -> List[dict]:
    """
    Generates a query that fits the given model, meaning that any field
    in the model given as 'any' will not be queried, while all other fields
    must match.

    Parameters
    ----------
    model :: Scine::Database::Model
        The model for which a query list will be generated.

    Returns
    -------
    query :: List[dict]
        A list of queries for each element of the Model class.
        The list can be added to any '$and' or '$or' expression.

    Examples
    --------
    >>> selection = {'$and': [
    >>>     {'some': 'logic'},
    >>>     {'more': 'logic'}
    >>> ] + model_query(model)
    >>> }
    >>> for s in collection.iterate_structures(dumps(selection)):
    >>>     pass

    """
    result = []
    fields = ['spin_mode', 'basis_set', 'method', 'method_family', 'program', 'version', 'solvation', 'solvent',
              'embedding', 'periodic_boundaries', 'external_field', 'temperature', 'electronic_temperature']
    for field in fields:
        value = getattr(model, field)
        if value.lower() != "any":
            result.append({f"model.{field}": value})
    return result


def identical_reaction(
    lhs_aggregates: List[db.ID], rhs_aggregates: List[db.ID], lhs_types: List[db.CompoundOrFlask],
    rhs_types: List[db.CompoundOrFlask], reactions: db.Collection
) -> Union[db.Reaction, None]:
    """
    Searches for a reaction with the same aggregates, forward and backward reactions
    are categorized as the same reaction.

    Parameters
    ----------
    lhs_aggregates :: List[db.ID]
        The ids of the aggregates of the left hand side
    rhs_aggregates :: List[db.ID]
        The ids of the aggregates of the right hand side
    lhs_types :: List[db.ID]
        The types of the LHS aggregates.
    rhs_types :: List[db.ID]
        The types of the RHS aggregates.
    reactions :: db.Collection (Scine::Database::Collection)

    Returns
    -------
    reaction :: Union[db.Reaction, None]
        The identical reaction or None if no identical reaction found in the collection
    """

    lhs_list = []
    for i, j in zip(lhs_aggregates, lhs_types):
        a_type = "flask" if j == db.CompoundOrFlask.FLASK else "compound"
        lhs_list.append({"id": {"$oid": i.string()}, "type": a_type})
    rhs_list = []
    for i, j in zip(rhs_aggregates, rhs_types):
        a_type = "flask" if j == db.CompoundOrFlask.FLASK else "compound"
        rhs_list.append({"id": {"$oid": i.string()}, "type": a_type})
    selection = {
        "$or": [
            {
                "$and": [
                    {"lhs": {"$size": len(rhs_list), "$all": rhs_list}},
                    {"rhs": {"$size": len(lhs_list), "$all": lhs_list}},
                ]
            },
            {
                "$and": [
                    {"lhs": {"$size": len(lhs_list), "$all": lhs_list}},
                    {"rhs": {"$size": len(rhs_list), "$all": rhs_list}},
                ]
            },
        ]
    }
    for hit in reactions.query_reactions(dumps(selection)):
        if _verify_identical_reaction(lhs_aggregates, rhs_aggregates, hit) or _verify_identical_reaction(
            rhs_aggregates, lhs_aggregates, hit
        ):
            return hit
    return None


def _verify_identical_reaction(
    lhs_aggregates: List[db.ID], rhs_aggregates: List[db.ID], possible_reaction: db.Reaction
) -> bool:
    test_reactants = possible_reaction.get_reactants(db.Side.BOTH)
    test_lhs = test_reactants[0]
    test_rhs = test_reactants[1]
    return Counter([x.string() for x in lhs_aggregates]) == Counter([x.string() for x in test_lhs]) and Counter(
        [x.string() for x in rhs_aggregates]
    ) == Counter([x.string() for x in test_rhs])


def stationary_points() -> dict:
    """
    Setup query for 1) optimized structures linked to an aggregate and 2) transition states
    """
    selection = {
        "$or": [
            {"label": "ts_optimized"},
            {
                "$and": [
                    {
                        "$or": [
                            {"label": "minimum_optimized"},
                            {"label": "user_optimized"},
                            {"label": "complex_optimized"},
                        ]
                    },
                    {"aggregate": {"$ne": ""}},
                    {"exploration_disabled": {"$ne": True}},
                ]
            },
        ]
    }
    return selection


def select_calculation_by_structures(job_order: str, structure_id_list: List[db.ID], model: db.Model) -> dict:
    """
    Sets up a query for calculations with a specific job order and model working
    on all of the given structures irrespective of their ordering.

    Parameters
    ----------
    job_order : str
        The job order of the calculations to consider.
    structure_id_list : List[db.ID]
        The list of structure ids of interest.
    model : db.Model
        The model the calculations shall use.

    Returns
    -------
    dict
        The selection query dictionary.
    """
    struct_oids = [{"$oid": sid.string()} for sid in structure_id_list]
    selection = {
        "$and": [
            {"job.order": job_order},
            {"structures": {"$size": len(struct_oids), "$all": struct_oids}},
            *model_query(model)
        ]
    }
    return selection


def calculation_exists_in_structure(job_order: str, structure_id_list: List[db.ID], model: db.Model,
                                    structures: db.Collection, calculations: db.Collection,
                                    settings: Optional[Dict[str, Any]] = None,
                                    auxiliaries: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if a calculation exists that corresponds to the given structures, mode, settings, etc.

    Parameters
    ----------
    job_order : str
        The job order of the calculations to consider.
    structure_id_list : List[db.ID]
        The list of structure ids of interest.
    model : db.Model
        The model the calculations shall use.
    structures : db.Collection
        The structure collection.
    calculations : db.Collection
        The calculation collection.
    settings : dict (optional)
        The settings of the calculation.
    auxiliaries : dict (optional)
        The auxiliaries of the calculation.

    Returns
    -------
    True, if such a calculation exists. False, otherwise.

    """
    return get_calculation_id_from_structure(job_order, structure_id_list, model, structures,
                                             calculations, settings, auxiliaries) is not None


def get_calculation_id_from_structure(job_order: str, structure_id_list: List[db.ID], model: db.Model,
                                      structures: db.Collection, calculations: db.Collection,
                                      settings: Optional[Union[utils.ValueCollection, Dict[str, Any]]] = None,
                                      auxiliaries: Optional[Dict[str, Any]] = None) -> Union[db.ID, None]:
    """
    Search for a calculation corresponding to the given settings. If the calculation is found, its ID is returned.

    Parameters
    ----------
    job_order : str
        The job order of the calculations to consider.
    structure_id_list : List[db.ID]
        The list of structure ids of interest.
    model : db.Model
        The model the calculations shall use.
    structures : db.Collection
        The structure collection.
    calculations : db.Collection
        The calculation collection.
    settings : dict (optional)
        The settings of the calculation.
    auxiliaries : dict (optional)
        The auxiliaries of the calculation.

    Returns
    -------
    Returns the calculation ID if found. Returns None if no calculation corresponds to the given specification.

    """
    if len(structure_id_list) < 1:
        return None
    # settings type check, support both dict and ValueCollection and want ValueCollection for speed
    compare_settings = None
    if settings is not None:
        if isinstance(settings, utils.ValueCollection):
            compare_settings = settings
        elif isinstance(settings, dict):
            compare_settings = utils.ValueCollection(settings)
        else:
            raise TypeError(f"Gave incompatible type '{type(settings)}' to 'get_calculation_id_from_structure'")
    structure_0 = db.Structure(structure_id_list[0], structures)
    calc_id_set = set([c_id.string() for c_id in structure_0.query_calculations(job_order, model, calculations)])
    if not calc_id_set:
        return None
    for s_id in structure_id_list:
        structure = db.Structure(s_id, structures)
        struc_set = set([c_id.string() for c_id in structure.query_calculations(job_order, model, calculations)])
        calc_id_set = calc_id_set.intersection(struc_set)
    calc_id_str = [{"$oid": str_id} for str_id in calc_id_set]
    selection = {
        "$and": [{"_id": {"$in": calc_id_str}}]
    }
    for calculation in stop_on_timeout(calculations.iterate_calculations(dumps(selection))):
        calculation.link(calculations)
        structures_in_calc_ids = calculation.get_structures()
        if len(structure_id_list) != len(structures_in_calc_ids):
            continue
        # Check structure ids
        matching_structures = True
        for s_id in structures_in_calc_ids:
            if s_id not in structure_id_list:
                matching_structures = False
                break
        if not matching_structures:
            continue
        if compare_settings is not None:
            if compare_settings != calculation.get_settings():
                continue
        if auxiliaries is not None:
            if auxiliaries != calculation.get_auxiliaries():
                continue
        return calculation.id()
    return None


def get_calculation_id(job_order: str, structure_id_list: List[db.ID], model: db.Model,
                       calculations: db.Collection,
                       settings: Optional[Union[utils.ValueCollection, Dict[str, Any]]] = None,
                       auxiliaries: Optional[Dict[str, Any]] = None) -> Union[db.ID, None]:
    if len(structure_id_list) < 1:
        return None
    selection = select_calculation_by_structures(job_order, structure_id_list, model)
    # simple case of no required loop comparisons
    if settings is None and auxiliaries is None:
        hit = calculations.get_one_calculation(dumps(selection))
        if hit is not None:
            return hit.id()
        return None
    # settings type check, support both dict and ValueCollection and want ValueCollection for speed
    compare_settings = None
    if settings is not None:
        if isinstance(settings, utils.ValueCollection):
            compare_settings = settings
        elif isinstance(settings, dict):
            compare_settings = utils.ValueCollection(settings)
        else:
            raise TypeError(f"Gave incompatible type '{type(settings)}' to 'get_calculation_id'")

    for calculation in stop_on_timeout(calculations.iterate_calculations(dumps(selection))):
        calculation.link(calculations)
        if compare_settings is not None:
            if compare_settings != calculation.get_settings():
                continue
        if auxiliaries is not None:
            if auxiliaries != calculation.get_auxiliaries():
                continue
        return calculation.id()
    return None

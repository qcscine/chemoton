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
from typing import List, Union

# Third party imports
import scine_database as db


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
            raise StopIteration
        except RuntimeError as e:
            if "socket error or timeout" in str(e):
                raise StopIteration
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
    lhs_compounds: List[db.ID], rhs_compounds: List[db.ID], reactions: db.Collection
) -> Union[db.Reaction, None]:
    """
    Searches for a reaction with the same compounds, forward and backward reactions
    are categorized as the same reaction.

    Parameters
    ----------
    lhs_compounds :: List[db.ID]
        The ids of the compounds of the left hand side
    rhs_compounds :: List[db.ID]
        The ids of the compounds of the right hand side
    reactions :: db.Collection (Scine::Database::Collection)

    Returns
    -------
    reaction :: Union[db.Reaction, None]
        The identical reaction or None if no identical reaction found in the collection
    """
    lhs_oids = [{"$oid": i.string()} for i in lhs_compounds]
    rhs_oids = [{"$oid": i.string()} for i in rhs_compounds]
    selection = {
        "$or": [
            {
                "$and": [
                    {"lhs": {"$size": len(rhs_oids), "$all": rhs_oids}},
                    {"rhs": {"$size": len(lhs_oids), "$all": lhs_oids}},
                ]
            },
            {
                "$and": [
                    {"lhs": {"$size": len(lhs_oids), "$all": lhs_oids}},
                    {"rhs": {"$size": len(rhs_oids), "$all": rhs_oids}},
                ]
            },
        ]
    }
    hits = reactions.query_reactions(dumps(selection))
    for hit in hits:
        hit.link(reactions)
        if _verify_identical_reaction(lhs_compounds, rhs_compounds, hit) or _verify_identical_reaction(
            rhs_compounds, lhs_compounds, hit
        ):
            return hit
    return None


def _verify_identical_reaction(
    lhs_compounds: List[db.ID], rhs_compounds: List[db.ID], possible_reaction: db.Reaction
) -> bool:
    test_reactants = possible_reaction.get_reactants(db.Side.BOTH)
    test_lhs = test_reactants[0]
    test_rhs = test_reactants[1]
    return Counter([x.string() for x in lhs_compounds]) == Counter([x.string() for x in test_lhs]) and Counter(
        [x.string() for x in rhs_compounds]
    ) == Counter([x.string() for x in test_rhs])


def stationary_points() -> dict:
    """
    Setup query for optimized structures linked to a compound and transition states
    """
    selection = {
        "$or": [
            {
                "$and": [
                    {
                        "$or": [
                            {"label": {"$eq": "minimum_optimized"}},
                            {"label": {"$eq": "user_optimized"}},
                        ]
                    },
                    {"compound": {"$ne": ""}},
                    {"exploration_disabled": {"$ne": True}},
                ]
            },
            {"label": {"$eq": "ts_optimized"}},
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
    struct_oids = [{"$oid": id.string()} for id in structure_id_list]
    selection = {
        "$and": [{"job.order": {"$eq": job_order}}, {"structures": {"$size": len(struct_oids), "$all": struct_oids}}]
        + model_query(model)
    }
    return selection

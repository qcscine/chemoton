#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""

from typing import Union

import scine_database as db


def get_compound_or_flask(object_id: db.ID, object_type: db.CompoundOrFlask, compounds: db.Collection,
                          flasks: db.Collection) -> Union[db.Compound, db.Flask]:
    """
    Construct the compound or flask object depending on the type. Flask and Compound have a large set of functions
    in common. Through this function, we do not have to differentiate between them at every point.

    Parameters
    ----------
    object_id :: db.ID
        The ID of the object to construct.
    object_type :: db.CompoundOrFlask
        The label for Compound or Flaks.
    compounds :: db.Collection
        The compounds collection.
    flasks :: db.Collection
        The flasks collection.

    Returns
    -------
    Either the flask or compound object.

    Note
    ----
    Raises a runtime error if the object_type is unknown.
    """
    if object_type == db.CompoundOrFlask.COMPOUND:
        return db.Compound(object_id, compounds)
    if object_type == db.CompoundOrFlask.FLASK:
        return db.Flask(object_id, flasks)
    raise RuntimeError("Requested aggregate type is not supported.")


def get_aggregate_type(structure: db.Structure) -> db.CompoundOrFlask:
    """
    Resolve the aggregate label from the structure label
    Parameters
    ----------
    structure :: db.Structure
        The structure.
    Returns
    -------
    db.CompoundOrFlask
        The aggregate type.

    """
    aggregate_type = db.CompoundOrFlask.COMPOUND
    if structure.get_label() in [db.Label.COMPLEX_OPTIMIZED, db.Label.COMPLEX_GUESS]:
        aggregate_type = db.CompoundOrFlask.FLASK
    return aggregate_type

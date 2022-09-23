#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Laboratory of Physical Chemistry, Reiher Group.
See LICENSE.txt for details.
"""
# from typing import Union
import scine_database as db
import scine_utilities as utils


def get_molecular_formula_of_structure(structure_id: db.ID, structures: db.Collection) -> str:
    """
    Get the molecular formula of a given structure, its charge ("c") and its multiplicity ("m") as a string.

    Parameters
    ----------
    structure_id :: db.ID
        The database ID of the structure.
    structures :: db.Collection
        The structure collection.

    Returns
    -------
    mform :: str
        The molecular formula of the given structure, e.g. for water "H2O (c:0, m:1)".
    """
    struct = db.Structure(structure_id, structures)

    atoms = struct.get_atoms()
    # # # Add chemical formula
    mform = utils.generate_chemical_formula(atoms.elements)
    # # # Add charge
    mform += '(c:' + str(struct.get_charge())
    # # # Add multiplicity
    mform += ', m:' + str(struct.get_multiplicity()) + ')'

    return mform


def get_molecular_formula_of_aggregate(object_id: db.ID, object_type: db.CompoundOrFlask, compounds: db.Collection,
                                       flasks: db.Collection, structures: db.Collection) -> str:
    """
    Get the molecular formula of an aggregate, its charge ("c") and its multiplicity ("m") as a string.

    Parameters
    ----------
    object_id :: db.ID
        The database ID of the aggregate.
    object_type :: db.CompoundOrFlask
        The aggregate type (db.CompoundOrFlask.COMPOUND or db.CompoundOrFlask.FLASK).
    compounds :: db.Collection
        The compound collection.
    flasks :: db.Collection
        The flask collection
    structures :: db.Collection
        The structure collection.

    Returns
    -------
    molecular formula :: str
        The molecular formula of the given aggregate according to the aggregates type.

    """
    if object_type == db.CompoundOrFlask.COMPOUND:
        molecular_formula = get_molecular_formula_of_compound(object_id, compounds, structures)
    elif object_type == db.CompoundOrFlask.FLASK:
        molecular_formula = get_molecular_formula_of_flask(object_id, flasks, compounds, structures)

    return molecular_formula


def get_molecular_formula_of_compound(compound_id: db.ID, compounds: db.Collection, structures: db.Collection) -> str:
    """
    Get the molecular formula of a given compound, its charge ("c") and its multiplicity ("m") as a string.

    Parameters
    ----------
    compound_id :: db.ID
        The database ID of the compound.
    compounds :: db.Collection
        The compound collection.
    structures :: db.Collection
        The structure collection.

    Returns
    -------
    str
        The molecular formula of the given compound, e.g. for water "H2O (c:0, m:1)".
    """
    compound = db.Compound(compound_id, compounds)
    centroid_id = compound.get_centroid()
    return get_molecular_formula_of_structure(centroid_id, structures)


def get_molecular_formula_of_flask(
        flask_id: db.ID,
        flasks: db.Collection,
        compounds: db.Collection,
        structures: db.Collection) -> str:
    """
    Get the molecular formula of a given flask, its charge ("c") and its multiplicity ("m") and the identical
    information of the compounds the flask consists of as a string.

    Parameters
    ----------
    flask_id : db.ID
        The database ID of the flask.
    flasks : db.Collection
        The flask collection.
    compounds : db.Collection
        The compound collection.
    structures : db.Collection
        The structure collection.

    Returns
    -------
    str
        The molecular formula of the given flask and the molecular formulas of the compounds, e.g. for the water dimer
        "H4O2 (c:0, m:1) [H2O (c:0, m:1)|H2O (c:0, m:1)]".
    """
    flask = db.Flask(flask_id, flasks)
    centroid_id = flask.get_centroid()
    molecular_formula = get_molecular_formula_of_structure(centroid_id, structures) + " ["
    for compound_id in flask.get_compounds():
        molecular_formula += get_molecular_formula_of_compound(compound_id, compounds, structures)
        molecular_formula += "|"

    molecular_formula = molecular_formula[:-1] + "]"

    return molecular_formula

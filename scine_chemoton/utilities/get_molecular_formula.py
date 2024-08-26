#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__copyright__ = """ This code is licensed under the 3-clause BSD license.
Copyright ETH Zurich, Department of Chemistry and Applied Biosciences, Reiher Group.
See LICENSE.txt for details.
"""
from typing import Dict
from copy import deepcopy

import scine_database as db
import scine_utilities as utils
import scine_molassembler as masm


def get_elements_in_structure(structure: db.Structure) -> Dict[str, int]:
    """
    Get the number of atoms sorted by element for the given structure.

    Parameters
    ----------
    structure : db.Structure
        The structure.
    Returns
    -------
    Dict[str: int]
        A dictionary containing the element symbol and the number of occurrences.
    """
    elements = [str(e) for e in structure.get_atoms().elements]
    return {e: elements.count(e) for e in elements}


def combine_element_counts(counts_one: Dict[str, int], counts_two: Dict[str, int]) -> Dict[str, int]:
    """
    Combine two dictionaries with element counts.

    Parameters
    ----------
    counts_one : Dict[str, int]
        A dictionary with element counts. The keys are the element symbols. The values are the number of occurrences.
    counts_two : Dict[str, int]
        The second dictionary.

    Returns
    -------
    Dict[str, int]
        The combined dictionary.
    """
    total_counts: Dict[str, int] = deepcopy(counts_one)
    for key, value in counts_two.items():
        if key in total_counts:
            total_counts[key] += value
        else:
            total_counts[key] = value
    return total_counts


def get_molecular_formula_of_structure(structure_id: db.ID, structures: db.Collection) -> str:
    """
    Get the molecular formula of a given structure, its charge ("c") and its multiplicity ("m") as a string.

    Parameters
    ----------
    structure_id : db.ID
        The database ID of the structure.
    structures : db.Collection
        The structure collection.

    Returns
    -------
    mform : str
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
    object_id : db.ID
        The database ID of the aggregate.
    object_type : db.CompoundOrFlask
        The aggregate type (db.CompoundOrFlask.COMPOUND or db.CompoundOrFlask.FLASK).
    compounds : db.Collection
        The compound collection.
    flasks : db.Collection
        The flask collection
    structures : db.Collection
        The structure collection.

    Returns
    -------
    molecular formula : str
        The molecular formula of the given aggregate according to the aggregates type.

    """
    if object_type == db.CompoundOrFlask.COMPOUND:
        molecular_formula = get_molecular_formula_of_compound(object_id, compounds, structures)
    elif object_type == db.CompoundOrFlask.FLASK:
        molecular_formula = get_molecular_formula_of_flask(object_id, flasks, structures)
    else:
        raise RuntimeError('Invalid object type. Expected COMPOUND or FLASK.')

    return molecular_formula


def get_molecular_formula_of_compound(compound_id: db.ID, compounds: db.Collection, structures: db.Collection) -> str:
    """
    Get the molecular formula of a given compound, its charge ("c") and its multiplicity ("m") as a string.

    Parameters
    ----------
    compound_id : db.ID
        The database ID of the compound.
    compounds : db.Collection
        The compound collection.
    structures : db.Collection
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
    structures : db.Collection
        The structure collection.

    Returns
    -------
    str
        The molecular formula of the given flask and the molecular formulas of the compounds according to the CBOR
        graph of its structure, e.g. for the water dimer "H4O2 (c:0, m:1) [H2O | H2O ]".
    """
    flask = db.Flask(flask_id, flasks)
    centroid_id = flask.get_centroid()
    molecular_formula = get_molecular_formula_of_structure(centroid_id, structures) + " ["
    centroid = db.Structure(centroid_id, structures)
    # Loop over individual masm cbor graphs
    for cmp_cbor_graph in centroid.get_graph("masm_cbor_graph").split(";"):
        molecular_formula += get_molecular_formula_from_cbor_string(cmp_cbor_graph)
        molecular_formula += " | "
    molecular_formula = molecular_formula[:-3] + "]"

    return molecular_formula


def get_molecular_formula_from_cbor_string(cbor_graph: str) -> str:
    """
    Get the molecular formula of a given CBOR Graph.

    Parameters
    ----------
    cbor_graph : str
        The string of a CBOR graph.

    Returns
    -------
    str
        The plain molecular formula of the given CBOR graph.
    """
    binary = masm.JsonSerialization.base_64_decode(cbor_graph)
    serialization = masm.JsonSerialization(binary, masm.JsonSerialization.BinaryFormat.CBOR)
    masm_molecule = serialization.to_molecule()
    molecular_formula = utils.generate_chemical_formula([masm_molecule.graph[i] for i in masm_molecule.graph.atoms()])

    return molecular_formula
